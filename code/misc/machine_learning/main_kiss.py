#!/usr/bin/python3

import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as K
import tensorflow.keras.layers as kl
import soundfile as sf

from tqdm import trange

tf.enable_eager_execution()

def inference_net(input_dim, latent_dim):
    inpt = kl.Input((1, input_dim, 1))
    prev = inpt
    prev = kl.Conv2D(filters=32, kernel_size=(1, 5), strides=(1, 3), padding='same', activation='relu')(prev)
    prev = kl.Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu')(prev)
    prev = kl.Conv2D(filters=1,  kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu')(prev)
    return k.Model(inpt, prev, name='inference_net')

def generative_net(latent_dim, reconstruction_dim):
    inpt = kl.Input((1, latent_dim, 1))
    prev = inpt
    prev = kl.Conv2DTranspose(filters=32, kernel_size=(1, 3), strides=(1, 2), padding='same', output_padding=(0,0), activation='relu')(prev)
    prev = kl.Conv2DTranspose(filters=64, kernel_size=(1, 3), strides=(1, 2), padding='same', activation='relu')(prev)
    prev = kl.Conv2DTranspose(filters=1,  kernel_size=(1, 5), strides=(1, 3), padding='same', output_padding=(0,0), activation='relu')(prev)
    return k.Model(inpt, prev, name='generative_net')

def CVAE(input_dim, latent_dim):
    inf = inference_net(input_dim, latent_dim)
    inf.summary()
    gen = generative_net(14, input_dim)#latent_dim, input_dim)
    gen.summary()

    inpt = kl.Input((1, input_dim, 1))
    enc  = inf(inpt)
    dec  = gen(enc)

    model = k.Model(inpt, dec, name='cvae')
    model.compile(optimizer=k.optimizers.RMSprop(1e-5), loss='mse')
    """
    def elbo_loss(x, mu, sigma):
        def loss(y_true, y_pred):
            y = tf.sigmoid(y_pred)
            marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
            ELBO = tf.reduce_mean(marginal_likelihood) - tf.reduce_mean(KL_divergence)
            return -ELBO
        return loss

    model.compile(optimizer=k.optimizers.RMSprop(1e-2), loss=elbo_loss(inpt, mean, stdv))
    """
    model.summary()

    return model


def GAN(input_dim, latent_dim):
    gen = CVAE(input_dim, latent_dim)


def from_block(block, padding=None):
    if padding is None:
        padding = INPUT_DIM
    block   = tf.cast(block, tf.float16)
    n_block = len(block)
    block   = np.pad(block, (0, padding - n_block), 'constant')
    block   = (block + 1.) / 2.
    block   = np.reshape(block, (1, 1, padding, 1))
    return block

def to_block(code):
    code = np.reshape(code, (INPUT_DIM,))
    code = (code * 2.) - 1.
    return code

def next_file(flist, blocksize=1024, dur=3):
    fname   = random.choice(flist)
    info    = sf.info(fname)
    segsize = dur * info.samplerate
    start = random.randint(0, max(0, info.frames - segsize))
    return sf.blocks(fname, blocksize=blocksize, start=start, stop=start + segsize, overlap=0)

def next_block(files, block_gen, blocksize):
    try:
        block = next(block_gen)
    except:
        block_gen = next_file(files, blocksize=INPUT_DIM)
        block = next(block_gen)
    return block

class loss_history:
    def __init__(self):
        self.last_sample = 0.
        self.spl_list    = []

    def sample(self, s):
        self.last_sample = s
        self.spl_list.append(s)
        self.spl_list = self.spl_list[-1000:]

    def __str__(self):
        return '{:+.05e}({:+.05e})'.format(self.last_sample, np.mean(self.spl_list))

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

speech_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]
random.shuffle(speech_files)
train_files  = speech_files[:5]
valid_files  = speech_files[5:10]

INPUT_DIM  = 160
LATENT_DIM = INPUT_DIM // 10

tlosses = list(np.linspace(0, 0, 1000))
vlosses = list(np.linspace(0, 0, 1000))
           
ilosses = np.linspace(0, 1000, 1000)
i = 0

import wx
import wxmplot
import threading

app = wx.App()
frame = wxmplot.PlotFrame()

frame.plot(ilosses, tlosses, color='red')
frame.oplot(ilosses, vlosses, color='darkgreen')
frame.Show()
threading.Thread(target=app.MainLoop).start()

model  = CVAE(INPUT_DIM, LATENT_DIM)

sum_tloss = 0
cnt_tloss = 0

sum_vloss = 0
cnt_vloss = 0

patience  = 10000
best_loss = 1.
no_improvement = 0

class EarlyStop(Exception): pass

try:
    epoch_range = trange(10000)
    for e in epoch_range:
        model.reset_states()

        batch_range = trange(100)
        train_gen   = None
        valid_gen   = None

        for b in batch_range:
            train_block = next_block(train_files, train_gen, INPUT_DIM)
            valid_block = next_block(valid_files, valid_gen, INPUT_DIM)

            volume = min(0.5, random.random())
            noise  = min(0.,  max(0.1, random.random()))

            x       = model.predict(from_block(train_block))
            x_noise = from_block(volume * train_block + tf.random.normal(train_block.shape, 0., noise))
            model.train_on_batch(x_noise, x)

            x       = model.predict(model.predict(from_block(train_block)))
            x_noise = from_block(volume * train_block + tf.random.normal(train_block.shape, 0., noise))
            model.train_on_batch(x_noise, x)

            x       = from_block(train_block)
            x_noise = from_block(volume * train_block + tf.random.normal(train_block.shape, 0., noise))
            sum_tloss += model.train_on_batch(x_noise, x)
            cnt_tloss += 1
            avg_tloss  = sum_tloss / cnt_tloss

            if avg_tloss < best_loss:
                best_loss = avg_tloss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= patience:
                raise EarlyStop()

            tlosses.append(avg_tloss)
            tlosses = tlosses[-1000:]

            x       = from_block(valid_block)
            x_noise = from_block(volume * valid_block + tf.random.normal(valid_block.shape, 0., noise))
            sum_vloss += model.train_on_batch(x_noise, x)
            cnt_vloss += 1

            vlosses.append(sum_vloss / cnt_vloss)
            vlosses = vlosses[-1000:]

        frame.clear()
        frame.plot(ilosses, tlosses, color='red')
        frame.oplot(ilosses, vlosses, color='darkgreen')
        frame.Refresh()

        if e % 100 >= 0:
            with open('result_{:05}.wav'.format(e), 'wb+') as out_codec:
                output = []
                for block in sf.blocks(random.choice(speech_files), blocksize=INPUT_DIM, overlap=0):
                    block = from_block(block)
                    block = model.predict(block)
                    block = to_block(block)
                    output.extend(block)
                sf.write(out_codec, output, 16000)
except EarlyStop:
    print('No improvements, early stop!')
    

with open('result_99999.wav'.format(e), 'wb+') as out_codec:
    output = []
    for block in sf.blocks(random.choice(speech_files), blocksize=INPUT_DIM, overlap=0):
        block = from_block(block)
        block = model.predict(block)
        block = to_block(block)
        output.extend(block)
    sf.write(out_codec, output, 16000)
