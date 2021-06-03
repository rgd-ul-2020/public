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
from pesqnet import QNet, LivePlot, pesq

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
    model.compile(optimizer=k.optimizers.RMSprop(1e-5), loss='mae')
    model.summary()

    return model


def GAN(input_dim, latent_dim):
    gen  = CVAE(input_dim, latent_dim)
    crit = QNet()
    crit.trainable = False

    inpt = kl.Input((1, input_dim, 1))
    prev = gen(inpt)
    prev = crit([ kl.Reshape((input_dim, 1))(inpt), kl.Reshape((input_dim, 1))(prev) ])
    
    model = k.Model(inpt, prev)
    model.compile(optimizer=k.optimizers.RMSprop(1e-5), loss='mse') 

    return model, gen, crit

def from_block(block, padding=None):
    n_block = len(block)
    block   = np.pad(block, (0, 16000 - n_block), 'constant')
    block   = (block + 1.) / 2.
    block   = np.reshape(block, (100, 1, 160, 1))
    block   = tf.cast(block, tf.float16)
    return block

def to_block(code):
    code = np.reshape(code, (16000,))
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
        block_gen = next_file(files, blocksize=blocksize)
        block = next(block_gen)
    return block

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

plot  = LivePlot((800, 600))

model, gen, crit = GAN(INPUT_DIM, LATENT_DIM)

sum_tloss = 0
cnt_tloss = 0

sum_vloss = 0
cnt_vloss = 0

patience  = 10000
best_loss = float('inf')
no_improvement = 0

loss_hist = []
avg_hist  = []

class EarlyStop(Exception): pass

try:
    factors = [ 1, 0.85, 0.75, 0.625, 0.3215, 0.015625, 0.0078125, 0.005, 0.003, 0.001, 0 ]

    epoch_range = trange(10000)
    for e in epoch_range:
        batch_range = trange(100)
        train_gen   = None
        valid_gen   = None

        for b in batch_range:
            good  = from_block(next_block(train_files, train_gen, 16000))
            goodl = good[:-1,:,:,:]
            goodp = good[-1:,:,:,:]
            #valid_block = next_block(valid_files, valid_gen, INPUT_DIM)

            noise  = gen.predict_on_batch(good)
            noisel = noise[:-1,:,:,:]
            noisep = noise[-1:,:,:,:]

            score = pesq(good, noise) / 5

            goodl3  = np.reshape(goodl,  (99, 160, 1))
            goodp3  = np.reshape(goodp,  (1,  160, 1))
            noisel3 = np.reshape(noisel, (99, 160, 1))
            noisep3 = np.reshape(noisep, (1,  160, 1))

            crit.reset_states()
            _     = crit.test_on_batch([ goodl, goodl ], 99 * [1])
            rloss = crit.train_on_batch([ goodp, goodp ], [1])

            crit.reset_states()
            _     = crit.test_on_batch([ goodl, noisel ], 99 * [0])
            floss = crit.train_on_batch([ goodp, noisep ], [0])

            dloss = 0.5 * np.add(rloss, floss)

            model.reset_states()
            _     = model.test_on_batch([ goodl, noisel ], 99 * [1])
            gloss = model.train_on_batch([ goodp, noisep ], [1])

            loss = dloss

            if loss < best_loss:
                best_loss = loss

            loss_hist.append(loss)
            loss_hist = loss_hist[-800:]

            avg_hist.append(np.mean(loss_hist))
            avg_hist = avg_hist[-800:]
    
            if not plot.update([ loss_hist, avg_hist ], (best_loss, np.max(loss_hist))):
                raise EarlyStop()

        if e % 100 == 0:
            with open('result_{:05}.wav'.format(e), 'wb+') as out_codec:
                output = []
                for block in sf.blocks(random.choice(speech_files), blocksize=INPUT_DIM, overlap=0):
                    block = from_block(block)
                    block = gen.predict(block)
                    block = to_block(block)
                    output.extend(block)
                sf.write(out_codec, output, 16000)
except EarlyStop:
    print('No improvements, early stop!')
    

with open('result_99999.wav'.format(e), 'wb+') as out_codec:
    output = []
    for block in sf.blocks(random.choice(speech_files), blocksize=INPUT_DIM, overlap=0):
        block = from_block(block)
        block = gen.predict(block)
        block = to_block(block)
        output.extend(block)
    sf.write(out_codec, output, 16000)
