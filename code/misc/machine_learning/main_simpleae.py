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

def sampling_layer(mean, logvar, latent_dim):
    def reparametrization(args):
        mean, logvar = args
        eps = tf.random.normal((latent_dim,), 0., 1.)
        return mean + tf.exp(logvar / 2.) * eps
    return kl.Lambda(reparametrization, output_shape=(1, latent_dim,))([mean, logvar])

def upsample(inputs, channels):
    skip = inputs
    skip = kl.Conv2DTranspose(filters=channels, kernel_size=(1, 3), strides=(1, 2), activation='relu')(skip)
    conv = inputs
    conv = kl.Conv2DTranspose(filters=channels, kernel_size=(1, 3), strides=(1, 2), activation='relu')(conv)
    conv = kl.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv)
    return kl.Add()([skip, conv])

def downsample(inputs, channels):
    skip = inputs
    skip = kl.Conv2D(filters=channels, kernel_size=(1, 3), strides=(1, 2), activation='relu')(skip)
    conv = inputs
    conv = kl.Conv2D(filters=channels, kernel_size=(1, 3), strides=(1, 2), activation='relu')(conv)
    conv = kl.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv)
    return kl.Add()([skip, conv])

def channel_change(inputs, channels):
    skip = inputs
    skip = kl.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), activation='relu')(skip)
    conv = inputs
    conv = kl.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv)
    conv = kl.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv)
    return kl.Add()([skip, conv])

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return kl.Lambda(func)

def inference_net(input_dim, latent_dim):
    inpt = kl.Input((1, input_dim, 1))
    prev = inpt

    prev = kl.Dense(1, activation='relu')(prev)

    prev = [ crop(2, 16 * i, 16 * i + 16)(prev) for i in range(10) ]
    prev = [ kl.Dense(1)(p) for p in prev ]
    prev = [ kl.Flatten()(p) for p in prev ]
    prev = [ kl.Dense(4)(p) for p in prev ]
    prev = [ kl.Reshape((1, 4, 1))(p) for p in prev ]
    prev = kl.Concatenate(axis=2)(prev)

    prev = kl.Dense(1, activation='relu')(prev)
    prev = kl.Flatten()(prev)
    prev = kl.Dense(2 * latent_dim, activation='linear')(prev)
                            
    mean = prev[:,latent_dim:]
    stdv = 1e-6 + tf.nn.softplus(prev[:,:latent_dim])

    return k.Model(inpt, [mean, stdv], name='inference_net')

def generative_net(latent_dim, reconstruction_dim):
    inpt = kl.Input((latent_dim,))
    prev = inpt

    prev = kl.Dense(16, activation='relu')(prev)

    prev = [ crop(1, 2 * i, 2 * i + 2)(prev) for i in range(8) ]
    prev = [ kl.Dense(8)(p) for p in prev ]
    prev = kl.Concatenate(axis=1)(prev)

    prev = kl.Dense(reconstruction_dim)(prev)
    prev = kl.Reshape(target_shape=(1, reconstruction_dim, 1))(prev)
    return k.Model(inpt, prev, name='generative_net')

def CVAE(input_dim, latent_dim):
    inf = inference_net(input_dim, latent_dim)
    inf.summary()
    gen = generative_net(latent_dim, input_dim)
    gen.summary()

    inpt = kl.Input((1, input_dim, 1))
    prev = inpt
    mean, stdv = inf(prev)
    prev = sampling_layer(mean, stdv, latent_dim)
    prev = gen(prev)

    model = k.Model(inpt, prev, name='cvae')

    def elbo_loss(x, mu, sigma):
        def loss(y_true, y_pred):
            y = tf.sigmoid(y_pred)
            marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
            ELBO = tf.reduce_mean(marginal_likelihood) - tf.reduce_mean(KL_divergence)
            return -ELBO
        return loss

    model.compile(optimizer=k.optimizers.RMSprop(0.001), loss=elbo_loss(inpt, mean, stdv))
    model.summary()

    return model

def from_block(block, padding=None):
    if padding is None:
        padding = INPUT_DIM
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
    return sf.blocks(fname, blocksize=blocksize, start=start, stop=start + segsize, overlap=0, dtype='float32')

np.random.seed(42)
tf.set_random_seed(42)

training_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]

INPUT_DIM  = 160
LATENT_DIM = INPUT_DIM // 10

model = CVAE(INPUT_DIM, LATENT_DIM)

loss_10  = []
loss_100 = []
loss_1000 = []
mloss_100 = 0

epoch_range = trange(10000)
for e in epoch_range:
    batch_range = trange(100)
    block_gen   = None

    for b in batch_range:
        try:
            block = next(block_gen)
        except:
            block_gen = next_file(training_files, blocksize=INPUT_DIM)
            block = next(block_gen)

        x       = from_block(block)
        x_noise = from_block(block + tf.random.normal(block.shape, 0., 0.001))
        loss = model.train_on_batch(x_noise, x)

        loss_10.append(loss)
        loss_10 = loss_10[-10:]
        mloss_10 = np.mean(loss_10)

        if b % 10 == 0:
            loss_100.append(mloss_10)
            loss_100 = loss_100[-10:]
            mloss_100 = np.mean(loss_100)

        if b % 100 == 0:
            loss_1000.append(mloss_100)
            loss_1000 = loss_1000[-10:]

        batch_range.set_postfix({'loss': '{:.03e}'.format(loss), 'loss_10': '{:.03e} {:.03e} {:.03e}'.format(mloss_10, mloss_100, np.mean(loss_1000)) })

    if e % 100 == 0:
        with open('result{:04}.wav'.format(e), 'wb+') as out_codec:
            output = []
            for block in sf.blocks(random.choice(training_files), blocksize=INPUT_DIM, overlap=0, dtype='float32'):
                block = from_block(block)
                block = model.predict(block)
                block = to_block(block)
                output.extend(block)
            sf.write(out_codec, output, 16000)

