#!/usr/bin/python3

import numpy as np
import os
import scipy as sp
import scipy.signal
import sys
import tensorflow.keras as keras
import tensorflow.keras.layers as kl

from fread import *
from tqdm import *

def resnet(inpt, channels):
    skip = inpt
    prev = kl.Conv1D(channels, kernel_size=1)(inpt)
    prev = kl.Conv1D(channels, kernel_size=1)(prev)
    return kl.Add()([skip, prev])

def encoder(comp):
    inpt = kl.Input((D1_SIZE, 1), name='enc_input')
    prev = resnet(inpt, 1)
    prev = kl.Flatten()(prev)
    prev = kl.Dense(D1_SIZE // comp, activation='relu')(prev)
    return keras.Model(inpt, prev)

def decoder(comp):
    inpt = kl.Input((D1_SIZE // comp,), name='dec_input')
    prev = kl.Dense(D1_SIZE // comp, activation='relu', activity_regularizer=keras.regularizers.l1(10e-6))(inpt)
    prev = kl.Dense(D1_SIZE, activation='sigmoid')(prev)
    prev = kl.Reshape((D1_SIZE, 1))(prev)
    prev = resnet(prev, 1)
    return keras.Model(inpt, prev)

D1_SIZE = 512

def training(epochs=10, batch_size=100):
    optimizer = keras.optimizers.Adadelta()
    loss      = 'binary_crossentropy'

    enc_model = encoder(10)
    enc_model.summary()
    dec_model = decoder(10)
    dec_model.summary()

    inpt = kl.Input((D1_SIZE, 1))
    prev = enc_model(inpt)
    prev = dec_model(prev)
    codec_model = keras.Model(inpt, prev)
    codec_model.compile(optimizer=optimizer, loss=loss)
    codec_model.summary()

    for epoch in tqdm(range(epochs)):
        blocks   = None
        progress = trange(batch_size)
        for _ in progress:
            try:
                block = next(blocks)
            except:
                blocks = sf.blocks(random.choice(training_files), blocksize=512, overlap=32)
                block  = next(blocks)

            block = np.pad(block, (0, 512 - len(block)), 'constant')
            block = np.reshape(block, (1, D1_SIZE, 1))

            loss = codec_model.train_on_batch(block + np.random.normal(0, 1, block.shape), block)
            progress.set_postfix({ 'loss': loss })

    return codec_model, enc_model, dec_model


training_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]

output = []
cdc, enc, dec = training(100, 100)
with open('result.wav', 'wb+') as out_codec:
    for block in sf.blocks(random.choice(training_files), blocksize=512, overlap=0):
        block = np.pad(block, (0, 512 - len(block)), 'constant')

        block = np.reshape(block, (1, D1_SIZE, 1))
        block = dec.predict(enc.predict(block))
        block = np.reshape(block, (D1_SIZE,))

        output.extend(block)
    sf.write(out_codec, output, 16000)
