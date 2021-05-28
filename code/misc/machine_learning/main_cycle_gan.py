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

def channel_change(inputs, channels):
    skip = kl.Conv1D(channels, kernel_size=1)(inputs)
    prev = kl.Conv1D(channels, kernel_size=1)(inputs)
    prev = kl.Conv1D(channels, kernel_size=1)(prev)
    return kl.Add()([skip, prev])

def residual_block(inputs, channels, dilation):
    skip = inputs
    prev = kl.Conv1D(channels, kernel_size=1, dilation_rate=dilation)(inputs)
    prev = kl.Conv1D(channels, kernel_size=1, dilation_rate=dilation)(prev)
    return kl.Add()([skip, prev])

def downsample(inputs, channels, comp):
    skip = kl.Conv1D(channels, kernel_size=1, strides=comp)(inputs)
    prev = kl.Conv1D(channels, kernel_size=1, strides=comp)(inputs)
    prev = kl.Conv1D(channels, kernel_size=1, strides=1)(prev)
    return kl.Add()([skip, prev])

def upsample(inputs, channels, comp):
    skip = SubPixel1D(inputs, comp)
    skip = kl.Conv1D(channels, kernel_size=1)(skip)
    prev = SubPixel1D(inputs, comp)
    prev = kl.Conv1D(channels, kernel_size=1)(prev)
    return kl.Add()([skip, prev])

def encoder(comp):
    inpt = kl.Input((512, 1), name='enc_input')
    prev = channel_change(inpt, 64)
    for i in [1, 2, 4, 8]:
        prev = residual_block(prev, 64, i)
    prev = downsample(prev, 64, comp)
    for i in [1, 2, 4, 8]:
        prev = residual_block(prev, 64, i)
    prev = channel_change(prev,  1)
    return keras.Model(inpt, prev)

def decoder(comp):
    inpt = kl.Input((512 // comp, 1), name='dec_input')
    prev = channel_change(inpt, 64)
    for i in [1, 2, 4, 8]:
        prev = residual_block(prev, 64, i)
    prev = upsample(prev, 64, comp)
    for i in [1, 2, 4, 8]:
        prev = residual_block(prev, 64, i)
    prev = channel_change(prev,  1)
    return keras.Model(inpt, prev)

def critic(channels):
    inpt = kl.Input((channels, 1))
    prev = kl.Conv1D(channels // 8, kernel_size=3, strides=2)(inpt)
    prev = kl.Conv1D(channels // 4, kernel_size=3, strides=2)(prev)
    prev = kl.Conv1D(channels // 2, kernel_size=3, strides=2)(prev)
    prev = kl.Conv1D(channels,      kernel_size=3, strides=2)(prev)
    prev = kl.Flatten()(prev)
    prev = kl.Dense(1, name='after-flatten-' + str(random.random()))(prev)
    return keras.Model(inpt, prev)

def cycle_gan():
    optimizer = keras.optimizers.Adam(0.0002, 0.5)

    disc_enc = critic(64)
    disc_dec = critic(512)
    disc_enc.compile(loss='mse', optimizer=optimizer, metric=['accuracy'])
    disc_dec.compile(loss='mse', optimizer=optimizer, metric=['accuracy'])

    gen_encoder = encoder(8)
    gen_decoder = decoder(8)

    input_sample = kl.Input((512, 1))
    input_code   = kl.Input((256, 1))

    fake_code   = gen_encoder(input_sample)
    fake_sample = gen_decoder(input_code)

    rect_sample = gen_decoder(fake_code)
    rect_code   = gen_encoder(fake_sample)

    disc_enc.trainable = False
    disc_dec.trainable = False

    valid_code   = disc_enc(fake_code)
    valid_sample = disc_dec(fake_sample)

    model = keras.Model([input_sample, input_code], [ valid_code, valid_sample,
        rect_sample, rect_code ])
    model.compile(loss=[ 'mse', 'mse', 'mae', 'mae' ],
        loss_weights=[ 1, 1, 10, 10 ], optimizer=optimizer)

    epochs     = 10
    batch_size = 100

    valid = np.ones((batch_size,))
    fake  = np.zeros((batch_size,))

    for epoch in range(epochs):
        batch_idx = 0
        for block in sf.blocks(random.choice(training_files), blocksize=512, overlap=32):
            block = np.pad(block, (0, 512 - len(block)), 'constant')
            if batch_idx > batch_size:
                break
            batch_idx += 1
        #TODO: TRAIN
        
    return gen_encoder, gen_decoder

training_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]

enc, dec = cycle_gan()
with open('result.wav', 'wb+') as out:
    output = []
    for block in sf.blocks(training_files[0], blocksize=512, overlap=0):
        block = np.pad(block, (0, 512 - len(block)), 'constant')
        res = enc.predict(np.reshape(block, (1, 512, 1)))
        output.extend(np.reshape(dec.predict(res), (512,)))
    sf.write(out, output, 16000)

