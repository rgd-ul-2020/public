#!/usr/bin/python3

import numpy as np
import os
import random
import soundfile as sf

import tensorflow              as tf
import tensorflow.keras.layers as kl
import tensorflow.keras        as keras

from tqdm import tqdm

def SubPixel1D(I, r):
    """
        One-dimensional subpixel upsampling layer
        Calls a tensorflow function that directly implements this functionality.
        We assume input has dim (batch, width, r)

        https://github.com/kuleshov/audio-super-res/blob/master/src/models/layers/subpixel.py
    """
    with tf.name_scope('subpixel'):
        X = tf.transpose(I, [2,1,0]) # (r, w, b)
        X = tf.compat.v1.batch_to_space_nd(X, [r], [[0,0]]) # (1, r*w, b)
        X = tf.transpose(X, [2,1,0])
        return X

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

def channel_change(inputs, channels):
    skip = kl.Conv1D(channels, kernel_size=1)(inputs)
    prev = kl.Conv1D(channels, kernel_size=1)(inputs)
    prev = kl.Conv1D(channels, kernel_size=1)(prev)
    return kl.Add()([skip, prev])

LayerType = {
    'input':   lambda args, prev: kl.Input((args[1], 1)),
    'channel': lambda args, prev: channel_change(prev, args[1]), 
    'downspl': lambda args, prev: downsample(prev, args[1], args[2]),
    'upspl':   lambda args, prev: upsample(prev, args[1], args[2]),
    'output':  lambda args, prev: kl.Dense(1)(prev),
}

class Individual:
    _layers    = []

    def __init__(self, infl_size, comp_size):
        self._layers = [
            ('input', 512),
            ('channel', 64),
            ('downspl', 64, 8),
            ('channel', 1),
            ('output',),
            ('channel', 64),
            ('upspl', 64, 8),
            ('channel', 1),
            ('output',),
        ]
        self._invalid = True

    def get_model(self):
        if self._invalid:
            self._invalid = False

            inputs = None
            prev   = None
            for layer in self._layers:
                prev = LayerType[layer[0]](layer, prev)
                if layer[0] == 'input':
                    inputs = prev
            self._model = keras.Model(inputs, prev)
            self._model.compile(optimizer='adadelta', loss='logcosh')

        return self._model

    def mutate_random_weight(self):
        layers = self._model.layers
        if len(layers) == 0: 
            return
        sel_layer = layers[random.randrange(len(layers))]

        weights = sel_layer.get_weights()
        if len(weights) == 0:
            return
        sel_weight = weights[random.randrange(len(weights))]

        indexes   = np.prod(sel_weight.shape)
        #sel_index = random.randrange(indexes)
        sel_index = indexes - 1
        index = []
        for n in reversed(sel_weight.shape):
            index.append(sel_index % n)
            sel_index = sel_index // n
        index = tuple(reversed(index))

        sel_weight[index] = 2 * random.random() - 1
        sel_layer.set_weights(weights)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self._layers)

class Population:
    _individuals = []

    def __init__(self, min_size, max_size):
        self._individuals = [ Individual(512, 51) for _ in range(min_size) ]
        self._min_size    = min_size
        self._max_size    = max_size

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self._individuals)


training_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]

pop = Population(5, 150)

with open('result.wav', 'wb+') as out_codec:
    output1 = []
    output2 = []

    while True:
        for block in tqdm(sf.blocks(random.choice(training_files), blocksize=512, overlap=0)):
            block = np.pad(block, (0, 512 - len(block)), 'constant')
            block = np.reshape(block, (1, 512, 1))

            for i, ind in enumerate(pop._individuals):
                model    = ind.get_model()
                ind.loss = model.train_on_batch(block, block)
                ind.fit  = np.linalg.norm(model.predict(block) - block)

        for i, ind in enumerate(pop._individuals):
            print("{:03}:\t{:.5e}\t{:.5e}".format(i, ind.loss, ind.fit))
            ind.mutate_random_weight()

        #output1.extend(np.reshape(dec.predict(res), (512,)))

    #sf.write(out_codec,  output1, 16000)

