#!/usr/bin/python3

import matplotlib.pyplot as plt

import numpy as np
import os
import scipy
import soundfile as sf

from math import ceil
from librosa.core import lpc
from pesq  import pesq
from scipy import signal

from scipy.cluster.vq import whiten

FRAME_RATE     = 16000
FRAME_SIZE     = int(FRAME_RATE * 0.020)
SUBFRAME_SIZE  = int(FRAME_RATE * 0.005)
SUBFRAME_COUNT = FRAME_SIZE // SUBFRAME_SIZE

SUBFRAME4_SIZE  = 20
SUBFRAME8_SIZE  = 40
SUBFRAME16_SIZE = 80

def lerp(v0, v1, t):
    return v0 + t * (v1 - v0)

def downsample(x, factor):
    return x[::factor]

def upsample(x, factor):
    x2 = np.concatenate([x[1:], [x[-1]]])
    y = np.empty(x.size * factor, dtype=x.dtype)
    for i in range(factor):
        y[i::factor] = lerp(x, x2, i / factor)
    return y

def encode(x):
    xw = whiten(x)
    x8 = downsample(xw, 2)

    x4 = downsample(x8, 2)

    target = SUBFRAME4_SIZE * 4
    for _ in range(len(x4) // SUBFRAME4_SIZE):
        basis = target - min_lag_4hz
        print(target, basis)

        target += SUBFRAME_SIZE

    return x4

def decode(x):
    return upsample(x, 4)

sample_dir   = 'data/'
sample_files = [ os.path.join(sample_dir, f) for f in os.listdir(sample_dir) ]

data, samplerate = sf.read(sample_files[0])

size = 0
with sf.SoundFile('output/inp.wav', 'w+', samplerate=samplerate, channels=1) as inp, \
     sf.SoundFile('output/out.wav', 'w+', samplerate=samplerate, channels=1) as out:
    for i in range(ceil(len(data) / FRAME_SIZE)):
        x = data[i * FRAME_SIZE:i * FRAME_SIZE + FRAME_SIZE]
        c = encode(x)
        y = decode(c)
        inp.write(x)
        out.write(y)
        size += len(c)
print(len(data), size)

x, samplerate = sf.read('output/inp.wav')
y, samplerate = sf.read('output/out.wav')

print(pesq(samplerate, x, y, 'nb'))
print(pesq(samplerate, x, y, 'wb'))

plt.figure()
plt.plot(x)
plt.plot(y, linestyle='--')
plt.legend(['y', 'y_hat'])
plt.title('LPC')
plt.show()
