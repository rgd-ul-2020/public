#!/usr/bin/python3

import pyaudio
import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np
import struct
import time
import soundfile as sf

from deepspeech import Model

RATE    = 16000
CHUNK   = 1600
BUFSIZE = 3 * RATE
buf = [ 0 for _ in range(BUFSIZE) ]

audio  = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, frames_per_buffer=CHUNK, input=True)
output = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, frames_per_buffer=CHUNK, output=True)

model_path    = 'deepspeech-0.5.0-models/output_graph.pb'
alphabet_path = 'deepspeech-0.5.0-models/alphabet.txt'
lm_path       = 'deepspeech-0.5.0-models/lm.binary'
trie_path     = 'deepspeech-0.5.0-models/trie'
N_FEATURES = 26
N_CONTEXT  = 9
BEAM_WIDTH = 500
LM_WEIGHT = 1.5
VALID_WORD_COUNT_WEIGHT = 2.1

model = Model(model_path, N_FEATURES, N_CONTEXT, alphabet_path, BEAM_WIDTH)
model.enableDecoderWithLM(alphabet_path, lm_path, trie_path, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)

THRESHOLD = 2500

fig    = plt.figure()
screen = pf.screen(title='Plot')


def any_sc(array, fn):
    for item in array:
        if fn(item):
            return True
    return False

start   = time.time()
started = False
silence = 0
buf   = []
last  = start
gib_id = 0
while True:
    now = time.time()

    read = stream.read(CHUNK)
    if len(read) == 0:
        break;
    
    signal = np.frombuffer(read, dtype=np.int16)
    
    if not started:
        if any_sc(signal, lambda x: x > THRESHOLD):
            started = True
            silence = 0
            buf.extend(signal)
        else:
            buf = signal

    elif started:
        buf.extend(signal)
        if not any_sc(signal, lambda x: x > THRESHOLD):
            silence += 1
        if silence > 3:
            started = False

    if not started and len(buf) > 0:
        sf.write('out_{}.wav'.format(gib_id), buf, 16000)
        gib_id += 1

        guess = model.stt(buf, 16000)
        if guess != '':
            print('{:.03f}: {}'.format(last - start, guess))
            if guess == 'exit':
                break
        else:
            print('{:.03f}: Gibberish'.format(last - start, guess))
        buf = []
    last = now

stream.stop_stream()
stream.close()
audio.terminate()

