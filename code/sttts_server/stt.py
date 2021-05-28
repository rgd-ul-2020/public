#!/usr/bin/python3 

import numpy as np
import sys
import wave

from deepspeech import Model

model_path    = 'deepspeech-0.5.0-models/output_graph.pb'
alphabet_path = 'deepspeech-0.5.0-models/alphabet.txt'
lm_path       = 'deepspeech-0.5.0-models/lm.binary'
trie_path     = 'deepspeech-0.5.0-models/trie'
N_FEATURES = 26
N_CONTEXT  = 9
BEAM_WIDTH = 500
LM_WEIGHT = 1.5
VALID_WORD_COUNT_WEIGHT = 2.1

try:
    input_file = sys.argv[1]
except:
    print("stt-deepspeech.py <file.wav>")
    sys.exit(1)

wav_file = wave.open(input_file, 'rb')
wav_fr   = wav_file.getframerate()

audio     = np.frombuffer(wav_file.readframes(wav_file.getnframes()), np.int16)
audio_len = wav_file.getnframes()

model = Model(model_path, N_FEATURES, N_CONTEXT, alphabet_path, BEAM_WIDTH)
model.enableDecoderWithLM(alphabet_path, lm_path, trie_path, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)
print(model.stt(audio, wav_fr))

