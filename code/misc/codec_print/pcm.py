#!/usr/bin/python3

import numpy as np
import struct

from scipy.io import wavfile
from tqdm import tqdm

nolaw_enc_table = {}
mulaw_enc_table = {}
alaw_enc_table  = {}

nolaw_dec_table = {}
mulaw_dec_table = {}
alaw_dec_table  = {}

U16 = 65536
H16 = U16 // 2

U8  = 256
H8  = U8 // 2

def pcm_init():
    for i in range(U16):
        x   = (i - H16) / H16
        sgn = 1 if x >= 0 else -1
    
        idx = int(i - H16)
        nolaw_enc_table[idx] = int(U8 * x)
        mulaw_enc_table[idx] = int(sgn * H8 * np.log(1 + U8 * np.abs(x)) / np.log(1 + U8))

        a_code = 87.7 * np.abs(x)
        if np.abs(x) >= 1 / 87.7:
            a_code = 1 + np.log(a_code)
        alaw_enc_table[idx] = int(sgn * H8 * a_code / (1 + np.log(87.7)))

    for i in range(256):
        y   = (i - H8) / H8
        sgn = 1 if y >= 0 else -1

        idx = int(i - H8)
        nolaw_dec_table[idx] = int(U16 * y)
        mulaw_dec_table[idx] = int(sgn * H16 * (1 / U8) * ((1 + U8) ** np.abs(y) - 1))
        alaw_dec_table[idx]  = int(sgn * H16 * (1 / U8) * ((1 + U8) ** np.abs(y) - 1))

def pcm_enc(enc_table, x):
    i = min(32767, max(-32768, x))
    return enc_table[i]

def pcm_dec(dec_table, y):
    i = min(127, max(-128, y))
    return dec_table[i]

if __name__ == '__main__':
    INPUT  = 'costa.wav'
    OUTPUT = '/home/rafael/Projects/thesis/code/codecs/out.wav'

    mu = 255

    pcm_init()

    sum_err = 0
    num_spl = 0

    rate, data = wavfile.read(INPUT)
    for sample in tqdm(data[:1000000,0]):
        enc = pcm_enc(nolaw_enc_table, sample)
        dec = pcm_dec(nolaw_dec_table, enc)

        sum_err += np.abs(dec - sample)
        num_spl += 1

    print("NOLAW AVG: {}".format(sum_err / num_spl))

    sum_err = 0
    num_spl = 0

    rate, data = wavfile.read(INPUT)
    for sample in tqdm(data[:1000000,0]):
        enc = pcm_enc(mulaw_enc_table, sample)
        dec = pcm_dec(mulaw_dec_table, enc)

        sum_err += np.abs(dec - sample)
        num_spl += 1

    print("MULAW AVG: {}".format(sum_err / num_spl))

    sum_err = 0
    num_spl = 0

    rate, data = wavfile.read(INPUT)
    for sample in tqdm(data[:1000000,0]):
        enc = pcm_enc(alaw_enc_table, sample)
        dec = pcm_dec(alaw_dec_table, enc)

        sum_err += np.abs(dec - sample)
        num_spl += 1

    print("A-LAW AVG: {}".format(sum_err / num_spl))

        
