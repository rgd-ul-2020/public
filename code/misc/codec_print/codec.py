#!/usr/bin/python3

import numpy as np
import os, sys
import struct

import math

from PIL import Image, ImageColor
from scipy import signal

INPUT  = '/home/rafael/Projects/thesis/code/sttts_server/sounds/1_cropped/000_the_late_hour_make_the_city.wav'
OUTPUT = '/home/rafael/Projects/thesis/code/codecs/out.wav'

BUFFER_SIZE = 4800
SAMPLE_SIZE = 48000

def read_next(cursor):
    bytes_read = in_file.read(BUFFER_SIZE * 2)
    if len(bytes_read) == 0:
        return []

    shorts_read = struct.unpack(len(bytes_read) // 2 * 'h', bytes_read)
    samples = []
    for short in shorts_read:
        samples.append(short / 32768)
    for _ in range(BUFFER_SIZE - len(shorts_read)):
        samples.append(0)
    
    return np.array(samples)


with open(INPUT,  'rb+') as in_file, \
     open(OUTPUT, 'wb+') as out_file \
:
    out_file.write(in_file.read(44))

    in_file.seek(0, os.SEEK_END)
    size = in_file.tell()
    in_file.seek(44, os.SEEK_SET)

    padded_size = int(math.ceil((size - 44) / BUFFER_SIZE) * BUFFER_SIZE)
    
    prev_im    = None
    prev_width = 0

    x = 0
    while True:
        samples = read_next(in_file)
        if len(samples) == 0:
            break
            
        freqs, segs, stft = signal.stft(samples, 48000)

        new_width  = len(segs)
        new_height = len(freqs)
        new_im     = Image.new('RGB', (prev_width + new_width, new_height))
        if prev_im != None:
            new_im.paste(prev_im, (0, 0, prev_width, new_height))
        for i, freq in enumerate(freqs):
            for j, seg in enumerate(segs):
                try:
                    c = int(-10 * math.log(np.abs(stft.real[i, j])))
                except:
                    c = 0
                new_im.putpixel((prev_width + j, new_height - i - 1), (c, c, c))
        prev_im     = new_im
        prev_width += new_width
    prev_im.save('graph.png'.format(x)) # or any image format

        #for 
        #y = int(48 - (48 * ((sample / 2) + 0.5)))
        #try:
        #    im.putpixel((x, y), (255, 255, 255)) 
        #except:
        #    print(x,y,(int(math.ceil(size / 2)), FRAME_SIZE), m)
        #print(y)
        #x += 1

sys.exit(0)
"""
        energy   = 0
        freq, segs, specs = signal.spectrogram(fsamples, 48000, )
        print(freq, segs, specs)

        y = FRAME_SIZE - 1
        for bucket in freq:        
            val = bucket / energy
            if val < (1/3):
                color = (0, 0, int(3 * val * 255))
            elif val < (2/3):
                color = (0, int(3 * (val - 1/3) * 255), 0)
            else:
                color = (int(3 * (val - 2/3) * 255), 0, 0)
            y -= 1

        samples = np.fft.ifft(samples);
        samples = [ int(sample * 32768) for sample in samples.real ]
        samples = struct.pack(FRAME_SIZE * 'h', *samples)
        out_file.write(samples)

        x += 1   """

