import numpy as np
import random
import soundfile as sf
import tensorflow as tf

def read_next(cursor, nsamples, overlap=0, pad_with_zeroes=False):
    data, samplerate = sf.read(cursor)
    pos, size = 0, len(data)
    while True:
        segment = data[pos:pos+nsamples]
        if len(segment) == 0:
            break
        if len(segment) < nsamples and pad_with_zeroes:
            segment = np.pad(segment, (0, nsamples - len(segment)), 'constant')      
        yield segment
        pos += len(segment)

def write_batch(cursor, batch):
    batch   = np.reshape(batch, (WINDOW,))
    samples = []
    for number in batch:
        sample = int(min(32767, max(-32768, number * 32768)))
        samples.append(sample)
    try:
        cursor.write(struct.pack(len(samples) * 'h', *samples))
    except: 
        print(samples)
        raise

def generate_data(file_list, nsamples, overlap, pad_with_zeroes=False):
    while True:
        file_name = file_list[random.randrange(len(file_list))]
        with open(file_name, 'rb+') as cursor:
            for batch in read_next(cursor, nsamples, overlap, pad_with_zeroes):
                batch = np.reshape(batch, (1, nsamples, 1))
                yield batch

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
