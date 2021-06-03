#!/usr/bin/python3

import numpy as np
import pygame
import pypesq
import os
import random
import soundfile as sf
import sys
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl

from tqdm import trange

def QNet():
    tinputs = kl.Input((160, 1))
    pinputs = kl.Input((160, 1))
    
    prev = kl.Concatenate(axis=1)([tinputs, pinputs])
    prev = kl.Conv1D(4,  kernel_size=3, strides=2)(prev)
    prev = kl.Conv1D(16, kernel_size=5, strides=3)(prev)
    prev = kl.Conv1D(32, kernel_size=5, strides=3)(prev)
    prev = kl.Conv1D(1,  kernel_size=3, strides=2)(prev)
    prev = kl.LSTM(1)(prev)

    model = k.Model([tinputs, pinputs], prev)
    model.compile('adam', 'mae')
    model.summary()

    try:
        model.load_weights('checkpoints/QNET.h5')
    except:
        pass
    return model

def from_block(block, padding=None):
    n_block = len(block)
    block   = np.pad(block, (0, 16000 - n_block), 'constant')
    block   = (block + 1.) / 2.
    block   = np.reshape(block, (100, 160, 1))
    block   = tf.cast(block, tf.float16)
    return block

def to_block(code):
    code = np.reshape(code, (16000,))
    code = (code * 2.) - 1.
    return code

def next_file(flist, blocksize=16000, start=None, dur=3):
    fname   = random.choice(flist)
    info    = sf.info(fname)
    segsize = dur * info.samplerate
    if start == None:
        start = random.randint(0, max(0, info.frames - segsize))
    return sf.blocks(fname, blocksize=blocksize, start=start, stop=start + segsize, overlap=0)

def next_block(files, block_gen, blocksize):
    try:
        block = next(block_gen)
    except:
        block_gen = next_file(files, blocksize=blocksize)
        block = next(block_gen)
    return from_block(block, blocksize)

def pesq(ref, deg):
    try:
        ref = np.reshape(ref, (16000,))
        deg = np.reshape(deg, (16000,))
        return pypesq.pypesq(16000, ref, deg, 'nb')
    except:
        return 0

class LivePlot:
    def __init__(self, mode, colors=None):
        pygame.init()  
        self.display = pygame.display.set_mode(mode)
        self.font    = pygame.font.Font(pygame.font.get_default_font(), 18)
        self.colors  = colors
        if colors == None:
            self.colors = [
                (255, 0, 0),
                (0, 0, 255)
            ]

    def update(self, vecs, limits):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.display.fill((255,255,255))

        for i, vec in enumerate(vecs):
            self.draw_vec(vec, self.colors[i], limits)

        pygame.display.update()
        return True

    def draw_vec(self, vec, color, limits):
        if len(vec) == 0:
            return

        vec_limits = (np.min(vec), np.max(vec))

        adj_max = limits[1] - limits[0]
        if adj_max == 0:
            adj_max = 1

        inv_y = 600 * (vec[0] - limits[0]) / adj_max
        if inv_y ==  float('inf'): inv_y = 600
        if inv_y == -float('inf'): inv_y = 0

        prev = (0, int(600 - inv_y))
        for (x, l) in enumerate(vec):
            inv_y = 600 * (l - limits[0]) / adj_max
            if inv_y ==  float('inf'): inv_y = 600
            if inv_y == -float('inf'): inv_y = 0
            curr = (x, int(600 - inv_y))
            pygame.draw.line(self.display, color, prev, curr, 1)
            prev = curr

            if l in limits or l in vec_limits:
                text = self.font.render('{:.05e}'.format(l), True, color)
                pygame.draw.circle(self.display, color, curr, 5)
                self.display.blit(text, curr + (5, 0))

class EarlyStop(Exception): pass

if __name__ == '__main__':
    tf.enable_eager_execution()

    plot = LivePlot((800, 600))

    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    speech_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]
    #speech_files = [ 'inputs/3752-4943-0020.flac' ]
    random.shuffle(speech_files)
    train_files  = speech_files[:5]
    valid_files  = speech_files[5:10]

    model = QNet()

    train_hist = []
    test_hist  = []

    best_loss = 1

    factors = [ 1, 0.85, 0.75, 0.625, 0.3215, 0.015625, 0.0078125, 0.005, 0.003, 0.001, 0 ]
    closed  = False

    try:
        for epoch in trange(100):
            brange = trange(100)

            factor = int(random.random() * 10)
            bnoise  = np.random.uniform(0, factors[factor], (100, 160, 1)) 
            bnoisel = bnoise[:-1,:,:]
            bnoisep = bnoise[-1:,:,:]

            for batch in brange:
                model.reset_states()

                good  = next_block(train_files, None, 16000)
                goodl = good[:-1,:,:]
                goodp = good[-1:,:,:]
                #test  = next_block(valid_files, None, 16000)

                noise  = np.clip(good  + bnoise,  0, 1)
                noisel = np.clip(goodl + bnoisel, 0, 1)
                noisep = np.clip(goodp + bnoisep, 0, 1)
                #tnoise = test + bnoise

                score = pesq(good, noise) / 5 # 5 = max narrowband score

                _     = model.test_on_batch([ goodl, noisel ], 99 * [score])
                loss  = np.mean(model.train_on_batch([ goodp, noisep ], [score]))
                guess = model.predict([ goodp, noisep ])[-1,0]

                train_hist.append(loss)
                train_hist = train_hist[-800:]

                test_hist.append(np.mean(train_hist))
                test_hist = test_hist[-800:]
                """
                inputs  = tf.concat([ test, tnoise ], axis=1)
                score   = pesq(test, tnoise) / 5 # 5 = max narrowband score
                loss    = np.mean(model.test_on_batch(inputs, [score]))
                guess   = np.mean(model.predict(inputs))
                test_hist.append(loss)
                test_hist = test_hist[-800:]
                """
                if loss < best_loss:
                    model.save_weights('checkpoints/QNET.h5')
                    best_loss = loss

                brange.set_postfix({ 'factor': factor / 2, 'score': 5 * score, 'guess': 5 * guess, 'loss': loss })

                if not plot.update([ train_hist, test_hist ], (best_loss, np.max(train_hist))):
                    raise EarlyStop()

    except EarlyStop as e:
        closed = True

    except Exception as e:
        print('!', e)
        raise

    if not closed:
        import time
        if not plot.update([ train_hist, test_hist ], (best_loss, np.max(train_hist))):
            time.sleep(0.2)

    model.save_weights('checkpoints/QNET.h5')
