#!/usr/bin/python3

import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as K
import tensorflow.keras.layers as kl
import soundfile as sf
import wxmplot.interactive as wi
import wx
import wxmplot
import threading

from tqdm import trange

tf.enable_eager_execution()

class CausalConv1D(kl.Conv1D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )
       
    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)

class TemporalBlock(kl.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2, 
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )        
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv2")
        self.down_sample = None
    
    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = kl.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = kl.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1, 
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)
    
    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)

class TemporalConvNet(kl.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1, dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )
    
    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs

def from_block(block, padding=None):
    if padding is None:
        padding = INPUT_DIM
    block   = tf.cast(block, tf.float16)
    block   = np.pad(block, (0, padding - len(block)), 'constant')
    block   = (block + 1.) / 2.
    block   = np.reshape(block, (1, len(block), 1))
    return block

def to_block(code):
    code = np.reshape(code, (INPUT_DIM,))
    code = (code * 2.) - 1.
    return code

def next_file(flist, blocksize=1024, dur=3):
    fname   = random.choice(flist)
    info    = sf.info(fname)
    segsize = dur * info.samplerate
    start = random.randint(0, max(0, info.frames - segsize))
    return sf.blocks(fname, blocksize=blocksize, start=start, stop=start + segsize, overlap=0)

def next_block(files, block_gen, blocksize):
    try:
        block = next(block_gen)
    except:
        block_gen = next_file(files, blocksize=INPUT_DIM)
        block = next(block_gen)
    return block

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

speech_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]
random.shuffle(speech_files)
train_files  = speech_files[:5]
valid_files  = speech_files[5:10]

INPUT_DIM  = 160
LATENT_DIM = INPUT_DIM // 10

optimizer = tf.keras.optimizers.Adam(0.001)

inputs = kl.Input((INPUT_DIM, 1)) 
prev   = TemporalConvNet([160], 8, 0.1)(inputs, training=True)
prev   = kl.Dense(1, activation='softmax')(prev)
model  = tf.keras.Model(inputs, prev)
model.summary()
model.compile(optimizer=optimizer, loss='mse')

tlosses = list(np.linspace(0, 0, 1000))
vlosses = list(np.linspace(0, 0, 1000))
           
ilosses = np.linspace(0, 1000, 1000)
i = 0

app = wx.App()
frame = wxmplot.PlotFrame()

frame.plot(ilosses, tlosses, color='red')
frame.oplot(ilosses, vlosses, color='darkgreen')
frame.Show()
threading.Thread(target=app.MainLoop).start()

epoch_range = trange(100)
for e in epoch_range:
    batch_range = trange(100)
    train_gen   = None
    valid_gen   = None

    for b in batch_range:
        train_block = next_block(train_files, train_gen, INPUT_DIM)
        valid_block = next_block(valid_files, valid_gen, INPUT_DIM)

        volume = min(0.5, random.random())
        noise  = min(0.,  max(0.1, random.random()))

        x       = from_block(train_block)
        x_noise = from_block(volume * train_block + tf.random.normal(train_block.shape, 0., noise))
        tlosses.append(model.train_on_batch(x_noise, x))
        tlosses = tlosses[-1000:]

        x       = from_block(valid_block)
        x_noise = from_block(volume * valid_block + tf.random.normal(valid_block.shape, 0., noise))
        vlosses.append(model.train_on_batch(x_noise, x))
        vlosses = vlosses[-1000:]

    frame.clear()
    frame.plot(ilosses, tlosses,  color='red')
    frame.oplot(ilosses, vlosses, color='darkgreen')

    with open('result{:04}.wav'.format(e), 'wb+') as out_codec:
        output = []
        for block in sf.blocks(random.choice(speech_files), blocksize=INPUT_DIM, overlap=0):
            block = from_block(block)
            block = model.predict(block)
            block = to_block(block)
            output.extend(block)
        sf.write(out_codec, output, 16000)

