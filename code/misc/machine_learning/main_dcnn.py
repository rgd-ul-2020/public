#!/home/rafael/anaconda3/bin/python3

import numpy as np
import os
import random
import soundfile as sf
import sys
import torch                          
import torch.nn as nn

from torch.nn.utils import weight_norm
from tqdm import trange

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                    dropout=dropout)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train(training_set):
    model.train()

def evaluate(valid_set):
    model.eval()

def from_block(block, padding=None):
    if padding is None:
        padding = INPUT_DIM
    n_block = len(block)
    block   = np.pad(block, (0, padding - n_block), 'constant')
    block   = (block + 1.) / 2.
    block   = np.reshape(block, (1, padding, 1))
    return torch.from_numpy(block)

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
torch.manual_seed(42)

speech_files = [ os.path.join('inputs/', f) for f in os.listdir('inputs/') ]
random.shuffle(speech_files)
train_files  = speech_files[:5]
valid_files  = speech_files[5:10]

INPUT_DIM  = 160
LATENT_DIM = INPUT_DIM // 10

model = TemporalConvNet(INPUT_DIM, [ 160, 80, 40, 80, 160 ], 3)

epoch_range = trange(100)
for e in epoch_range:
    batch_range = trange(1000)
    train_gen   = None
    valid_gen   = None

    for b in batch_range:
        train_block = next_block(train_files, train_gen, INPUT_DIM)
        valid_block = next_block(valid_files, valid_gen, INPUT_DIM)

        volume = min(0.5, random.random())
        noise  = min(0.,  max(0.1, random.random()))

        train(train_files)
        evaluate(valid_files)
        x       = from_block(train_block)
        x_noise = from_block(volume * train_block + np.random.normal(0., 1., train_block.shape))

        x       = from_block(valid_block)
        x_noise = from_block(volume * valid_block + np.random.normal(0., 1., valid_block.shape))

    if e >= 0:
        with open('result{:04}.wav'.format(e), 'wb+') as out_codec:
            output = []
            for block in sf.blocks(random.choice(speech_files), blocksize=INPUT_DIM, overlap=0):
                block = from_block(block)
                block = model(block)
                block = to_block(block)
                output.extend(block)
            sf.write(out_codec, output, 16000)

