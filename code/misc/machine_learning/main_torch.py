#!/home/rafael/anaconda3/bin/python

import argparse
import os
import math
import numpy as np
import liveplot as lp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [ nn.Linear(in_feat, out_feat) ]
            if normalize: 
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

adversarial_loss = torch.nn.BCELoss()

generator     = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

D_hist = []
G_hist = []

bound_min = ( 1, -1)
bound_max = (-1, -1)

plot = lp.LivePlot((800, 600))

def lerp(v0, v1, t):
    return v0 + t * (v1 - v0)

etqdm = tqdm.trange(opt.n_epochs)
for epoch in etqdm:
    btqdm = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (imgs, _) in btqdm:
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.), requires_grad=False)
        fake  = Variable(Tensor(imgs.size(0), 1).fill_(0.), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        optimizer_G.zero_grad()
        z        = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = generator(z)
        g_loss   = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(real_imgs), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        G_hist.append(g_loss.item())
        G_hist = G_hist[-800:]

        D_hist.append(d_loss.item())
        D_hist = D_hist[-800:]

        it = epoch * len(dataloader) + i
        mn = min(g_loss.item(), d_loss.item())
        mx = max(g_loss.item(), d_loss.item())

        local_min = min(np.min(G_hist), np.min(D_hist))
        local_min = lerp(bound_min[0], local_min, (it - bound_min[1]) / 8000)

        local_max = max(np.max(G_hist), np.max(D_hist))
        local_max = lerp(bound_max[0], local_max, (it - bound_max[1]) / 8000)

        if mn < local_min:
            bound_min = (mn, it)

        if mx > local_max:
            bound_max = (mx, it)

        if not plot.update([ D_hist, G_hist ], (local_min * 0.9, local_max * 1.05)):
            raise EarlyStop()

        btqdm.set_postfix({
            "D loss": d_loss.item(),
            "G loss": g_loss.item()
        })

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images{}.png".format(batches_done), nrow=5, normalize=True)



