#make discriminator a autoencoder
import argparse
import os
import numpy as np
import math
import time
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epoches", type=int, default=1, help="number of epoches of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam:decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type = int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=7, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
CUDA = True if torch.cuda.is_available() else False


# dataloader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

writer = SummaryWriter('run/ebgan')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            #nn.functional.interpolate(self, scale_factor=2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.functional.interpolate(self, scale_factor=2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #conv2d downsampling upchanneling
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, 64, 3, 2, 1), 
            nn.ReLU()
        )

        #Fully-connected layers
        self.down_size =  opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            #nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),

            nn.Linear(32, down_dim),
            #nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )

        #upsample & conv2d
        self.up = nn.Sequential(
            #nn.functional.interpolate(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, opt.channels, 3, 1, 1),
        )
    
    def forward(self, img):
        out = self.down(img)

        out = out.view(out.size(0), -1)
        embedding = self.embedding(out)#[1, 32]
        

        out = self.fc(embedding)

        out = out.view(out.size(0), 64, self.down_size, self.down_size)
        out = self.up(out)

        return out, embedding

#reconstruciton loss of autoencoder
pixelwise_loss = nn.MSELoss()

#initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


if CUDA:
    generator.cuda()
    discriminator.cuda()
    pixelwise_loss.cuda()


#initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings**2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1 + 10e-8    ))
    return loss_pt





########################train#############################
#hyperparameter
lambda_pt = 0.1
margin = max(1, opt.batch_size/64.0)

start_time = time.time()
for epoch in range(opt.n_epoches):
    start_epoch = time.time()

    for i, (imgs, _) in enumerate(dataloader):

        start_image = time.time()
###########inputs#############
        real_imgs = Variable(imgs.type(Tensor))

######################train generator##########################
        optimizer_G.zero_grad()

        #generate noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (img_shape[0], opt.latent_dim))))

        #generate a batch of images, and discriminate
        gen_imgs = generator(z)
        recon_imgs, img_embeddings = discriminator(gen_imgs)

        #g_loss
        g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)
        
        g_loss.backward()
        optimizer_G.step()


#####################train discriminator########################
        optimizer_D.zero_grad()

        real_recon, _ = discriminator(real_imgs)
        fake_recon, _ = discriminator(gen_imgs.detach())

        d_loss_real = pixelwise_loss(real_recon, real_imgs)
        d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

        d_loss = d_loss_real
        if (margin - d_loss_fake.data).item() > 0:
            d_loss += margin - d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        #writer tensorboard
        writer.add_scalar('g_loss', g_loss, epoch*len(dataloader)+i)
        writer.add_scalar('d_loss', d_loss, epoch*len(dataloader)+i)
###############print results#################

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epoches, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/ebgan%d.png" % batches_done, nrow=5, normalize=True)

        end_image = time.time()
        image_time = end_image - start_image
        print("image time is {:f}".format(image_time))

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    print("epoch time is {:f}".format(epoch_time))
    
end_time = time.time()
total_time = end_time - start_time
print("total time is {:f}".format(total_time))