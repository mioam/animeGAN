import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

import dataloader
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt


SZ = 40
batch_size = 32
lr = .1

label_dim = 10
img_dim = (3, SZ, SZ)
latent_dim = 400

def plot(name,gen):
    gen.eval()
    latent = torch.randn((100, gen.latent_dim), device=device)
    imgs = gen.sample(latent).cpu()

    m = np.zeros((SZ*10,SZ*10))
    for i in range(imgs.shape[0]):
        x = i // 10
        y = i % 10
        m[x*28:(x+1)*28,y*28:(y+1)*28] = imgs[i,0]
    plt.imsave(name,m)

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = dataloader.load()
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

generator = Generator(latent_dim, img_dim)
discriminator = Discriminator(img_dim)
generator.to(device)
discriminator.to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


num_epochs = 20
rd = torch.distributions.Normal(0, 1)
for epoch in range(num_epochs):
    generator.train()
    for it, data in enumerate(dataloader):
        img0, label = data
        
        latent = rd.sample((img0.shape[0],latent_dim)).to(device)
        img1 = generator(latent)
        img0 = img0.to(device)
        output0 = discriminator(img0)
        output1 = discriminator(img1)
        
        loss0 = -torch.mean(torch.log(output0+0.01))
        loss1 = -torch.mean(torch.log(1.01-output1))
        loss_d = loss0 + loss1
        d_optimizer.zero_grad()
        loss_d.backward()
        d_optimizer.step()

        latent = rd.sample((img0.shape[0],latent_dim)).to(device)
        img1 = generator(latent)
        output1 = discriminator(img1)
        loss_g = -torch.mean(torch.log(output1+0.01))

        g_optimizer.zero_grad()
        loss_g.backward()
        g_optimizer.step()
        if it % 100 == 0:
            print('epoch: %d, iter: %d, loss0: %f, loss1: %f'%(epoch,it,loss0.item(),loss1.item()))
    plot(generator,'./imgs/%d.png'%(epoch))
    if epoch % 20 == 19:
            torch.save(generator,'./model/generator-%d.npy'%(epoch))
            torch.save(discriminator,'./model/discriminator-%d.npy'%(epoch))
