import torch
from torch import nn
from torch._C import device
from torch.nn import functional as F
from torch import optim

import numpy as np
import matplotlib.pyplot as plt

import models

def weights_init_ortho(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, .8)

class ReDO():
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.SZ = 128
        self.C = 3
        self.nMask = 2
        self.latent = 32
        nf = 64

        self.EncM = models.EncM(sizex=self.SZ, nIn=self.C, nMasks=self.nMask, nRes=3, nf=nf, temperature=1).to(device)
        self.GenX = models.GenX(sizex=self.SZ, nOut=self.C, nc=self.latent, nf=nf, nMasks=self.nMask, selfAtt=False).to(device)
        self.RecZ = models.RecZ(sizex=self.SZ, nIn=self.C, nc=self.latent, nf=nf, nMasks=self.nMask).to(device)
        self.D = models.Discriminator(nIn=self.C, nf=nf, selfAtt=False).to(device)
        self.rd = torch.distributions.Normal(0, 1)

        self.optEncM = optim.Adam(self.EncM.parameters(), lr=1e-5, betas=(0, 0.9), weight_decay=1e-4, amsgrad=False)
        self.optGenX = optim.Adam(self.GenX.parameters(), lr=1e-4, betas=(0, 0.9), amsgrad=False)
        self.optRecZ = optim.Adam(self.RecZ.parameters(), lr=1e-4, betas=(0, 0.9), amsgrad=False)
        self.optD = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0, 0.9), amsgrad=False)

    def step(self,x,gstep=True,dstep=True):
        x = x.to(self.device)
        self.EncM.zero_grad()
        self.GenX.zero_grad()
        self.RecZ.zero_grad()
        self.D.zero_grad()
        self.EncM.train()
        self.GenX.train()
        self.RecZ.train()
        self.D.train()

        device = self.device
        batch = x.shape[0]
        z = self.rd.sample((batch,2,self.latent,1,1)).to(device)

        loss = [None, None]
        if gstep:
            mask = self.EncM(x)
            hGen = self.GenX(mask, z)
            xGen = (hGen + ((1 - mask.unsqueeze(2)) * x.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))
            dGen = self.D(xGen)
            lossG = - dGen.mean()
            zRec = self.RecZ(hGen.sum(1))
            
            err_recZ = ((z - zRec) * (z - zRec)).mean()
            lossG += err_recZ * 5
            loss[0] = lossG.item()
            lossG.backward()
            self.optEncM.step()
            self.optGenX.step()
            self.optRecZ.step()
        if dstep:
            self.D.zero_grad()
            with torch.no_grad():
                mask = self.EncM(x)
                hGen = self.GenX(mask, z)
                xGen = (hGen + ((1 - mask.unsqueeze(2)) * x.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))
            dPosX = self.D(x)
            dNegX = self.D(xGen)
            err_dPosX = (-1 + dPosX)
            err_dNegX = (-1 - dNegX)
            err_dPosX = ((err_dPosX < 0).float() * err_dPosX).mean()
            err_dNegX = ((err_dNegX < 0).float() * err_dNegX).mean()
            loss[1] = (-err_dPosX - err_dNegX).item()
            (-err_dPosX - err_dNegX).backward()
            self.optD.step()

        return loss
    def plot(self,x,name):
        x = x.to(self.device)
        self.EncM.eval()
        self.GenX.eval()
        self.RecZ.eval()
        self.D.eval()
        device = self.device
        batch = x.shape[0]
        z = self.rd.sample((batch,2,self.latent,1,1)).to(device)
        
        with torch.no_grad():
            mask = self.EncM(x)
            hGen = self.GenX(mask, z)
            xGen = (hGen + ((1 - mask.unsqueeze(2)) * x.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))

            imgs = xGen.cpu().numpy()
            SZ = self.SZ
            s = batch * self.nMask
            l = 8
            m = np.zeros((3,SZ*(s // l),SZ*l))
            for i in range(imgs.shape[0]):
                x = i // l
                y = i % l
                m[:,x*SZ:(x+1)*SZ,y*SZ:(y+1)*SZ] = imgs[i]
            m = m.transpose((1,2,0)).clip(0,1)
            
            plt.imsave(name,m)
