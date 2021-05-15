import torch
from torch.utils.data import DataLoader

import loader
import model


SZ = 128
batch_size = 32

# label_dim = 10
img_dim = (3, SZ, SZ)


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = loader.load()
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
ReDO = model.ReDO(device)


num_epochs = 20
for epoch in range(num_epochs):
    for data in dataloader:
        ReDO.plot(data,'./imgs/%d.png'%(epoch))
        break
    exit()
    for it, data in enumerate(dataloader):
        loss0, loss1 = ReDO.step(data)
        if it % 100 == 0:
            print('epoch: %d, iter: %d, loss0: %f, loss1: %f'%(epoch,it,loss0,loss1))
    
