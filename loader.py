import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms
import numpy as np

class AnimeFace(Dataset):
    def __init__(self,data:np.ndarray,transforms:nn.Module=None):
        self.transforms=transforms
        self.data = data
        self.len=data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.transforms(self.data[index])
SZ = 128

def _load(path = './data/', sz = (SZ,SZ)):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(sz),
        transforms.Normalize([0.5],[0.5])])
    dataset = torchvision.datasets.ImageFolder(path,transform=trans)
    print("Dataset loaded: %d imgs"%(len(dataset)))
    torch.save(dataset,path + 'dataset.pt')
    return dataset

def tonpy(path = './data/cropped/', sz = (SZ,SZ)):
    files = os.listdir(path)
    a  = []
    for name in files:
        img = Image.open(path + name)
        img = img.resize(sz)
        tmp = np.array(img).astype(np.uint8)
        a.append(tmp)
        # print(tmp.shape)
        # exit()
    a = np.stack(a,0)
    np.save('./data/dataset.npy',a)
    return a

def load(path = './data/'):
    d = np.load(path + 'dataset.npy')
    dataset = AnimeFace(d,transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])]))
    print("Dataset loaded: %d imgs"%(len(dataset)))
    return dataset

def check(path = './data/cropped/'):
    files = os.listdir(path)
    for name in files:
        img = Image.open(path + name)
        if img.size[0] < 90:
            img.close()
            os.remove(path + name)

if __name__ == "__main__":
    # check()
    dataset = tonpy()
