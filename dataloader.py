import os
from PIL import Image
import torchvision
from torchvision import transforms



def load(path = './data/', sz = (40,40)):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(sz),
        transforms.Normalize([0.5],[0.5])])
    dataset = torchvision.datasets.ImageFolder(path,transform=trans)
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
    dataset = load()
    for img, x in dataset:
        print(img.shape)