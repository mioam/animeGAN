import numpy
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, img_dim, hidden_size=256):
        super(Discriminator, self).__init__()
        self.img_dim = img_dim
        self.hidden_size = hidden_size
        # Layers
        self.img_fc = nn.Linear(int(np.prod(self.img_dim)), 4 * self.hidden_size)
        self.fc2 = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size)
        self.bn2 = nn.BatchNorm1d(2 * self.hidden_size)
        self.fc3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

        self._initialize(0., 0.02)

    def _initialize(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, img):
        img = torch.flatten(img, start_dim=1, end_dim=-1)
        x = F.leaky_relu(self.img_fc(img), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x)).squeeze(dim=-1)
        return x