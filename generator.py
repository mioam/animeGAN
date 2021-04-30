import numpy
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim: tuple, hidden_size=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_dim = img_dim  # (C, H, W)
        self.hidden_size = hidden_size
        # Layers
        self.latent_fc = nn.Linear(self.latent_dim, self.hidden_size)
        self.latent_bn = nn.BatchNorm1d(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.bn1 = nn.BatchNorm1d(2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, 4 * self.hidden_size)
        self.bn2 = nn.BatchNorm1d(4 * self.hidden_size)
        self.fc3 = nn.Linear(4 * self.hidden_size, int(np.prod(self.img_dim)))

        self._initialize(0., 0.02)

    def _initialize(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, z):
        z = F.relu(self.latent_bn(self.latent_fc(z)))
        latent = z
        out = F.relu(self.bn1(self.fc1(latent)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = torch.tanh(self.fc3(out))
        out = torch.reshape(out, (-1,) + self.img_dim)
        return out

    def sample(self, z):
        with torch.no_grad():
            out = self.forward(z)
            out = (out + 1) / 2 
        return out