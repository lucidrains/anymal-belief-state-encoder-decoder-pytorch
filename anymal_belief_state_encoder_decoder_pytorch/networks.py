import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import GRU

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

# add basic MLP

class MLP(nn.Module):
    def __init__(self, dims, activation = nn.LeakyReLU):
        super().__init__()
        assert isinstance(dims, (list, tuple))
        dim_pairs = list(zip(dims[:-1], dims[1:]))
        *dim_pairs, dim_out_pair = dim_pairs

        layers = []
        for dim_in, dim_out in dim_pairs:
            layers.extend([
                nn.Linear(dim_in, dim_out),
                activation()
            ])

        layers.append(nn.Linear(*dim_out_pair))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = torch.cat(x, dim = -1)

        return self.net(x)

# they use basic PPO for the teacher

class PPO(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Student(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
