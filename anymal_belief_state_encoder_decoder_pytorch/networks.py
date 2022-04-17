import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import GRU

from einops import rearrange
from einops_exts import check_shape

# helper functions

def exists(val):
    return val is not None

# in the paper
# the network attention gates the exteroception, and then sums it to the belief state
# but zero pads since they have different feature dimensions
# not sure why they didn't project to the same dimensions

def sum_with_zeropad(x, y):
    x_dim, y_dim = x.shape[-1], y.shape[-1]

    if x_dim == y_dim:
        return x + y

    if x_dim < y_dim:
        x = F.pad(x, (y_dim - x_dim, 0))

    if y_dim < x_dim:
        y = F.pad(y, (x_dim - y_dim, 0))

    return x + y

# add basic MLP

class MLP(nn.Module):
    def __init__(self, dims, activation = nn.LeakyReLU):
        super().__init__()
        assert isinstance(dims, (list, tuple))
        assert len(dims) > 2, 'must have at least 3 dimensions (input, *hiddens, output)'

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

# they use basic PPO for training the teacher with privileged information
# then they used noisy student training, using the trained "oracle" teacher as guide

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
    def __init__(
        self,
        num_actions,
        proprio_dim = 133,
        extero_dim = 208,
        latent_extero_dim = 24,
        extero_encoder_hidden = (80, 60),
        privileged_dim = 50,
        latent_privileged_dim = 24,
        privileged_encoder_hidden = (64, 32),
        mlp_hidden = (256, 160, 128),
        num_legs = 4
    ):
        super().__init__()
        self.num_legs = num_legs
        self.proprio_dim = proprio_dim
        self.extero_dim = extero_dim
        self.privileged_dim = privileged_dim

        self.extero_encoder = MLP((extero_dim, *extero_encoder_hidden, latent_extero_dim))
        self.privileged_encoder = MLP((privileged_dim, *privileged_encoder_hidden, latent_privileged_dim))

        self.to_actions_logits = MLP((
            latent_extero_dim * num_legs + latent_privileged_dim + proprio_dim,
            *mlp_hidden,
            num_actions
        ))

    def forward(
        self,
        proprio,
        extero,
        privileged
    ):
        check_shape(proprio, 'b d', d = self.proprio_dim)
        check_shape(extero, 'b n d', n = self.num_legs, d = self.extero_dim)
        check_shape(privileged, 'b d', d = self.privileged_dim)

        latent_extero = self.extero_encoder(extero)
        latent_privileged = self.privileged_encoder(privileged)

        latent = torch.cat((
            proprio,
            rearrange(latent_extero, 'b ... -> b (...)'),
            latent_privileged,
        ), dim = -1)

        return self.to_actions_logits(latent)
