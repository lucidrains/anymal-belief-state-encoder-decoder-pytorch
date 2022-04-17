import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import GRUCell

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
    def __init__(
        self,
        num_actions,
        proprio_dim = 133,
        extero_dim = 52,  # in paper, height samples was marked as 208, but wasn't sure if that was per leg, or (4 legs x 52) = 208
        latent_extero_dim = 24,
        extero_encoder_hidden = (80, 60),
        belief_state_encoder_hiddens = (64, 64),
        extero_gate_encoder_hiddens = (64, 64),
        belief_state_dim = 120,  # should be equal to teacher's extero_dim + privileged_dim (part of the GRU's responsibility is to maintain a hidden state that forms an opinion on the privileged information)
        gru_num_layers = 2,
        gru_hidden_size = 50,
        mlp_hidden = (256, 160, 128),
        num_legs = 4,
        privileged_dim = 50,
        privileged_decoder_hiddens = (64, 64),
        extero_decoder_hiddens = (64, 64),
    ):
        super().__init__()
        assert belief_state_dim > (num_legs * latent_extero_dim)
        self.num_legs = num_legs
        self.proprio_dim = proprio_dim
        self.extero_dim = extero_dim        

        # encoding of exteroception

        self.extero_encoder = MLP((extero_dim, *extero_encoder_hidden, latent_extero_dim))

        # GRU related parameters

        gru_input_dim = (latent_extero_dim * num_legs) + proprio_dim
        gru_input_dims = (gru_input_dim, *((gru_hidden_size,) * (gru_num_layers - 1)))
        self.gru_cells = nn.ModuleList([GRUCell(input_dim, gru_hidden_size) for input_dim in gru_input_dims])

        # belief state encoding

        self.belief_state_encoder = MLP((gru_hidden_size, *belief_state_encoder_hiddens, belief_state_dim))

        # attention gating of exteroception

        self.to_latent_extero_attn_gate = MLP((gru_hidden_size, *extero_gate_encoder_hiddens, latent_extero_dim * num_legs))

        # belief state decoder

        self.privileged_decoder = MLP((gru_hidden_size, *privileged_decoder_hiddens, privileged_dim))
        self.extero_decoder = MLP((gru_hidden_size, *extero_decoder_hiddens, extero_dim * num_legs))

        self.to_extero_attn_gate = MLP((gru_hidden_size, *extero_gate_encoder_hiddens, extero_dim * num_legs))

        # final MLP to action logits

        self.to_action_logits = MLP((
            belief_state_dim,
            *mlp_hidden,
            num_actions
        ))

    def forward(
        self,
        proprio,
        extero,
        hiddens = None,
        return_estimated_info = False  # for returning estimated privileged info + exterceptive info, for reconstruction loss
    ):
        check_shape(proprio, 'b d', d = self.proprio_dim)
        check_shape(extero, 'b n d', n = self.num_legs, d = self.extero_dim)

        latent_extero = self.extero_encoder(extero)
        latent_extero = rearrange(latent_extero, 'b ... -> b (...)')

        # RNN

        if not exists(hiddens):
            hiddens = (None,) * len(self.gru_cells)

        gru_input = torch.cat((proprio, latent_extero), dim = -1)

        next_hiddens = []
        for gru_cell, prev_hidden in zip(self.gru_cells, hiddens):
            gru_input = gru_cell(gru_input, prev_hidden)
            next_hiddens.append(gru_input)

        gru_output = gru_input

        # attention gating of exteroception

        latent_extero_attn_gate = self.to_latent_extero_attn_gate(gru_output)
        gated_latent_extero = latent_extero * latent_extero_attn_gate.sigmoid()

        # belief state and add gated exteroception

        belief_state = self.belief_state_encoder(gru_output)
        belief_state = sum_with_zeropad(belief_state, gated_latent_extero)

        # to action logits

        action_logits = self.to_action_logits(belief_state)

        if not return_estimated_info:
            return action_logits, next_hiddens

        # belief state decoding
        # for reconstructing privileged and exteroception information from hidden belief states

        recon_privileged = self.privileged_decoder(gru_output)
        recon_extero = self.extero_decoder(gru_output)
        extero_attn_gate = self.to_extero_attn_gate(gru_output)

        gated_extero = rearrange(extero, 'b ... -> b (...)') * extero_attn_gate.sigmoid()
        recon_extero = recon_extero + gated_extero
        recon_extero = rearrange(recon_extero, 'b (n d) -> b n d', n = self.num_legs)

        return action_logits, hiddens, (recon_privileged, extero)

class Teacher(nn.Module):
    def __init__(
        self,
        num_actions,
        proprio_dim = 133,
        extero_dim = 52,  # in paper, height samples was marked as 208, but wasn't sure if that was per leg, or (4 legs x 52) = 208
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
        latent_extero = rearrange(latent_extero, 'b ... -> b (...)')

        latent_privileged = self.privileged_encoder(privileged)

        latent = torch.cat((
            proprio,
            latent_extero,
            latent_privileged,
        ), dim = -1)

        return self.to_actions_logits(latent)
