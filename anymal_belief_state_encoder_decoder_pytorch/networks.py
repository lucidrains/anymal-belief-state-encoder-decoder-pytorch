import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import GRUCell
from torch.distributions import Categorical
from torch.optim import Adam

from einops import rearrange
from einops_exts import check_shape
from einops.layers.torch import Rearrange

from anymal_belief_state_encoder_decoder_pytorch.running import RunningStats

# helper functions

def exists(val):
    return val is not None

# freezing of neural networks (teacher needs to be frozen)

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# in the paper
# the network attention gates the exteroception, and then sums it to the belief state
# todo: make sure the padding is on the right side

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
    def __init__(
        self,
        dims,
        activation = nn.LeakyReLU,
        final_activation = False
    ):
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

        if final_activation:
            layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = torch.cat(x, dim = -1)

        return self.net(x)

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
        self.gru_hidden_size = gru_hidden_size

        # belief state encoding

        self.belief_state_encoder = MLP((gru_hidden_size, *belief_state_encoder_hiddens, belief_state_dim))

        # attention gating of exteroception

        self.to_latent_extero_attn_gate = MLP((gru_hidden_size, *extero_gate_encoder_hiddens, latent_extero_dim * num_legs))

        # belief state decoder

        self.privileged_decoder = MLP((gru_hidden_size, *privileged_decoder_hiddens, privileged_dim))
        self.extero_decoder = MLP((gru_hidden_size, *extero_decoder_hiddens, extero_dim * num_legs))

        self.to_extero_attn_gate = MLP((gru_hidden_size, *extero_gate_encoder_hiddens, extero_dim * num_legs))

        # final MLP to action logits

        self.to_logits = MLP((
            belief_state_dim + proprio_dim,
            *mlp_hidden
        ))

        self.to_action_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden[-1], num_actions)
        )

    def get_gru_hiddens(self):
        device = next(self.parameters()).device
        return torch.zeros((len(self.gru_cells), self.gru_hidden_size))

    def forward(
        self,
        proprio,
        extero,
        hiddens = None,
        return_estimated_info = False,  # for returning estimated privileged info + exterceptive info, for reconstruction loss
        return_action_categorical_dist = False
    ):
        check_shape(proprio, 'b d', d = self.proprio_dim)
        check_shape(extero, 'b n d', n = self.num_legs, d = self.extero_dim)

        latent_extero = self.extero_encoder(extero)
        latent_extero = rearrange(latent_extero, 'b ... -> b (...)')

        # RNN

        if not exists(hiddens):
            prev_hiddens = (None,) * len(self.gru_cells)
        else:
            prev_hiddens = hiddens.unbind(dim = -2)

        gru_input = torch.cat((proprio, latent_extero), dim = -1)

        next_hiddens = []
        for gru_cell, prev_hidden in zip(self.gru_cells, prev_hiddens):
            gru_input = gru_cell(gru_input, prev_hidden)
            next_hiddens.append(gru_input)

        gru_output = gru_input

        next_hiddens = torch.stack(next_hiddens, dim = -2)

        # attention gating of exteroception

        latent_extero_attn_gate = self.to_latent_extero_attn_gate(gru_output)
        gated_latent_extero = latent_extero * latent_extero_attn_gate.sigmoid()

        # belief state and add gated exteroception

        belief_state = self.belief_state_encoder(gru_output)
        belief_state = sum_with_zeropad(belief_state, gated_latent_extero)

        # to action logits

        belief_state_with_proprio = torch.cat((
            proprio,
            belief_state,
        ), dim = 1)

        logits = self.to_logits(belief_state_with_proprio)

        pi_logits = self.to_action_head(logits)

        return_action = Categorical(pi_logits.softmax(dim = -1)) if return_action_categorical_dist else pi_logits

        if not return_estimated_info:
            return return_action, next_hiddens

        # belief state decoding
        # for reconstructing privileged and exteroception information from hidden belief states

        recon_privileged = self.privileged_decoder(gru_output)
        recon_extero = self.extero_decoder(gru_output)
        extero_attn_gate = self.to_extero_attn_gate(gru_output)

        gated_extero = rearrange(extero, 'b ... -> b (...)') * extero_attn_gate.sigmoid()
        recon_extero = recon_extero + gated_extero
        recon_extero = rearrange(recon_extero, 'b (n d) -> b n d', n = self.num_legs)

        # whether to return raw policy logits or action probs wrapped with Categorical

        return return_action, next_hiddens, (recon_privileged, recon_extero)

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

        self.to_logits = MLP((
            latent_extero_dim * num_legs + latent_privileged_dim + proprio_dim,
            *mlp_hidden
        ))

        self.to_action_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden[-1], num_actions)
        )

        self.to_value_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden[-1], 1),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        proprio,
        extero,
        privileged,
        return_value_head = False,
        return_action_categorical_dist = False
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

        logits = self.to_logits(latent)

        pi_logits = self.to_action_head(logits)

        if not return_value_head:
            return pi_logits

        value_logits = self.to_value_head(logits)

        return_action = Categorical(pi_logits.softmax(dim = -1)) if return_action_categorical_dist else pi_logits
        return return_action, value_logits

# manages both teacher and student under one module

class Anymal(nn.Module):
    def __init__(
        self,
        num_actions,
        proprio_dim = 133,
        extero_dim = 52,
        privileged_dim = 50,
        num_legs = 4,
        latent_extero_dim = 24,
        latent_privileged_dim = 24,
        teacher_extero_encoder_hidden = (80, 60),
        teacher_privileged_encoder_hidden = (64, 32),
        student_extero_gate_encoder_hiddens = (64, 64),
        student_belief_state_encoder_hiddens = (64, 64),
        student_belief_state_dim = 120,
        student_gru_num_layers = 2,
        student_gru_hidden_size = 50,
        student_privileged_decoder_hiddens = (64, 64),
        student_extero_decoder_hiddens = (64, 64),
        student_extero_encoder_hidden = (80, 60),
        mlp_hidden = (256, 160, 128),
        recon_loss_weight = 0.5
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.num_legs = num_legs
        self.extero_dim = extero_dim

        self.student = Student(
            num_actions = num_actions,
            proprio_dim = proprio_dim,
            extero_dim = extero_dim,
            latent_extero_dim = latent_extero_dim,
            extero_encoder_hidden = student_extero_encoder_hidden,
            belief_state_encoder_hiddens = student_belief_state_encoder_hiddens,
            extero_gate_encoder_hiddens = student_extero_gate_encoder_hiddens,
            belief_state_dim = student_belief_state_dim,
            gru_num_layers = student_gru_num_layers,
            gru_hidden_size = student_gru_hidden_size,
            mlp_hidden = mlp_hidden,
            num_legs = num_legs,
            privileged_dim = privileged_dim,
            privileged_decoder_hiddens = student_privileged_decoder_hiddens,
            extero_decoder_hiddens = student_extero_decoder_hiddens,
        )

        self.teacher = Teacher(
            num_actions = num_actions,
            proprio_dim = proprio_dim,
            extero_dim = extero_dim,
            latent_extero_dim = latent_extero_dim,
            extero_encoder_hidden = teacher_extero_encoder_hidden,
            privileged_dim = privileged_dim,
            latent_privileged_dim = latent_privileged_dim,
            privileged_encoder_hidden = teacher_privileged_encoder_hidden,
            mlp_hidden = mlp_hidden,
            num_legs = num_legs
        )

        self.recon_loss_weight = recon_loss_weight

    def get_observation_running_stats(self):
        return RunningStats(self.proprio_dim), RunningStats((self.num_legs, self.extero_dim))

    def init_student_with_teacher(self):
        self.student.extero_encoder.load_state_dict(self.teacher.extero_encoder.state_dict())
        self.student.to_logits.load_state_dict(self.teacher.to_logits.state_dict())
        self.student.to_action_head.load_state_dict(self.teacher.to_action_head.state_dict())

    def forward_teacher(self, *args, return_value_head = False, **kwargs):
        return self.teacher(*args, return_value_head = return_value_head, **kwargs)

    def forward_student(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    # main forward for training the student with teacher as guide

    def forward(
        self,
        proprio,
        extero,
        privileged,
        teacher_states = None,
        hiddens = None,
        noise_strength = 0.1
    ):
        self.teacher.eval()
        freeze_all_layers_(self.teacher)

        with torch.no_grad():
            teacher_proprio, teacher_extero = teacher_states if exists(teacher_states) else (proprio, extero)
            teacher_action_logits = self.forward_teacher(teacher_proprio, teacher_extero, privileged)

        noised_extero = extero + torch.rand_like(extero) * noise_strength

        student_action_logits, hiddens, recons = self.student(proprio, noised_extero, hiddens = hiddens, return_estimated_info = True)

        # calculate reconstruction loss of privileged and denoised exteroception

        (recon_privileged, recon_extero) = recons
        recon_loss = F.mse_loss(recon_privileged, privileged) + F.mse_loss(recon_extero, extero)

        # calculate behavior loss, which is also squared distance?

        behavior_loss = F.mse_loss(teacher_action_logits, student_action_logits) # why not kl div on action probs?

        loss = behavior_loss + recon_loss * self.recon_loss_weight
        return loss, hiddens
