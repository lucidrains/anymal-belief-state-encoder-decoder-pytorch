from collections import namedtuple, deque

import torch
from torch import nn, cat, stack
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam

from assoc_scan import AssocScan

from anymal_belief_state_encoder_decoder_pytorch import Anymal
from anymal_belief_state_encoder_decoder_pytorch.networks import unfreeze_all_layers_

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

# they use basic PPO for training the teacher with privileged information
# then they used noisy student training, using the trained "oracle" teacher as guide

# ppo data

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])

def create_shuffled_dataloader(data, batch_size):
    ds = TensorDataset(*data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

# ppo helper functions

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

# generalized advantage estimate

def calc_generalized_advantage_estimate(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    device, is_cuda = rewards.device, rewards.is_cuda
    use_accelerated = default(use_accelerated, is_cuda)

    values, values_next = values[:-1], values[1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)
    return gae + values

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

# mock environment

class MockEnv(object):
    def __init__(
        self,
        proprio_dim,
        extero_dim,
        privileged_dim,
        num_legs = 4
    ):
        self.proprio_dim = proprio_dim
        self.extero_dim = extero_dim
        self.privileged_dim = privileged_dim
        self.num_legs = num_legs

    def rand_state(self):
        return (
            torch.randn((self.proprio_dim,)),
            torch.randn((self.num_legs, self.extero_dim,)),
            torch.randn((self.privileged_dim,))
        )

    def reset(self):
        return self.rand_state()

    def step(self, action):
        reward = torch.randn((1,))
        done = torch.tensor([False])
        return self.rand_state(), reward, done, None

# main ppo class

class PPO(Module):
    def __init__(
        self,
        *,
        env,
        anymal,
        epochs = 2,
        lr = 5e-4,
        betas = (0.9, 0.999),
        eps_clip = 0.2,
        beta_s = 0.005,
        value_clip = 0.4,
        max_timesteps = 10000,
        update_timesteps = 5000,
        lam = 0.95,
        gamma = 0.99,
        minibatch_size = 8300
    ):
        super().__init__()
        assert isinstance(anymal, Anymal)
        self.env = env
        self.anymal = anymal

        self.minibatch_size = minibatch_size
        self.optimizer = Adam(anymal.teacher.parameters(), lr = lr, betas = betas)
        self.epochs = epochs

        self.max_timesteps = max_timesteps
        self.update_timesteps = update_timesteps

        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.lam = lam
        self.gamma = gamma

        # in paper, they said observations fed to teacher were normalized
        # by running mean

        self.running_proprio, self.running_extero = anymal.get_observation_running_stats()

    def learn_from_memories(
        self,
        memories,
        next_states
    ):
        device = next(self.parameters()).device

        # retrieve and prepare data from memory for training

        (
            states,
            actions,
            old_log_probs,
            rewards,
            dones,
            values
        ) = tuple(zip(*memories))

        states = tuple(zip(*states))

        # calculate generalized advantage estimate

        rewards = cat(rewards).to(device)
        values = cat(values).to(device).detach()
        masks = 1. - cat(dones).to(device).float()

        next_states = [t.to(device) for t in next_states]
        next_states = [rearrange(t, '... -> 1 ...') for t in next_states]

        with torch.no_grad():
            self.anymal.eval()
            _, next_value = self.anymal.forward_teacher(*next_states, return_value_head = True)
            next_value = next_value.detach()

        values_with_next = cat((values, next_value))

        returns = calc_generalized_advantage_estimate(rewards, values_with_next, masks, self.gamma, self.lam).detach()

        # convert values to torch tensors

        to_torch_tensor = lambda t: stack(t).to(device).detach()

        states = map(to_torch_tensor, states)
        actions = to_torch_tensor(actions)
        old_log_probs = to_torch_tensor(old_log_probs)

        # prepare dataloader for policy phase training

        dl = create_shuffled_dataloader([*states, actions, old_log_probs, rewards, values], self.minibatch_size)

        # policy phase training, similar to original PPO

        for _ in range(self.epochs):
            for proprio, extero, privileged, actions, old_log_probs, rewards, old_values in dl:

                dist, values = self.anymal.forward_teacher(
                    proprio, extero, privileged,
                    return_value_head = True,
                    return_action_categorical_dist = True
                )

                action_log_probs = dist.log_prob(actions)

                entropy = dist.entropy()
                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                (policy_loss.mean() + value_loss.mean()).backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    # does one episodes worth of learning

    def forward(self):
        device = next(self.parameters()).device
        unfreeze_all_layers_(self.anymal)

        time = 0
        states = self.env.reset() # states assumed to be (proprioception, exteroception, privileged information)
        memories = deque([])

        self.running_proprio.clear()
        self.running_extero.clear()

        for timestep in range(self.max_timesteps):
            time += 1

            states = list(map(lambda t: t.to(device), states))
            proprio, extero, privileged = states

            # update running means for observations, for teacher

            self.running_proprio.push(proprio)
            self.running_extero.push(extero)

            # normalize observation states for teacher (proprio and extero)

            states = (
                self.running_proprio.norm(proprio),
                self.running_extero.norm(extero),
                privileged
            )

            anymal_states = list(map(lambda t: rearrange(t, '... -> 1 ...'), states))

            dist, values = self.anymal.forward_teacher(
                *anymal_states,
                return_value_head = True,
                return_action_categorical_dist = True
            )

            action = dist.sample()
            action_log_prob = dist.log_prob(action)

            next_states, reward, done, _ = self.env.step(action)

            memory = Memory(states, action, action_log_prob, reward, done, values)
            memories.append(memory)

            states = next_states

            if time % self.update_timesteps == 0:
                self.learn_from_memories(memories, next_states)
                memories.clear()

            if done:
                break

        print('trained for 1 episode')
