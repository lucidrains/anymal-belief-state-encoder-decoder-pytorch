from collections import namedtuple, deque

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from anymal_belief_state_encoder_decoder_pytorch import Anymal
from anymal_belief_state_encoder_decoder_pytorch.networks import unfreeze_all_layers_

from einops import rearrange

# they use basic PPO for training the teacher with privileged information
# then they used noisy student training, using the trained "oracle" teacher as guide

# ppo data

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])

class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

# ppo helper functions

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

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

class PPO(nn.Module):
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
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in memories:
            states.append(mem.state)
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        states = tuple(zip(*states))

        # calculate generalized advantage estimate

        next_states = map(lambda t: t.to(device), next_states)
        next_states = map(lambda t: rearrange(t, '... -> 1 ...'), next_states)

        _, next_value = self.anymal.forward_teacher(*next_states, return_value_head = True)
        next_value = next_value.detach()

        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # convert values to torch tensors

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = map(to_torch_tensor, states)
        actions = to_torch_tensor(actions)
        old_log_probs = to_torch_tensor(old_log_probs)

        old_values = to_torch_tensor(values[:-1])
        old_values = rearrange(old_values, '... 1 -> ...')

        rewards = torch.tensor(returns).float().to(device)

        # prepare dataloader for policy phase training

        dl = create_shuffled_dataloader([*states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

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
            action = action.item()

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
