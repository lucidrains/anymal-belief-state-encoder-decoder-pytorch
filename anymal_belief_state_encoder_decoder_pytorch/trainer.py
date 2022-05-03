import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from collections import deque
from einops import rearrange

from anymal_belief_state_encoder_decoder_pytorch import Anymal

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

class StudentTrainer(nn.Module):
    def __init__(
        self,
        *,
        anymal,
        env,
        epochs = 2,
        lr = 5e-4,
        max_timesteps = 10000,
        update_timesteps = 5000,
        minibatch_size = 16,
        truncate_tpbtt = 10
    ):
        super().__init__()
        self.env = env
        self.anymal = anymal
        self.optimizer = Adam(anymal.student.parameters(), lr = lr)
        self.epochs = epochs

        self.max_timesteps = max_timesteps
        self.update_timesteps = update_timesteps
        self.minibatch_size = minibatch_size
        self.truncate_tpbtt = truncate_tpbtt

    def learn_from_memories(
        self,
        memories,
        next_states,
        noise_strength = 0.
    ):
        device = next(self.parameters()).device

        # retrieve and prepare data from memory for training

        states = []
        hiddens = []

        for (state, hidden) in memories:
            states.append(state)
            hiddens.append(hidden)

        states = tuple(zip(*states))

        # convert values to torch tensors

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = map(to_torch_tensor, states)
        hiddens = to_torch_tensor(hiddens)

        # prepare dataloader for policy phase training

        dl = create_shuffled_dataloader([*states, hiddens], self.minibatch_size)

        # policy phase training, similar to original PPO
        for _ in range(self.epochs):
            for ind, (proprio, extero, privileged, hiddens) in enumerate(dl):

                loss, hidden = self.anymal(
                    proprio,
                    extero,
                    privileged,
                    hiddens = hiddens,
                    noise_strength = noise_strength
                )

                loss.backward()

                if not ((ind + 1) % self.truncate_tpbtt): # how far back in time should the gradients go for recurrence
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def forward(
        self,
        noise_strength = 0.
    ):
        device = next(self.parameters()).device

        time = 0
        states = self.env.reset()
        memories = deque([])

        hidden = self.anymal.student.get_gru_hiddens()
        hidden = rearrange(hidden, 'l d -> 1 l d')

        for timestep in range(self.max_timesteps):
            time += 1

            states = list(map(lambda t: t.to(device), states))
            anymal_states = list(map(lambda t: rearrange(t, '... -> 1 ...'), states))

            memories.append((states, rearrange(hidden, '1 ... -> ...')))

            dist, hidden = self.anymal.forward_student(
                *anymal_states[:-1],
                hiddens = hidden,
                return_action_categorical_dist = True
            )

            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            next_states, _, done, _ = self.env.step(action)

            states = next_states

            if time % self.update_timesteps == 0:
                self.learn_from_memories(memories, next_states, noise_strength = noise_strength)
                memories.clear()

            if done:
                break
