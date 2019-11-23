import numpy as np
import random
from collections import namedtuple, deque

from model import DeterministicActorCriticNet
from random_process import OrnsteinUhlenbeckProcess, GaussianProcess

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 env,
                 state_dim,
                 action_dim,
                 memory_size=int(1e4),
                 warm_up=int(1e3),
                 batch_size=128,
                 discount=0.995,
                 tau=1e-3,
                 gradient_clip=1,
                 random_process_class=GaussianProcess,
                 random_std=0.8,
                 random_std_decay=0.91,
                 update_every=5,
                 update_epochs=5):

        # Store main params
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Store hyper-params
        self.memory_size = memory_size
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.gradient_clip = gradient_clip
        self.random_process_class = random_process_class
        self.random_std = random_std
        self.random_std_decay = random_std_decay
        self.update_every = update_every
        self.update_epochs = update_epochs

        # Create networks
        self.network = DeterministicActorCriticNet(state_dim, action_dim)
        self.network.to(device)
        self.target_network = DeterministicActorCriticNet(state_dim, action_dim)
        self.target_network.to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.random_process = random_process_class(size=(action_dim,), std=random_std)

        # Init environment and score tracking
        self.state = torch.tensor(env.reset()).float().to(device)
        self.memory = Replay(self.memory_size)
        self.steps = 0
        self.episodes = 1
        self.episode_score = 0
        self.scores = []
        self.scores_window = deque(maxlen=100)

    def train_step(self):
        # Pick action
        action = self.network.actor(self.state).cpu().detach().numpy()
        action += self.random_process.sample()
        action = np.clip(action, -1, 1)

        # Apply to environment
        next_state, reward, done, _ = self.env.step(action)
        self.episode_score += np.mean(reward)

        # Rollover the end of episodes
        if np.any(done):
            self.random_process.reset_states()
            self.random_process.std *= self.random_std_decay
            next_state = self.env.reset()
            self.scores.append(self.episode_score)
            self.scores_window.append(self.episode_score)
            print(f'Episode {self.episodes}\tScore: {self.episode_score}\tAverage Score: {np.mean(self.scores_window):.2f}')
            self.episodes += 1
            self.episode_score = 0

        # Save experience
        next_state = torch.tensor(next_state).float().to(device)
        for experience in zip(self.state, action, reward, next_state, done):
            self.memory.append(experience)
        self.state = next_state

        self.steps += 1
        if len(self.memory) >= self.warm_up and self.steps % self.update_every == 0:
            for _ in range(self.update_epochs):
                self._learn_step()

    def _learn_step(self):
        # Sample from memory and convert to tensor
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.stack(states)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).unsqueeze(-1)
        next_states = torch.stack(next_states)
        mask = torch.tensor(1 - np.array(dones)).unsqueeze(-1)

        # Prepare critic loss
        next_actions = self.target_network.actor(next_states)
        next_qs = rewards + self.discount * mask * self.target_network.critic(next_states, next_actions).cpu()
        qs = self.network.critic(states, actions)
        critic_loss = F.mse_loss(qs, next_qs.detach().to(device))

        # Update critic
        self.network.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.network.critic_optimizer.step()

        # Prepare actor loss
        actor_loss = -self.network.critic(states, self.network.actor(states)).mean()

        # Update actor
        self.network.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.network.actor_optimizer.step()

        # Soft update network
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.tau) + param * self.tau)


class Replay:
    def __init__(self, size):
        self.samples = deque(maxlen=size)

    def append(self, exp):
        self.samples.append(exp)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.samples), batch_size)
        return zip(*[self.samples[i] for i in indices])

    def __len__(self):
        return len(self.samples)
