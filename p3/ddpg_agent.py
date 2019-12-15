import numpy as np
import random
from collections import namedtuple, deque
import json
import os
import time

from model import DeterministicActorCriticNet
from random_process import GaussianProcess, OUProcess

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self,
                 env,
                 state_dim,
                 action_dim,
                 params):

        # Store main params
        self.params = params
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = params['name']
        self.warm_up = params['warm_up']
        self.random_std_decay = params['random_std_decay']
        self.update_every = params['update_every']
        self.update_epochs = params['update_epochs']
        self.batch_size = params['batch_size']
        self.discount = params['discount']
        self.tau = params['tau']
        self.gradient_clip = params['gradient_clip']

        # Create networks
        h1_size = params['h1_size']
        h2_size = params['h2_size']
        actor_lr = params['actor_lr']
        critic_lr = params['critic_lr']
        self.network = DeterministicActorCriticNet(state_dim, action_dim, h1_size, h2_size, actor_lr, critic_lr)
        self.network.to(device)
        self.target_network = DeterministicActorCriticNet(state_dim, action_dim, h1_size, h2_size, actor_lr, critic_lr)
        self.target_network.to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        if params['random_process'] == 'gaussian':
            self.random_process = GaussianProcess(size=(action_dim,), std=params['random_std'])
        else:
            self.random_process = OUProcess(size=(action_dim,), std=params['random_std'], theta=params['random_theta'])

        # Init environment and score tracking
        self.state = torch.tensor(env.reset()).float().to(device)
        self.memory = Replay(params['memory_size'])
        self.steps = 0
        self.episodes = 1
        self.episode_scores = np.zeros((2,))
        self.scores = []
        self.short_scores_window = deque(maxlen=10)
        self.scores_window = deque(maxlen=100)
        self.start_time = time.time()

    def train_step(self):
        # Pick action
        action = self.network.actor(self.state).cpu().detach().numpy()
        action += self.random_process.sample()
        action = np.clip(action, -1, 1)

        # Apply to environment
        next_state, reward, done, _ = self.env.step(action)
        self.episode_scores += reward

        # Rollover the end of episodes
        if np.any(done):
            self.random_process.reset_states()
            self.random_process.std *= self.random_std_decay
            next_state = self.env.reset()
            episode_score = np.max(self.episode_scores)
            self.scores.append(episode_score)
            self.short_scores_window.append(episode_score)
            self.scores_window.append(episode_score)
            dt = (time.time() - self.start_time)/60
            print(
                f'Time: {dt:.1f}min\tEpisode {self.episodes}\tScore: {episode_score:.2f}\tAvg 10: {np.mean(self.short_scores_window):.2f}\tAvg 100: {np.mean(self.scores_window):.2f}\tRand std: {self.random_process.std:.2f}')
            self.episodes += 1
            self.episode_scores = np.zeros((2,))

        # Save experience
        next_state = torch.tensor(next_state).float().to(device)
        for experience in zip(self.state, action, reward, next_state, done):
            self.memory.append(experience)
        self.state = next_state

        self.steps += 1
        if len(self.memory) >= self.warm_up and len(self.memory) >= self.batch_size and self.steps % self.update_every == 0:
            for _ in range(self.update_epochs):
                self._learn_step()

    def _learn_step(self):
        # Sample from memory and convert to tensor
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.stack(states)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).unsqueeze(-1)
        next_states = torch.stack(next_states)
        mask = torch.tensor(1. - np.array(dones)).unsqueeze(-1).float()

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

    def save(self):
        model_dir = f'models/{self.name}'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.network, f'{model_dir}/weights.pth')
        json.dump(self.params, open(f'{model_dir}/params.json', 'w'))
        json.dump(self.scores, open(f'{model_dir}/scores.json', 'w'))


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
