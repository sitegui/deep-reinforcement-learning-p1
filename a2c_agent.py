import numpy as np
import random
from collections import namedtuple, deque

from model import GaussianActorCriticNet

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 discount=0.99,
                 use_gae=True,
                 gae_tau=1.0,
                 entropy_weight=0.01,
                 rollout_length=5,
                 gradient_clip=5,
                 max_steps=int(2e7)):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.entropy_weight = entropy_weight
        self.rollout_length = rollout_length
        self.gradient_clip = gradient_clip
        self.max_steps = max_steps

        self.network = GaussianActorCriticNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, back_prop_reward)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def end_episode(self):
        """ Signal the agent that the current episode has ended """
        self.memory.end_episode()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Calculate max_a(Q) for all samples using the target network
        with torch.no_grad():
            maxQ, _ = self.qnetwork_target(next_states).max(dim=1)
        maxQ = maxQ.unsqueeze(-1)

        # Calculate target values
        targets = rewards + gamma * (1 - dones) * maxQ

        # Optimize
        all_action_values = self.qnetwork_local(states)
        values = all_action_values.gather(1, actions)
        loss = self.loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, back_prop_reward):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            back_prop_reward (bool): whether to calculated expected reward of actions at the end of the episode
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.back_prop_reward = back_prop_reward
        self.episode_memory = []

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        if self.back_prop_reward:
            self.episode_memory.append(e)
        else:
            self.memory.append(e)

    def end_episode(self):
        """ Signal the agent that the current episode has ended """
        if self.back_prop_reward:
            # R_bar: Expected reward
            # R_bar(n) = R(n) + gamma * R_bar(n+1)
            expected_future_reward = 0
            for i in reversed(range(len(self.episode_memory))):
                mem = self.episode_memory[i]
                expected_reward = mem.reward + GAMMA * expected_future_reward
                self.episode_memory[i] = mem._replace(reward=expected_reward)
                expected_future_reward = expected_reward

        # Push episode experient to main memory
        self.memory.extend(self.episode_memory)
        self.episode_memory = []

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)