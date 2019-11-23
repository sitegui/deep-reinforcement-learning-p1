import numpy as np
import random
from collections import namedtuple, deque

from model import GaussianActorCriticNet
from normalizer import MeanStdNormalizer

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
                 discount=0.99,
                 use_gae=False,
                 gae_tau=0.95,
                 gradient_clip=1,
                 rollout_length=512,
                 optimization_epochs=10,
                 mini_batch_size=128,
                 ppo_ratio_clip=0.25,
                 entropy_weight=1e-3):

        # Store main params
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Store hyper-params
        self.discount = discount
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.gradient_clip = gradient_clip
        self.rollout_length = rollout_length
        self.optimization_epochs = optimization_epochs
        self.mini_batch_size = mini_batch_size
        self.ppo_ratio_clip = ppo_ratio_clip
        self.entropy_weight = entropy_weight

        # Create network
        self.network = GaussianActorCriticNet(state_dim, action_dim)
        self.network.to(device)
        self.state_normalizer = MeanStdNormalizer()
        self.optimizer = optim.Adam(self.network.parameters(), 1e-4)

        # Init environment and score tracking
        self.state = torch.tensor(self.state_normalizer(env.reset())).float().to(device)
        self.episodes = 1
        self.episode_score = 0
        self.scores = []
        self.scores_window = deque(maxlen=100)

    def train_step(self):
        # Prepare experience buffers ("l_" means list)
        l_state = []
        l_action = []
        l_log_prob = []
        l_value = []
        l_reward = []
        l_continue = []

        # Collect experience
        for _ in range(self.rollout_length):
            # Apply action
            prediction = self.network(self.state)
            action = prediction['action'].detach()
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            self.episode_score += np.mean(reward)

            # Rollover the end of episodes
            if np.all(done):
                next_state = self.env.reset()
                self.scores.append(self.episode_score)
                self.scores_window.append(self.episode_score)
                print(f'Episode {self.episodes}\tAverage Score: {np.mean(self.scores_window):.2f}')
                self.episodes += 1
                self.episode_score = 0

            # Store values
            l_state.append(self.state)
            l_action.append(action)
            l_log_prob.append(prediction['log_prob'].detach())
            l_value.append(prediction['value'].detach().cpu())
            l_reward.append(torch.tensor(reward).unsqueeze(-1))
            continue_ = 1 - torch.tensor(done).float()
            l_continue.append(continue_.unsqueeze(-1))

            # Prepare next step
            self.state = torch.tensor(self.state_normalizer(next_state)).float().to(device)

        # Predict final state value
        prediction = self.network(self.state)
        l_value.append(prediction['value'].detach().cpu())

        # Calculate return and advantage
        l_return = [None] * self.rollout_length
        l_advantage = [None] * self.rollout_length
        advantage = torch.tensor(np.zeros((20, 1)))
        return_ = l_value[-1]
        for i in reversed(range(self.rollout_length)):
            return_ = l_reward[i] + self.discount * l_continue[i] * return_
            if not self.use_gae:
                advantage = return_ - l_value[i]
            else:
                td_error = l_reward[i] + self.discount * l_continue[i] * l_value[i + 1] - l_value[i]
                advantage = advantage * self.gae_tau * self.discount * l_continue[i] + td_error
            l_advantage[i] = advantage
            l_return[i] = return_

        # Concat collect values ("t_" means tensor)
        t_state = torch.cat(l_state)
        t_action = torch.cat(l_action).to(device)
        t_log_prob_old = torch.cat(l_log_prob).to(device)
        t_return = torch.cat(l_return).to(device)
        t_advantage = torch.cat(l_advantage).to(device)

        # Normalize advantage
        t_advantage = (t_advantage - t_advantage.mean()) / t_advantage.std()

        # Train the network
        for _ in range(self.optimization_epochs):
            # Determine the batch indexes to use
            shuffled_indices = np.random.permutation(np.arange(self.rollout_length))
            used_indices = len(shuffled_indices) // self.mini_batch_size * self.mini_batch_size
            batches_indices = shuffled_indices[:used_indices].reshape(-1, self.mini_batch_size)

            for batch_indices in batches_indices:
                # Gather minibatch data
                batch_indices = torch.tensor(batch_indices).long()
                batch_state = t_state[batch_indices]
                batch_action = t_action[batch_indices]
                batch_log_prob_old = t_log_prob_old[batch_indices]
                batch_return = t_return[batch_indices]
                batch_advantage = t_advantage[batch_indices]

                # Calculate loss terms. This is mostly black-magic math I don't quite understand...
                prediction = self.network(batch_state, batch_action)
                ratio = (prediction['log_prob'] - batch_log_prob_old).exp()
                obj = ratio * batch_advantage
                obj_clipped = ratio.clamp(
                    1.0 - self.ppo_ratio_clip,
                    1.0 + self.ppo_ratio_clip) * batch_advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.entropy_weight * prediction['entropy'].mean()
                value_loss = 0.5 * (batch_return - prediction['value']).pow(2).mean()

                # Optimize network weights
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                self.optimizer.step()
