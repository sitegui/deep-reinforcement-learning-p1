import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units=(64, 64), output_dim=1, gate=F.relu):
        super().__init__()
        dims = (input_dim,) + hidden_units + (output_dim,)
        self.layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out)
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        self.gate = gate

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        return self.layers[-1](x)


class GaussianActorCriticNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = DNN(state_dim, output_dim=action_dim, gate=torch.tanh)
        self.critic = DNN(state_dim, gate=torch.tanh)

        self.actor_params = self.actor.parameters()
        self.critic_params = self.critic.parameters()

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, action=None):
        mean = torch.tanh(self.actor(state))
        value = self.critic(state)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value
        }
