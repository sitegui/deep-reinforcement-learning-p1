import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units=(512, 256), output_dim=1, gate=F.relu):
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
            action = dist.sample().clamp(-1, 1)
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value
        }


class DeterministicActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, h1_size=256, h2_size=512):
        super().__init__()

        self.actor_h1 = layer_init(nn.Linear(state_dim, h1_size), 1e-1)
        self.actor_h2 = layer_init(nn.Linear(h1_size, h2_size), 1e-1)
        self.actor_out = layer_init(nn.Linear(h2_size, action_dim), 1e-3)
        actor_params = [*self.actor_h1.parameters(), *self.actor_h2.parameters(), *self.actor_out.parameters()]
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=1e-3)

        self.critic_h1 = layer_init(nn.Linear(state_dim, h1_size), 1e-1)
        self.critic_h2 = layer_init(nn.Linear(h1_size + action_dim, h2_size), 1e-1)
        self.critic_out = layer_init(nn.Linear(h2_size, 1), 1e-3)
        critic_params = [*self.critic_h1.parameters(), *self.critic_h2.parameters(), *self.critic_out.parameters()]
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=1e-4)

    def actor(self, state):
        x = F.relu(self.actor_h1(state))
        x = F.relu(self.actor_h2(x))
        return torch.tanh(self.actor_out(x))

    def critic(self, state, action):
        x = F.relu(self.critic_h1(state))
        x = F.relu(self.critic_h2(torch.cat([x, action], dim=1)))
        return self.critic_out(x)


def layer_init(layer, w_scale):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
