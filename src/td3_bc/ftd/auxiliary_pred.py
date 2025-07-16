import torch
import torch.nn as nn

from .modules import weight_init


class RewardPredictor(nn.Module):

    def __init__(self, encoder, action_dim, hidden_dim):
        super().__init__()

        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mlp.apply(weight_init)

    def forward(self, x, action):
        x = self.encoder(x)
        x = torch.cat([x, action], dim=1)
        x = self.mlp(x)

        return x


class InverseDynamicPredictor(nn.Module):

    def __init__(self, encoder, action_dim, hidden_dim):
        super().__init__()

        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.mlp.apply(weight_init)

    def forward(self, x, next_x):
        x = self.encoder(x)
        next_x = self.encoder(next_x)

        x = torch.cat((x, next_x), dim=1)
        x = self.mlp(x)

        return x
