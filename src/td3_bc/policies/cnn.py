import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

from .policy import PolicyConfig, BaseActor, BaseCritic


@PolicyConfig.register_subclass("cnn")
@dataclass
class CnnPolicyConfig(PolicyConfig):
    encoder_hidden_dim: int = 16

    critic_hidden_dim: int = 16
    critic_n_layers: int = 1

    actor_hidden_dim: int = 16
    actor_n_layers: int = 1


class CnnEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], hidden_dim: int):
        super().__init__()

        self.input_shape = (
            int(torch.tensor(obs_shape[:-2]).prod().item()),
            *obs_shape[-2:],
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[0], hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output dimension
        with torch.no_grad():
            dummy_input = torch.randn((1, *self.input_shape), device=next(self.parameters()).device)
            self.output_dim = self.encoder(dummy_input).shape[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.reshape(obs, (-1, *self.input_shape))
        return self.encoder(obs)


class CnnActor(BaseActor):
    def __init__(
        self,
        encoder: CnnEncoder,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        hidden_dim: int,
        n_layers: int,
        max_action: float,
    ):
        super().__init__(obs_shape, action_dim, max_action)
        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        action = self.fc(features) * self.max_action
        return action


class CnnCritic(BaseCritic):
    def __init__(
        self, encoder: CnnEncoder, obs_shape: Tuple[int, int, int], action_dim: int, hidden_dim: int, n_layers: int
    ):
        super().__init__(obs_shape, action_dim)
        self.encoder = encoder

        self.action_encoder1 = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.action_encoder2 = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.critic1 = nn.Sequential(
            nn.Linear(encoder.output_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
            nn.Linear(hidden_dim, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(encoder.output_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_feat = self.encoder(obs)
        act_feat = self.action_encoder1(action)
        return self.critic1(torch.cat([obs_feat, act_feat], dim=-1))

    def q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_feat = self.encoder(obs)
        act_feat = self.action_encoder2(action)
        return self.critic2(torch.cat([obs_feat, act_feat], dim=-1))


if __name__ == "__main__":
    # Example usage
    obs_shape = (256, 256, 3)
    action_dim = 4
    hidden_dim = 64
    n_layers = 2
    max_action = 1.0

    batch_size = 32

    # Create actor and critic
    encoder = CnnEncoder(obs_shape, hidden_dim)
    actor = CnnActor(encoder, obs_shape, action_dim, hidden_dim, n_layers, max_action)
    critic = CnnCritic(encoder, obs_shape, action_dim, hidden_dim, n_layers)

    obs = torch.randn(batch_size, *obs_shape)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, action)
    action = actor(obs)

    print("Action:", action.shape)
    print("Q1:", q1.shape)
    print("Q2:", q2.shape)
