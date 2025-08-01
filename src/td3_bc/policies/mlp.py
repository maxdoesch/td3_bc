import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Tuple

from .policy import PolicyConfig, BaseActor, BaseCritic


@PolicyConfig.register_subclass("mlp")
@dataclass
class MlpPolicyConfig(PolicyConfig):
    critic_hidden_dim: int = 256
    critic_n_layers: int = 2

    actor_hidden_dim: int = 256
    actor_n_layers: int = 2


class MlpActor(BaseActor):
    def __init__(
        self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, hidden_dim: int, n_layers: int, max_action: float
    ):
        super().__init__(obs_shape, action_dim, max_action)

        assert isinstance(obs_shape, int) or len(obs_shape) == 1, "MLP Policy requires a 1D observation shape."

        self.obs_dim = obs_shape if isinstance(obs_shape, int) else obs_shape[0]

        self.model = nn.Sequential(
            *[
                nn.Linear(self.obs_dim, hidden_dim),
                nn.ReLU(),
                *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            ]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs) * self.max_action


class MlpCritic(BaseCritic):
    def __init__(self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__(obs_shape, action_dim)

        assert isinstance(obs_shape, int) or len(obs_shape) == 1, "MLP Critic requires a 1D observation shape."

        self.obs_dim = obs_shape if isinstance(obs_shape, int) else obs_shape[0]

        self.critic1 = nn.Sequential(
            *[
                nn.Linear(self.obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
                nn.Linear(hidden_dim, 1),
            ]
        )

        self.critic2 = nn.Sequential(
            *[
                nn.Linear(self.obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
                nn.Linear(hidden_dim, 1),
            ]
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        return q1, q2

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, action], dim=-1)
        return self.critic1(sa)

    def q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, action], dim=-1)
        return self.critic2(sa)


if __name__ == "__main__":
    # Example usage
    obs_shape = (3,)
    action_dim = 4
    hidden_dim = 64
    n_layers = 2
    max_action = 1.0

    batch_size = 32

    # Create actor and critic
    actor = MlpActor(obs_shape, action_dim, hidden_dim, n_layers, max_action)
    critic = MlpCritic(obs_shape, action_dim, hidden_dim, n_layers)

    obs = torch.randn(batch_size, *obs_shape)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, action)
    action = actor(obs)

    print("Action:", action.shape)
    print("Q1:", q1.shape)
    print("Q2:", q2.shape)
