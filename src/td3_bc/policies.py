from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseActor(nn.Module, ABC):
    @abstractmethod
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_layers: int, max_action: float):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_action = max_action

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass


class BaseCritic(nn.Module, ABC):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    @abstractmethod
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass


class MlpActor(BaseActor):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_layers: int, max_action: float):
        super().__init__(obs_dim, action_dim, hidden_dim, n_layers, max_action)

        self.model = nn.Sequential(
            *[
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            ]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs) * self.max_action


class MlpCritic(BaseCritic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__(obs_dim, action_dim, hidden_dim, n_layers)

        self.critic1 = nn.Sequential(
            *[
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                *[layer for _ in range(n_layers - 1) for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
                nn.Linear(hidden_dim, 1),
            ]
        )

        self.critic2 = nn.Sequential(
            *[
                nn.Linear(obs_dim + action_dim, hidden_dim),
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


class CnnActor(BaseActor):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_layers: int, max_action: float):
        super().__init__(obs_dim, action_dim, hidden_dim, n_layers, max_action)

        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass


class CnnCritic(BaseCritic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__(obs_dim, action_dim, hidden_dim, n_layers)

        pass

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    def q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass


def policy_factory(name: str, obs_dim: int, action_dim: int, max_action: float, device) -> Tuple[BaseActor, BaseCritic]:
    if name == "mlp":
        actor = MlpActor(obs_dim, action_dim, hidden_dim=256, n_layers=2, max_action=max_action)
        critic = MlpCritic(obs_dim, action_dim, hidden_dim=256, n_layers=2)
    elif name == "cnn":
        actor = CnnActor(obs_dim, action_dim, hidden_dim=256, n_layers=2, max_action=max_action)
        critic = CnnCritic(obs_dim, action_dim, hidden_dim=256, n_layers=2)
    else:
        raise ValueError(f"Unknown policy name: {name}")

    return actor.to(device), critic.to(device)


if __name__ == "__main__":
    # Example usage
    obs_dim = 3
    action_dim = 4
    hidden_dim = 64
    n_layers = 2
    max_action = 1.0

    batch_size = 32

    # Create actor and critic
    actor = MlpActor(obs_dim, action_dim, hidden_dim, n_layers, max_action)
    critic = MlpCritic(obs_dim, action_dim, hidden_dim, n_layers)

    obs = torch.randn(batch_size, obs_dim)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, action)
    action = actor(obs)

    print("Action:", action.shape)
    print("Q1:", q1.shape)
    print("Q2:", q2.shape)
