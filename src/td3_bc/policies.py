from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn


class BaseActor(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, hidden_dim: int, n_layers: int, max_action: float
    ):
        super().__init__()
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_action = max_action

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass


class BaseCritic(nn.Module, ABC):
    def __init__(self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
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
    def __init__(
        self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, hidden_dim: int, n_layers: int, max_action: float
    ):
        super().__init__(obs_shape, action_dim, hidden_dim, n_layers, max_action)

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
        super().__init__(obs_shape, action_dim, hidden_dim, n_layers)

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


class CnnEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], hidden_dim: int):
        super().__init__()
        c = obs_shape[2]
        self.encoder = nn.Sequential(
            nn.Conv2d(c, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape).permute(0, 3, 1, 2)
            self.output_dim = self.encoder(dummy_input).shape[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.encoder(obs)


class CnnActor(BaseActor):
    def __init__(self, encoder: CnnEncoder, action_dim: int, hidden_dim: int, n_layers: int, max_action: float):
        super().__init__(encoder.output_dim, action_dim, hidden_dim, n_layers, max_action)
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
    def __init__(self, encoder: CnnEncoder, action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__(encoder.output_dim, action_dim, hidden_dim, n_layers)
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


def policy_factory(name: str, obs_dim: int, action_dim: int, max_action: float, device) -> Tuple[BaseActor, BaseCritic]:
    if name == "mlp":
        actor = MlpActor(obs_dim, action_dim, hidden_dim=256, n_layers=2, max_action=max_action).to(device)
        critic = MlpCritic(obs_dim, action_dim, hidden_dim=256, n_layers=2).to(device)
    elif name == "cnn":
        shared_encoder = CnnEncoder(obs_dim, hidden_dim=16).to(device)
        actor = CnnActor(shared_encoder, action_dim, hidden_dim=16, n_layers=1, max_action=max_action).to(device)
        critic = CnnCritic(shared_encoder, action_dim, hidden_dim=16, n_layers=1).to(device)
    else:
        raise ValueError(f"Unknown policy name: {name}")

    return actor, critic


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
