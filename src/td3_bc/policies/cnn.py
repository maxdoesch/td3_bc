import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, List

from .policy import PolicyConfig, BaseActor, BaseCritic
from .mlp import MlpActor, MlpCritic
from .utils import get_out_shape, weight_init


@dataclass
class CnnEncoderConfig:
    cnn_filter_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    cnn_kernel_strides: List[int] = field(default_factory=lambda: [2, 2, 1, 1])


@dataclass
class CnnActorConfig:
    linear_trunk_dim: int = 64

    hidden_dim: int = 256
    n_layers: int = 2


@dataclass
class CnnCriticConfig:
    linear_trunk_dim: int = 64

    hidden_dim: int = 256
    n_layers: int = 2


@PolicyConfig.register_subclass("cnn")
@dataclass
class CnnPolicyConfig(PolicyConfig):
    cnn_encoder_cfg: CnnEncoderConfig = field(default_factory=CnnEncoderConfig)

    actor_cfg: CnnActorConfig = field(default_factory=CnnActorConfig)
    critic_cfg: CnnCriticConfig = field(default_factory=CnnCriticConfig)


class CnnEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], cfg: CnnEncoderConfig):
        super().__init__()

        self.input_shape = (
            int(torch.tensor(obs_shape[:-2]).prod().item()),
            *obs_shape[-2:],
        )

        assert len(cfg.cnn_filter_dims) == len(cfg.cnn_kernel_strides), (
            "cnn_filter_dims and cnn_strides must have the same length"
        )

        encoder_layers = []
        in_channels = self.input_shape[0]
        for out_channels, stride in zip(cfg.cnn_filter_dims, cfg.cnn_kernel_strides):
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride))
            encoder_layers.append(nn.GELU())
            in_channels = out_channels
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder.apply(weight_init)

        self.output_dim = get_out_shape(self.input_shape, self.encoder)[0]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.reshape(obs, (-1, *self.input_shape))
        return self.encoder(obs)


class CnnActor(BaseActor):
    def __init__(
        self, encoder: CnnEncoder, obs_shape: Tuple[int, ...], action_dim: int, max_action: float, cfg: CnnActorConfig
    ):
        super().__init__(obs_shape, action_dim, max_action)
        self.encoder = encoder

        self.linear_trunk_layers = nn.Sequential(
            nn.Linear(encoder.output_dim, cfg.linear_trunk_dim),
            nn.LayerNorm(cfg.linear_trunk_dim),
            nn.Tanh(),
        )

        self.linear_trunk_layers.apply(weight_init)

        self.actor = MlpActor(cfg.linear_trunk_dim, action_dim, cfg.hidden_dim, cfg.n_layers, max_action)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_feat = self.encoder(obs)
        obs_feat = self.linear_trunk_layers(obs_feat)
        action = self.actor(obs_feat)
        return action


class CnnCritic(BaseCritic):
    def __init__(self, encoder: CnnEncoder, obs_shape: Tuple[int, ...], action_dim: int, cfg: CnnCriticConfig):
        super().__init__(obs_shape, action_dim)
        self.encoder = encoder

        self.linear_trunk_layers = nn.Sequential(
            nn.Linear(encoder.output_dim, cfg.linear_trunk_dim),
            nn.LayerNorm(cfg.linear_trunk_dim),
            nn.Tanh(),
        )

        self.linear_trunk_layers.apply(weight_init)

        self.critic = MlpCritic(cfg.linear_trunk_dim, action_dim, cfg.hidden_dim, cfg.n_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs_feat = self.encoder(obs)
        obs_feat = self.linear_trunk_layers(obs_feat)

        return self.critic(obs_feat, action)

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs_feat = self.encoder(obs)
        obs_feat = self.linear_trunk_layers(obs_feat)

        return self.critic.q1(obs_feat, action)


if __name__ == "__main__":
    # Example usage
    obs_shape = (3, 256, 256)
    action_dim = 4
    max_action = 1.0

    batch_size = 32

    # Create actor and critic
    cfg = CnnPolicyConfig()
    encoder = CnnEncoder(obs_shape, cfg.cnn_encoder_cfg)
    actor = CnnActor(encoder, obs_shape, action_dim, max_action, cfg.actor_cfg)
    critic = CnnCritic(encoder, obs_shape, action_dim, cfg.critic_cfg)

    obs = torch.randn(batch_size, *obs_shape)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, action)
    action = actor(obs)

    print("Action:", action.shape)
    print("Q1:", q1.shape)
    print("Q2:", q2.shape)
