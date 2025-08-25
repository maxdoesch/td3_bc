import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, Union, Optional

from .policy import PolicyConfig, BaseActor, BaseCritic
from .mlp import MlpActor, MlpCritic
from .utils import get_out_shape, weight_init


@dataclass
class ImageAttentionSelectorConfig:
    conv_layers: int = 5  # Number of convolutional layers
    conv_filters: int = 32  # Number of filters in conv layers
    attention_embed_dim: int = 128  # Embedding dimension for attention
    attention_heads: int = 4  # Number of attention heads


@dataclass
class FeatureExtractorConfig:
    conv_layers: int = 11
    conv_filters: int = 32


@dataclass
class SharedFTDLayersConfig:
    attn_selector_cfg: Optional[ImageAttentionSelectorConfig] = field(default_factory=ImageAttentionSelectorConfig)
    feature_extractor_cfg: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)


@dataclass
class FTDActorConfig:
    rl_projection_dim: int = 100

    hidden_dim: int = 256
    n_layers: int = 2


@dataclass
class FTDCriticConfig:
    rl_projection_dim: int = 100

    hidden_dim: int = 256
    n_layers: int = 2


@PolicyConfig.register_subclass("ftd")
@dataclass
class FtdPolicyConfig(PolicyConfig):
    shared_layers_cfg: SharedFTDLayersConfig = field(default_factory=SharedFTDLayersConfig)

    actor_cfg: FTDActorConfig = field(default_factory=FTDActorConfig)
    critic_cfg: FTDCriticConfig = field(default_factory=FTDCriticConfig)


class RLProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.projection = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh())
        self.out_dim = out_dim

        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class FeatureExtractorCNN(nn.Module):
    def __init__(self, in_channels: int, frame_stack: int, cfg: FeatureExtractorConfig):
        super().__init__()

        self.in_channels = in_channels
        self.frame_stack = frame_stack

        self.conv_layers = cfg.conv_layers
        self.conv_filters = cfg.conv_filters

        self.feature_extractor = [
            nn.Conv2d(
                in_channels=self.frame_stack * self.in_channels, out_channels=self.conv_filters, kernel_size=3, stride=2
            )
        ]
        for _ in range(1, self.conv_layers):
            self.feature_extractor.append(nn.ReLU())
            self.feature_extractor.append(
                nn.Conv2d(in_channels=self.conv_filters, out_channels=self.conv_filters, kernel_size=3, stride=1)
            )
        self.feature_extractor.append(nn.Flatten())
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        self.feature_extractor.apply(weight_init)

    def forward(self, x):
        x = self.feature_extractor(x)

        return x


class ImageAttentionSelectorLayers(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        region_num: int,
        in_channels: int,
        frame_stack: int,
        cfg: ImageAttentionSelectorConfig,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.region_num = region_num
        self.in_channels = in_channels
        self.frame_stack = frame_stack

        self.conv_layers = cfg.conv_layers
        self.conv_filters = cfg.conv_filters
        self.attention_embed_dim = cfg.attention_embed_dim
        self.attention_heads = cfg.attention_heads

        reduced_img_height = input_shape[1] // 2
        self.layers = [
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_filters, kernel_size=3, stride=2, padding=1)
        ]
        for _ in range(1, self.conv_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(
                nn.Conv2d(
                    in_channels=self.conv_filters, out_channels=self.conv_filters, kernel_size=3, stride=1, padding=1
                )
            )
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            reduced_img_height = reduced_img_height // 2
        self.layers.append(nn.Flatten())

        out_num = self.conv_filters * reduced_img_height**2
        self.layers.append(nn.Linear(out_num, self.attention_embed_dim))

        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(weight_init)

        self.q = nn.Linear(self.attention_embed_dim, self.attention_heads * self.attention_embed_dim)
        self.k = nn.Linear(self.attention_embed_dim, self.attention_heads * self.attention_embed_dim)
        # no 'v' network, we use the raw input images as the values

    def forward(self, x, return_logits=False, return_head_logits=False, return_all=False):
        # (batch_size, frame_stack * region_num * channels , height, width)
        # Last region is the whole frame
        B, _, H, W = x.shape
        S, R, C = self.frame_stack, self.region_num, self.in_channels
        x = x.reshape(-1, C, H, W)

        mask = torch.sum(x, dim=(1, 2, 3)).reshape(B * S, 1, -1)[:, :, :-1]
        mask = torch.where(mask != 0, False, True)

        tokens = self.layers(x).reshape(B * S, R, -1)
        tokens_frame = tokens[:, -1:, :]
        tokens_segment = tokens[:, :-1, :]
        q = self.q(tokens_frame).reshape(B * S, 1, self.attention_heads, self.attention_embed_dim).transpose(-3, -2)
        k = (
            self.k(tokens_segment)
            .reshape(B * S, R - 1, self.attention_heads, self.attention_embed_dim)
            .transpose(-3, -2)
        )
        v = x.reshape(B * S, R, C * H * W)[:, :-1, :]

        attention = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
        mask = torch.cat([torch.unsqueeze(mask, dim=1)] * self.attention_heads, dim=1)
        attention = attention.masked_fill_(mask, float("-inf"))

        multi_probs = torch.softmax(attention, dim=-1)
        probs = torch.mean(multi_probs, dim=1)
        ret_obs = torch.matmul(probs, v)

        # vector 2 image
        ret_obs = ret_obs.reshape(-1, S * C, H, W)

        if return_logits:
            return probs
        elif return_head_logits:
            return multi_probs
        elif return_all:
            return ret_obs, probs
        else:
            return ret_obs


class SharedFTDLayers(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], frame_stack: int, cfg: SharedFTDLayersConfig):
        super().__init__()
        self.cfg = cfg

        H, W = obs_shape[-2:]
        nd = len(obs_shape)

        if nd == 3:  # (C, H, W)
            C, _, _ = obs_shape
            self.num_channels, region_num, eff_channels = 3, C // 3, C
        elif nd == 4:
            N, C, _, _ = obs_shape
            if frame_stack > 1:  # (N, C, H, W)
                assert frame_stack == N
                self.num_channels, region_num, eff_channels = C, 1, N * C
            else:  # (R, C, H, W)
                self.num_channels, region_num, eff_channels = C, N, N * C
        elif nd == 5:  # (N, R, C, H, W)
            N, R, C, _, _ = obs_shape
            assert frame_stack == N
            self.num_channels, region_num, eff_channels = C, R, N * R * C
        else:
            raise ValueError(f"Unsupported obs_shape {obs_shape}")

        self.input_shape = (eff_channels, H, W)

        self.image_attention_selector = (
            ImageAttentionSelectorLayers(
                input_shape=self.input_shape,
                region_num=region_num,
                in_channels=self.num_channels,
                frame_stack=frame_stack,
                cfg=cfg.attn_selector_cfg,
            )
            if cfg.attn_selector_cfg
            else nn.Identity()
        )

        self.feature_extractor_cnn = FeatureExtractorCNN(
            in_channels=self.num_channels, frame_stack=frame_stack, cfg=cfg.feature_extractor_cfg
        )
        self.out_dim = get_out_shape(
            in_shape=self.input_shape,
            module=nn.Sequential(
                self.image_attention_selector,
                self.feature_extractor_cnn,
            ),
        )[0]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.reshape(obs, (-1, *self.input_shape))
        obs = self.image_attention_selector(obs)
        obs = self.feature_extractor_cnn(obs)
        return obs

    def select_image(self, obs: torch.Tensor):
        with torch.no_grad():
            obs = self.image_attention_selector(obs.unsqueeze(0))  # Add batch dimension
            obs = obs.squeeze()[-self.num_channels :]
            obs = obs.permute(1, 2, 0)  # Convert to (H, W, C) format
            obs = (255 * obs).to(torch.uint8)  # Convert to uint8 format

        return obs.cpu().numpy()


class Encoder(nn.Module):
    def __init__(self, shared_ftd_layers: SharedFTDLayers, projection: RLProjection):
        super().__init__()
        self.shared_ftd_layers = shared_ftd_layers
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x):
        x = self.shared_ftd_layers(x)

        return self.projection(x)


class FTDActor(BaseActor):
    def __init__(
        self,
        shared_ftd_layers: SharedFTDLayers,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        max_action: float,
        cfg: FTDActorConfig,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, max_action=max_action)

        rl_projection = RLProjection(
            in_dim=shared_ftd_layers.out_dim,
            out_dim=cfg.rl_projection_dim,
        )

        self.encoder = Encoder(shared_ftd_layers=shared_ftd_layers, projection=rl_projection)

        self.actor = MlpActor(
            obs_shape=cfg.rl_projection_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            max_action=max_action,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proj = self.encoder(obs)
        action = self.actor(proj)
        return action


class FTDCritic(BaseCritic):
    def __init__(
        self,
        shared_layers: SharedFTDLayers,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        cfg: FTDCriticConfig,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim)

        rl_projection = RLProjection(
            in_dim=shared_layers.out_dim,
            out_dim=cfg.rl_projection_dim,
        )

        self.encoder = Encoder(shared_ftd_layers=shared_layers, projection=rl_projection)

        self.critic = MlpCritic(
            obs_shape=cfg.rl_projection_dim, action_dim=action_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        proj = self.encoder(obs)
        q1, q2 = self.critic(proj, action)
        return q1, q2

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        proj = self.encoder(obs)
        return self.critic.q1(proj, action)


def main():
    # Example usage
    frame_stack = 4
    obs_shape = (frame_stack, 11, 3, 256, 256)
    action_dim = 4
    max_action = 1.0

    batch_size = 32

    # Create actor and critic
    config = FtdPolicyConfig()
    encoder = SharedFTDLayers(obs_shape, frame_stack, config.shared_layers_cfg)
    actor = FTDActor(encoder, obs_shape, action_dim, max_action, config.actor_cfg)
    critic = FTDCritic(encoder, obs_shape, action_dim, config.critic_cfg)

    obs = torch.randn(batch_size, *obs_shape)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, action)
    action = actor(obs)

    print("Action:", action.shape)
    print("Q1:", q1.shape)
    print("Q2:", q2.shape)


if __name__ == "__main__":
    main()
