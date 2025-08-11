import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Union

from .policy import PolicyConfig, BaseActor, BaseCritic
from .mlp import MlpActor, MlpCritic


@PolicyConfig.register_subclass("ftd")
@dataclass
class FtdPolicyConfig(PolicyConfig):
    num_regions: int = 10  # Maximum number of segmented regions
    num_channels: int = 3  # Number of input channels
    num_stack: int = 1  # Number of frames stacked together as a single observation
    num_selector_layers: int = 5  # Number of convolutional layers in the attention selector
    num_filters: int = 32  # Number of filters in the convolutional layers
    embed_dim: int = 128  # Dimension of the embedding space for attention
    num_attention_heads: int = 4  # Number of attention heads
    num_shared_layers: int = 11  # Number of shared convolutional layers
    num_head_layers: int = 0  # Number of hidden layers in the head CNN
    projection_dim: int = (
        100  # Dimension of the projection space for actor and critic; must match actor and critic input dim
    )
    skip_attention_selector: bool = False  # skip attention selector layers (Debugging feature)


def _get_out_shape(in_shape, layers, device="cpu"):
    x = torch.randn(*in_shape).to(device).unsqueeze(0)
    return layers(x).squeeze(0).shape


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.0


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(nn.Linear(in_shape[0], out_dim), nn.LayerNorm(out_dim), nn.Tanh())
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class SelectorCNN(nn.Module):
    def __init__(
        self, selector_layers, obs_shape, region_num=5, in_channels=3, stack_num=3, num_shared_layers=11, num_filters=32
    ):
        super().__init__()
        assert len(obs_shape) == 3
        # assert region_num * in_channels * stack_num == obs_shape[0]
        self.obs_shape = obs_shape
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.num_filters = num_filters

        self.selector_layers = selector_layers

        self.shared_layers = [nn.Conv2d(self.stack_num * self.in_channels, num_filters, 3, stride=2)]
        for _ in range(1, num_shared_layers):
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.shared_layers = nn.Sequential(*self.shared_layers)

        self.out_shape = _get_out_shape(
            [self.stack_num * self.in_channels, self.obs_shape[-2], self.obs_shape[-1]], self.shared_layers
        )
        self.shared_layers.apply(weight_init)

    def forward(self, x):
        x = self.selector_layers(x)
        x = self.shared_layers(x)

        return x


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class ImageAttentionSelectorLayers(nn.Module):
    def __init__(self, obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads):
        super().__init__()

        # self.preprocess_layer = nn.Sequential(*[NormalizeImg()]) #@maxdoesch normalization is taken care of in trainer.py via buffer
        self.layers = [nn.Conv2d(in_channels, num_filters, 3, stride=2, padding=1)]
        self.shape = obs_shape[1:]
        self.region_num = region_num
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.current_image_size = obs_shape[1] // 2
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.current_image_size = self.current_image_size // 2
        self.layers.append(Flatten())
        out_num = num_filters * self.current_image_size**2
        self.layers.append(nn.Linear(out_num, embed_dim))
        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(weight_init)

        self.q = nn.Linear(embed_dim, num_heads * embed_dim)
        self.k = nn.Linear(embed_dim, num_heads * embed_dim)
        # no 'v' network, we use the raw input images as the values

    def forward(self, x, return_logits=False, return_head_logits=False, return_all=False):
        # (batch_size, stack_num * (region_num + 1) * channels , height, width)
        # Last region is the whole frame
        S, R, C, H, W = self.stack_num, self.region_num + 1, self.in_channels, self.shape[0], self.shape[1]
        x = x.reshape(-1, C, H, W)
        B = x.shape[0] // S // R
        # x = self.preprocess_layer(x)

        mask = torch.sum(x, dim=(1, 2, 3)).reshape(B * S, 1, -1)[:, :, :-1]
        mask = torch.where(mask != 0, False, True)

        tokens = self.layers(x).reshape(B * S, R, -1)
        tokens_frame = tokens[:, -1:, :]
        tokens_segment = tokens[:, :-1, :]
        q = self.q(tokens_frame).reshape(B * S, 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        k = self.k(tokens_segment).reshape(B * S, R - 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        v = x.reshape(B * S, R, C * H * W)[:, :-1, :]

        attention = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
        mask = torch.cat([torch.unsqueeze(mask, dim=1)] * self.num_heads, dim=1)
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
    def __init__(self, obs_shape: tuple[int, int, int], cfg: FtdPolicyConfig):
        super().__init__()
        self.cfg = cfg

        self.selector_layers = (
            ImageAttentionSelectorLayers(
                obs_shape,
                cfg.num_regions,
                cfg.num_channels,
                cfg.num_stack,
                cfg.num_selector_layers,
                cfg.num_filters,
                cfg.embed_dim,
                cfg.num_attention_heads,
            )
            if not cfg.skip_attention_selector
            else nn.Identity()
        )

        self.selector_cnn = SelectorCNN(
            self.selector_layers,
            obs_shape,
            cfg.num_regions,
            cfg.num_channels,
            cfg.num_stack,
            cfg.num_shared_layers,
            cfg.num_filters,
        )

        self.head_cnn = HeadCNN(self.selector_cnn.out_shape, cfg.num_head_layers, cfg.num_filters)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.selector_cnn(obs)
        obs = self.head_cnn(obs)
        return obs


class FTDActor(BaseActor):
    def __init__(
        self, shared_layers: SharedFTDLayers, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, max_action: float
    ):
        super().__init__(obs_shape, action_dim, max_action)
        self.shared_layers = shared_layers

        projection = RLProjection(shared_layers.head_cnn.out_shape, shared_layers.cfg.projection_dim)

        self.encoder = Encoder(shared_layers.selector_cnn, shared_layers.head_cnn, projection)

        self.actor = MlpActor(self.encoder.out_dim, action_dim, hidden_dim=256, n_layers=2, max_action=max_action)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proj = self.encoder(obs)
        action = self.actor(proj)
        return action


class FTDCritic(BaseCritic):
    def __init__(self, shared_layers: SharedFTDLayers, obs_shape: Union[int, Tuple[int, ...]], action_dim: int):
        super().__init__(obs_shape, action_dim)
        self.shared_layers = shared_layers

        projection = RLProjection(shared_layers.head_cnn.out_shape, shared_layers.cfg.projection_dim)

        self.encoder = Encoder(shared_layers.selector_cnn, shared_layers.head_cnn, projection)

        self.critic = MlpCritic(self.encoder.out_dim, action_dim, hidden_dim=256, n_layers=2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        proj = self.encoder(obs)
        q1, q2 = self.critic(proj, action)
        return q1, q2

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        proj = self.encoder(obs)
        return self.critic.q1(proj, action)

    def q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        proj = self.encoder(obs)
        return self.critic.q2(proj, action)


if __name__ == "__main__":
    # Example usage
    obs_shape = (33, 256, 256)
    action_dim = 4
    hidden_dim = 64
    n_layers = 2
    max_action = 1.0

    batch_size = 32

    # Create actor and critic
    config = FtdPolicyConfig()
    encoder = SharedFTDLayers(obs_shape, config)
    actor = FTDActor(encoder, obs_shape, action_dim, max_action)
    critic = FTDCritic(encoder, obs_shape, action_dim)

    obs = torch.randn(batch_size, *obs_shape)
    action = torch.randn(batch_size, action_dim)

    q1, q2 = critic(obs, action)
    action = actor(obs)

    print("Action:", action.shape)
    print("Q1:", q1.shape)
    print("Q2:", q2.shape)
