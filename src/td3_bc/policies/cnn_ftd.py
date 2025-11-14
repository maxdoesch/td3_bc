import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .policy import PolicyConfig
from .cnn import CnnEncoderConfig, CnnPolicyConfig
from .ftd import ImageAttentionSelectorLayers, ImageAttentionSelectorConfig
from .utils import get_out_shape, weight_init, identify_obs_shape


@dataclass
class CnnFtdEncoderConfig(CnnEncoderConfig):
    attn_selector_cfg: Optional[ImageAttentionSelectorConfig] = field(default_factory=ImageAttentionSelectorConfig)


@PolicyConfig.register_subclass("cnn_ftd")
@dataclass
class CnnFtdPolicyConfig(CnnPolicyConfig):
    cnn_encoder_cfg: CnnFtdEncoderConfig = field(default_factory=CnnFtdEncoderConfig)


class CnnFtdEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], frame_stack: int, cfg: CnnFtdEncoderConfig):
        super().__init__()

        self.frame_stack = frame_stack
        self.input_shape, self.num_channels, region_num = identify_obs_shape(obs_shape, frame_stack)

        self.image_attention_selector = ImageAttentionSelectorLayers(
            input_shape=self.input_shape,
            region_num=region_num,
            in_channels=self.num_channels,
            frame_stack=frame_stack,
            cfg=cfg.attn_selector_cfg,
        )

        assert len(cfg.cnn_filter_dims) == len(cfg.cnn_kernel_strides), (
            "cnn_filter_dims and cnn_strides must have the same length"
        )

        encoder_layers = []
        in_channels = self.num_channels * frame_stack
        for out_channels, stride in zip(cfg.cnn_filter_dims, cfg.cnn_kernel_strides):
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride))
            encoder_layers.append(nn.GELU())
            in_channels = out_channels
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder.apply(weight_init)

        self.output_dim = get_out_shape((self.num_channels * frame_stack, *self.input_shape[-2:]), self.encoder)[0]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.reshape(obs, (-1, *self.input_shape))
        obs = self.image_attention_selector(obs)
        return self.encoder(obs)

    def select_image(self, obs: torch.Tensor):
        with torch.no_grad():
            obs = self.image_attention_selector(obs.unsqueeze(0))  # Add batch dimension
            obs = obs.squeeze(0)  # Remove batch dimension

            # obs shape: (F*C, H, W)
            F, C, H, W = self.frame_stack, self.num_channels, *obs.shape[-2:]

            obs = obs.view(F, C, H, W)
            # Convert to (H, F*W, C)
            obs = obs.permute(2, 0, 3, 1).reshape(H, F * W, C)

            obs = (255 * obs).to(torch.uint8)

        return obs.cpu().numpy()
