import torch
import torch.nn as nn
from typing import Tuple


def get_out_shape(in_shape: Tuple[int, ...], module: nn.Module) -> Tuple[int, ...]:
    x = torch.randn((1, *in_shape), device=next(module.parameters()).device)
    return module(x)[0].shape


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


def identify_obs_shape(obs_shape: Tuple[int, ...], frame_stack: int) -> Tuple:
    H, W = obs_shape[-2:]
    nd = len(obs_shape)

    if nd == 3:  # (C, H, W)
        C, _, _ = obs_shape
        assert frame_stack == 1
        num_channels, region_num, eff_channels = 3, C // 3, C
    elif nd == 4:
        N, C, _, _ = obs_shape
        if frame_stack > 1:  # (N, C, H, W)
            assert frame_stack == N
            num_channels, region_num, eff_channels = C, 1, N * C
        else:  # (R, C, H, W)
            num_channels, region_num, eff_channels = C, N, N * C
    elif nd == 5:  # (N, R, C, H, W)
        N, R, C, _, _ = obs_shape
        assert frame_stack == N
        num_channels, region_num, eff_channels = C, R, N * R * C
    else:
        raise ValueError(f"Unsupported obs_shape {obs_shape}")

    obs_shape = (eff_channels, H, W)

    return obs_shape, num_channels, region_num
