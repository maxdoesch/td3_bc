import numpy as np
from PIL import Image
from typing import Tuple


def resize_stacked_images(
    stacked_image: np.ndarray,
    shape: Tuple[int, int],
    is_channels_first: bool,
) -> np.ndarray:
    """
    Nearest-neighbor resize for stacked RGB images.

    Supports:
      - 3D, channels-first: (3*k, H, W)
      - 3D, channels-last : (H, W, 3*k)
      - 4D, channels-first: (N, 3*k, H, W)
      - 4D, channels-last : (N, H, W, 3*k)

    Returns the same layout with spatial dims resized to (new_H, new_W).
    """
    if stacked_image.ndim not in (3, 4):
        raise ValueError("Expected 3D or 4D input array")
    new_h, new_w = shape
    if new_h <= 0 or new_w <= 0:
        raise ValueError("Target shape must be positive")

    # Identify axes for H and W based on layout
    if stacked_image.ndim == 3:
        H_axis, W_axis = (1, 2) if is_channels_first else (0, 1)
        C_axis = 0 if is_channels_first else 2
        C = stacked_image.shape[C_axis]
    else:  # 4D
        H_axis, W_axis = (2, 3) if is_channels_first else (1, 2)
        C_axis = 1 if is_channels_first else 3
        C = stacked_image.shape[C_axis]

    if C % 3 != 0:
        raise ValueError(f"Channel dimension must be a multiple of 3 (got {C}).")

    # Original sizes
    H = stacked_image.shape[H_axis]
    W = stacked_image.shape[W_axis]

    # Nearest-neighbor index maps (vectorized, no Python loops)
    y_idx = np.clip((np.arange(new_h) * H / new_h).astype(int), 0, H - 1)
    x_idx = np.clip((np.arange(new_w) * W / new_w).astype(int), 0, W - 1)

    # Resize along H and W using np.take to preserve layout
    out = np.take(stacked_image, y_idx, axis=H_axis)
    out = np.take(out, x_idx, axis=W_axis)

    return out


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSV format (H, W, 3)"""
    rgb = rgb.astype(np.float32) / 255.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc + 1e-10  # avoid division by zero
    s = np.where(maxc == 0, 0, delta / maxc)

    h = np.zeros_like(maxc)

    mask_r = r == maxc
    mask_g = g == maxc
    mask_b = b == maxc

    h[mask_r] = ((b - g) / delta)[mask_r]
    h[mask_g] = (2.0 + (r - b) / delta)[mask_g]
    h[mask_b] = (4.0 + (g - r) / delta)[mask_b]

    h = (h / 6.0) % 1.0  # normalize to [0, 1]
    return np.stack([h, s, v], axis=-1)


def replace_green_bg(x, bg: np.ndarray) -> np.ndarray:
    assert x.ndim == 3 and bg.ndim == 3, "Input images must be 3-dimensional"
    assert x.dtype == np.uint8 and bg.dtype == np.uint8

    channels_first = False if x.shape[-1] == 3 else True

    x_rgb = x if x.shape[-1] == 3 else np.moveaxis(x, 0, -1)  # ensure H W C Shape
    bg_rgb = bg if bg.shape[-1] == 3 else np.moveaxis(bg, 0, -1)

    hsv = rgb_to_hsv_np(x_rgb)

    h, s, v = hsv[..., 0] * 360, hsv[..., 1] * 255, hsv[..., 2] * 255

    green_mask = (100 <= h) & (h <= 185) & (80 <= s) & (s <= 255) & (70 <= v) & (v <= 255)

    out = x_rgb.copy()
    out[green_mask] = bg_rgb[green_mask]

    return np.moveaxis(out, -1, 0) if channels_first else out


def interpolate_bg(bg: np.ndarray, size: tuple) -> np.ndarray:
    is_channels_first = bg.shape[1] == 3  # assume RGB
    resized_frames = []

    for frame in bg:
        frame_img = np.moveaxis(frame, 0, -1) if is_channels_first else frame  # to HWC
        img = Image.fromarray(frame_img.astype(np.uint8))
        img_resized = img.resize((size[1], size[0]), resample=Image.BILINEAR)
        frame_resized = np.array(img_resized)
        if is_channels_first:
            frame_resized = np.moveaxis(frame_resized, -1, 0)  # back to CHW
        resized_frames.append(frame_resized)

    return np.stack(resized_frames, axis=0)
