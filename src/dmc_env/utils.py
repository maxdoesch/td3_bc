import numpy as np
from PIL import Image
from typing import Tuple

def resize_stacked_images(stacked_image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if stacked_image.ndim != 3:
        raise ValueError("Expected 3D input array")

    new_height, new_width = shape

    if stacked_image.shape[2] % 3 == 0:
        # Likely HWC
        H, W, C = stacked_image.shape
        num_imgs = C // 3
        scale_y = new_height / H
        scale_x = new_width / W

        out = np.zeros((new_height, new_width, C), dtype=stacked_image.dtype)

        for i in range(num_imgs):
            img = stacked_image[:, :, i * 3 : (i + 1) * 3]
            y_idx = np.clip((np.arange(new_height) / scale_y).astype(int), 0, H - 1)
            x_idx = np.clip((np.arange(new_width) / scale_x).astype(int), 0, W - 1)
            out[:, :, i * 3 : (i + 1) * 3] = img[y_idx[:, None], x_idx[None, :]]
        return out

    elif stacked_image.shape[0] % 3 == 0:
        # If first dimension is divisible by 3, assume CHW stacked RGB images
        C, H, W = stacked_image.shape
        num_imgs = C // 3
        scale_y = new_height / H
        scale_x = new_width / W

        out = np.zeros((C, new_height, new_width), dtype=stacked_image.dtype)

        for i in range(num_imgs):
            img = stacked_image[i * 3 : (i + 1) * 3, :, :]
            y_idx = np.clip((np.arange(new_height) / scale_y).astype(int), 0, H - 1)
            x_idx = np.clip((np.arange(new_width) / scale_x).astype(int), 0, W - 1)
            out[i * 3 : (i + 1) * 3, :, :] = img[:, y_idx[:, None], x_idx[None, :]]
        return out

    else:
        raise ValueError("Input shape doesn't match expected CHW or HWC stacked RGB format.")


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
