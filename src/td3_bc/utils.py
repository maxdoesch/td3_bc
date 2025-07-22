from typing import Tuple
import gymnasium as gym
import numpy as np


def is_image_space(space: gym.Space) -> bool:
    return (
        isinstance(space, gym.spaces.Box)
        and len(space.shape) in {2, 3}
        and space.shape[0] >= 32
        and space.shape[1] >= 32
        and space.dtype == np.uint8
        and bool(np.all(space.low == 0))
        and bool(np.all(space.high == 255))
    )


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


def combine_stacked_frames(observation: np.ndarray) -> np.ndarray:
    if len(observation.shape) == 3:
        C, H, W = observation.shape
        observation = observation.reshape(C // 3, 3, H, W)
        observation = observation.transpose(2, 0, 3, 1)  # (H, C//3, W, 3)
        observation = observation.reshape(H, W * (C // 3), 3)  # (H, W * (C // 3), 3)
    elif len(observation.shape) == 4:
        F, C, H, W = observation.shape
        observation = observation.reshape(F, C // 3, 3, H, W)
        observation = observation.transpose(0, 3, 1, 4, 2)  # (N, H, C//3, W, 3)
        observation = observation.reshape(F, H, W * (C // 3), 3)  # (N, H, W * (C // 3), 3)

    return observation


def uncombine_stacked_frames(observation: np.ndarray) -> np.ndarray:
    if len(observation.shape) == 3:
        H, W, C = observation.shape
        num_channels = W // H

        observation = observation.reshape(H, num_channels, W // num_channels, 3)
        observation = observation.transpose(1, 3, 0, 2)  # (num_channels, 3, H, W // num_channels)
        observation = observation.reshape(num_channels * 3, H, W // num_channels)
    elif len(observation.shape) == 4:
        N, H, W, C = observation.shape
        num_channels = W // H

        observation = observation.reshape(N, H, num_channels, W // num_channels, 3)
        observation = observation.transpose(0, 2, 4, 1, 3)  # (N, num_channels, 3, H, W // num_channels)
        observation = observation.reshape(N, num_channels * 3, H, W // num_channels)

    return observation
