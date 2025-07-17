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

def combine_stacked_frames(observation: np.ndarray) -> np.ndarray:
    if len(observation.shape) == 3:
        C, H, W = observation.shape
        observation = observation.reshape(C // 3, 3, H, W)
        observation = observation.transpose(2, 0, 3, 1)  # (H, C//3, W, 3)
        observation = observation.reshape(H, W * (C // 3), 3)     # (H, W * (C // 3), 3)
    elif len(observation.shape) == 4:
        F, C, H, W = observation.shape
        observation = observation.reshape(F, C // 3, 3, H, W)
        observation = observation.transpose(0, 3, 1, 4, 2)  # (N, H, C//3, W, 3)
        observation = observation.reshape(F, H, W * (C // 3), 3)     # (N, H, W * (C // 3), 3)

    return observation