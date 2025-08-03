from collections import deque
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import gymnasium as gym

from .segmentation import MobileSAMV2, MobileSAMV2Config
from .utils import resize_stacked_images


@dataclass
class FTDObservationWrapperConfig:
    # MobileSAMv2 parameters
    sam_config: MobileSAMV2Config = field(default_factory=MobileSAMV2Config)

    # FTD parameters
    num_regions: int = 10  # Number of segmented regions
    num_channels: int = 3  # Number of input channels
    add_original_frame: bool = True  # Whether to add the original frame as the last channel

    # Mask sorting
    sort_by: str = "area"  # Mode for sorting masks (e.g., 'area', 'score')


class FTDObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, config: FTDObservationWrapperConfig = FTDObservationWrapperConfig()):
        super().__init__(env)

        # Set up configurations
        assert isinstance(config, FTDObservationWrapperConfig), "config must be an instance of FTDWrapperConfig"
        assert config.num_channels in [1, 3], "num_channels must be either 1 (grayscale) or 3 (RGB)"
        assert config.sort_by in ["area", "score"], "sort_mode must be either 'area' or 'score'"
        self.config = config

        # Load MobileSAMv2
        self.mobilesamv2 = MobileSAMV2(config.sam_config)
        self.mobilesamv2.enable_logging(False)

        # Set new observation space
        old_shape = env.observation_space.shape
        assert len(old_shape) == 3, "Observation space must be a 3D space (C, H, W)"
        assert old_shape[0] == config.num_channels, (
            f"Expected {config.num_channels} channels in observation space, got {old_shape[0]}"
        )
        self.H, self.W = old_shape[1:]
        self.num_regions_with_original = config.num_regions + int(config.add_original_frame)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((config.num_channels * self.num_regions_with_original, self.H, self.W)),  # (C * R, H, W)
            dtype=np.uint8,
        )

    def __get_predictions(self, observation: np.ndarray) -> dict[str, torch.Tensor]:
        """
        Generate masks for the input observation tensor using MobileSAMv2.
        """
        # If the image is grayscale, convert it to RGB by repeating the channels
        if observation.shape[0] == 1:
            observation = np.repeat(observation, 3, axis=0)

        observation = np.transpose(observation, [1, 2, 0])
        pred = self.mobilesamv2.get_prediction(observation)

        return pred

    def __sort_predictions(self, pred: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the sorted indices of the masks based on the specified sorting mode (either area or score).
        """
        masks, scores = pred["masks"], pred["scores"]  # (N, H, W), (N,)

        if self.config.sort_by == "area":
            areas = torch.sum(masks, dim=(1, 2))
            sorted_indices = torch.argsort(areas, descending=True)
        else:
            sorted_indices = torch.argsort(scores, descending=True)

        return sorted_indices

    def __pad_or_trim_masks(self, masks: torch.Tensor):
        """
        Pad masks with black regions or trim excess masks to ensure the number of masks matches num_regions.
        """
        if masks.shape[0] < self.config.num_regions:
            pad = torch.zeros(
                (self.config.num_regions - masks.shape[0], self.H, self.W), device=masks.device, dtype=masks.dtype
            )
            masks = torch.cat([masks, pad], dim=0)
        else:
            masks = masks[: self.config.num_regions]

        return masks

    def observation(self, observation):
        # Get Masks
        pred = self.__get_predictions(observation)
        if pred is None:
            return np.zeros((self.config.num_channels * self.num_regions_with_original, self.H, self.W), dtype=np.uint)
        sorted_indices = self.__sort_predictions(pred)
        masks = pred["masks"][sorted_indices]
        masks = self.__pad_or_trim_masks(masks)

        full_frame = torch.from_numpy(observation).to(masks.device).unsqueeze(0)  # (1, C, H, W)
        masked_segments = full_frame.masked_fill(~masks.bool().unsqueeze(1), 0)

        # Append original frame last and reshape
        if self.config.add_original_frame:
            all_segments = torch.cat([masked_segments, full_frame], dim=0)  # (R+1, C, H, W)
        else:
            all_segments = masked_segments

        all_segments = all_segments.reshape(self.num_regions_with_original * self.config.num_channels, self.H, self.W)

        return all_segments.byte().cpu().numpy()


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=np.uint8):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3 : (i + 1) * 3]


class ResizeObservation(gym.ObservationWrapper):
    """Resize the observation to a given shape."""

    def __init__(self, env, shape: Tuple, is_channels_first: bool = True):
        super().__init__(env)
        self.shape = shape
        self.is_channels_first = is_channels_first

        if self.is_channels_first:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(env.observation_space.shape[0], shape[0], shape[1]), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(shape[0], shape[1], env.observation_space.shape[-1]), dtype=np.uint8
            )

    def observation(self, observation):
        observation = resize_stacked_images(observation, self.shape)
        return observation


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=((shp[0] * k,) + shp[1:]), dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return LazyFrames(list(self._frames))
