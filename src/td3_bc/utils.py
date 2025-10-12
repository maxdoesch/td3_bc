import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms import functional, RandomCrop
from typing import Optional



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
    if len(observation.shape) == 4:
        S, C, H, W = observation.shape
        observation = observation.transpose(2, 0, 3, 1)  # (H, S, W, C)
        observation = observation.reshape(H, W * S, C)  # (H, W * S, C)
    elif len(observation.shape) == 5:
        N, S, C, H, W = observation.shape
        observation = observation.transpose(0, 3, 1, 4, 2)  # (N, H, S, W, C)
        observation = observation.reshape(N, H, W * S, C)  # (N, H, W * S, C)

    return observation


def uncombine_stacked_frames(observation: np.ndarray) -> np.ndarray:
    """
    Convert horizontally stacked RGB frames into channel-first format.

    Input:
        (H, W, C)  → Output: (num_regions, 3, H, W/num_regions)
        (N, H, W, C) → Output: (N, num_regions, 3, H, W/num_regions)

    Assumes W = num_regions * frame_width and C = 3.
    """
    if len(observation.shape) == 3:
        H, W, C = observation.shape
        num_regions = W // H

        observation = observation.reshape(H, num_regions, W // num_regions, 3)
        observation = observation.transpose(1, 3, 0, 2)  # (num_regions, 3, H, W // num_regions)
        observation = observation.reshape(num_regions, 3, H, W // num_regions)
    elif len(observation.shape) == 4:
        N, H, W, C = observation.shape
        num_regions = W // H

        observation = observation.reshape(N, H, num_regions, W // num_regions, 3)
        observation = observation.transpose(0, 2, 4, 1, 3)  # (N, num_regions, 3, H, W // num_regions)
        observation = observation.reshape(N, num_regions, 3, H, W // num_regions)

    observation = observation.squeeze()

    return observation


class RandomCropDual(RandomCrop):
    def _do_padding(self, img: torch.Tensor) -> torch.Tensor:
        if self.padding is not None:
            img = functional.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = functional.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = functional.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = functional.pad(img, padding, self.fill, self.padding_mode)

        return img

    def forward(self, img):
        """
        Args:
            img: Image(s) to be cropped.

        Returns:
            Tensor: Cropped image(s).
        """

        if isinstance(img, (tuple, list)):
            img, img2 = img[0], img[1]
        else:
            img2 = None
        
        img = self._do_padding(img)
        if img2 is not None:
            img2 = self._do_padding(img2)

        i, j, h, w = self.get_params(img, self.size)

        img = functional.crop(img, i, j, h, w)
        if img2 is not None:
            img2 = functional.crop(img2, i, j, h, w)
            return img, img2

        return img
    

class RandomPartialRPermutation:
    def __init__(self, generator: Optional[torch.Generator] = None):
        """
        Args:
            generator: Optional torch.Generator for reproducible shuffling.
        """
        self.generator = generator

    def _find_r_dim(self, x: torch.Tensor) -> int:
        if x.ndim == 6:
            # (B, F, R, C, H, W)
            r_dim = 2
        elif x.ndim == 5:
            # (B, R, C, H, W)
            r_dim = 1
        else:
            raise ValueError(
                f"Unsupported tensor shape {tuple(x.shape)}. "
                "Expected 4D (R, C, H, W) or 5D (F, R, C, H, W)."
            )
        return r_dim

    def __call__(self, img) -> torch.Tensor:
        if isinstance(img, (tuple, list)):
            img, img2 = img[0], img[1]
        else:
            img2 = None
        
        r_dim = self._find_r_dim(img)

        R = img.shape[r_dim]

        # Indices 0..R-2 shuffled, R-1 kept at the end
        perm_first = torch.randperm(R - 1, generator=self.generator, device=img.device)
        perm = torch.cat([perm_first, torch.tensor([R - 1], device=img.device)])

        # Index along the R dimension
        img = img.index_select(dim=r_dim, index=perm)
        if img2 is not None:
            img2 = img2.index_select(dim=r_dim, index=perm)
            return img, img2
        
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(generator={self.generator})"