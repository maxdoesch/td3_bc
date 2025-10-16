import math
import random
import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms import functional, RandomCrop
from typing import Optional, Tuple



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

class ColorJitterDual:
    """
    Apply identical ColorJitter to obs and next_obs.
    Shapes: (B, F, R, C, H, W)  with values in [0,1].
    """
    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.generator = generator

    @torch.no_grad()
    def __call__(
        self,
        img
    ):
        """
        obs/next_obs: (B, F, R, C, H, W) in [0,1], float tensor
        Applies jitter to both obs and next_obs.
        Order: brightness -> contrast -> saturation.
        """
        if isinstance(img, (tuple, list)):
            img, img2 = img[0], img[1]
        else:
            img2 = None 

        assert img.shape == img2.shape
        assert img.dtype == img2.dtype and img.is_floating_point()
        B, F, R, C, H, W = img.shape
        device = img.device

        # sample factor for each of B, F
        def _rand_factors(delta):
            if delta <= 0:
                return torch.ones((B, F), device=device)
            lo, hi = 1.0 - delta, 1.0 + delta
            return torch.empty((B, F), device=device).uniform_(lo, hi, generator=self.generator)

        bfac = _rand_factors(self.brightness)   # (B,F)
        cfac = _rand_factors(self.contrast)      # (B,F)
        sfac = _rand_factors(self.saturation)   # (B,F)

        # Broadcast helpers to (1,B,F,1,1,1,1)
        def _bcast(v):
            return v.view(1, B, F, 1, 1, 1, 1)

        bfac_b = _bcast(bfac)
        cfac_b = _bcast(cfac)
        sfac_b = _bcast(sfac)

        # Stack so we apply once to both (2,B,F,R,C,H,W)
        x = torch.stack([img, img2], dim=0)

        # --- Brightness: x <- x * b ---
        x = x * bfac_b

        # --- Contrast: x <- (x - mean)*c + mean ---
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)   # per-sample mean over (C,H,W)
        x = (x - mean) * cfac_b + mean

        # --- Saturation: x <- (x - gray)*s + gray ---
        # Luma weights for RGB to gray; assumes C==3
        if C == 3:
            w = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1,1,1,1,3,1,1)
            gray = (x * w).sum(dim=4, keepdim=True)   # (...,1,H,W)
            x = (x - gray) * sfac_b + gray

        # Split back
        out1, out2 = x.unbind(dim=0)
        return out1.clamp_(0, 1), out2.clamp_(0, 1)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"generator={self.generator})"
        )
    
class RandomErasingDual:
    """
    Apply identical RandomErasing to obs and next_obs.
    Shapes: (B, F, R, C, H, W)  with values in [0,1].
    """
    def __init__(
        self,
        p: float = 0.4,
        scale: Tuple[float, float] = (0.02, 0.15),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.generator = generator

    @staticmethod
    def get_params(
        img: torch.Tensor, scale: tuple[float, float], ratio: tuple[float, float], value: Optional[list[float]] = None
    ) -> tuple[int, int, int, int, torch.Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v
            

    @torch.no_grad()
    def __call__(self, img):
        """
        obs/next_obs: (B, F, R, C, H, W) in [0,1]
        Returns augmented (obs_aug, next_obs_aug) with identical params per (b,f,r).
        """
        if isinstance(img, (tuple, list)):
            img, img2 = img[0], img[1]
        else:
            img2 = None

        assert img.shape == img2.shape
        assert img.dtype == img2.dtype and img.is_floating_point()
        B, F, R, C, H, W = img.shape

        out1 = img.clone()
        out2 = img2.clone()

        area = H * W

        for b in range(B):
            for f in range(F):
                for r in range(R):
                    if torch.rand(1, generator=self.generator) > self.p:
                        continue

                    # sample erasing rectangle
                    log_ratio = torch.log(torch.tensor(self.ratio))
                    for _ in range(10):  # try a few times to find a valid rectangle
                        erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1], generator=self.generator).item()
                        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1], generator=self.generator)).item()
                        h = int(round(math.sqrt(erase_area * aspect_ratio)))
                        w = int(round(math.sqrt(erase_area / aspect_ratio)))
                        if not (h < H and w < W):
                            continue

                        i = torch.randint(0, H - h + 1, size=(1,)).item()
                        j = torch.randint(0, W - w + 1, size=(1,)).item()

                    # apply same to both
                    out1[b, f, r, :, i:i+h, j:j+w] = self.value
                    out2[b, f, r, :, i:i+h, j:j+w] = self.value

        return out1, out2
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}, scale={self.scale}, ratio={self.ratio}, value={self.value}, generator={self.generator})"
        )