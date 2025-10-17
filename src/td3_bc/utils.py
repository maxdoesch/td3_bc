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

    @torch.no_grad()
    def __call__(self, img):
        # Expect a tuple/list (img, img2) and apply exactly the same rectangles to both
        if isinstance(img, (tuple, list)):
            img, img2 = img[0], img[1]
        else:
            img2 = None

        assert img2 is not None, "Pass (img, img2) to apply identical erasing to both."
        assert img.shape == img2.shape, "img and img2 must have identical shapes"
        assert img.dtype == img2.dtype and img.is_floating_point(), "imgs must be floating point tensors"

        B, F, R, C, H, W = img.shape
        device = img.device
        N = B * F * R

        # Flatten leading dims to (N, C, H, W)
        x1 = img.reshape(N, C, H, W).clone()
        x2 = img2.reshape(N, C, H, W).clone()

        # Decide which samples get erased
        apply_mask = (torch.rand(N, generator=self.generator, device=device) < self.p)

        if apply_mask.any():
            # Sample scale and ratio
            s_low, s_high = self.scale
            r_low, r_high = self.ratio

            # target erase area fraction (per-sample)
            scale_samples = torch.rand(N, generator=self.generator, device=device) * (s_high - s_low) + s_low
            # aspect ratios log-uniform
            log_r_low, log_r_high = math.log(r_low), math.log(r_high)
            ratio_samples = torch.exp(
                torch.rand(N, generator=self.generator, device=device) * (log_r_high - log_r_low) + log_r_low
            )

            total_area = float(H * W)
            erase_area = scale_samples * total_area
            # Derive h,w from area and ratio (h=sqrt(area*ratio), w=sqrt(area/ratio))
            h = torch.sqrt(erase_area * ratio_samples)
            w = torch.sqrt(erase_area / ratio_samples)

            # Round and clamp to valid integer sizes (>=1 and <=H/W-1 for diversity)
            # If H or W is 1, clamp upper bound to H/W respectively.
            h = torch.clamp(h.round().to(torch.int64), 1, max(H - 1, 1))
            w = torch.clamp(w.round().to(torch.int64), 1, max(W - 1, 1))

            # Some samples might end up with h/w exceeding dims after rounding/clamp; adjust apply flag
            valid_hw = (h <= H) & (w <= W)
            apply_mask = apply_mask & valid_hw

            if apply_mask.any():
                # Sample top-left positions (vectorized) using uniform [0,1)
                # i in [0, H-h], j in [0, W-w]
                # Use float sampling scaled by per-sample bounds, then floor.
                i_max = (H - h).clamp(min=0)
                j_max = (W - w).clamp(min=0)

                # Avoid torch.randint with per-sample bounds: scale rand() by (max+1)
                i = (torch.rand(N, generator=self.generator, device=device) * (i_max + 1).to(torch.float32)).floor().to(torch.int64)
                j = (torch.rand(N, generator=self.generator, device=device) * (j_max + 1).to(torch.float32)).floor().to(torch.int64)

                # Build a rectangle mask for all samples at once: (N,1,H,W)
                rows = torch.arange(H, device=device).view(1, 1, H, 1)
                cols = torch.arange(W, device=device).view(1, 1, 1, W)

                # reshape i,j,h,w for broadcasting (N,1,1,1)
                i_b = i.view(N, 1, 1, 1)
                j_b = j.view(N, 1, 1, 1)
                h_b = h.view(N, 1, 1, 1)
                w_b = w.view(N, 1, 1, 1)

                in_rows = (rows >= i_b) & (rows < (i_b + h_b))
                in_cols = (cols >= j_b) & (cols < (j_b + w_b))
                rect = in_rows & in_cols  # (N,1,H,W)

                # Only apply where apply_mask is True
                rect = rect & apply_mask.view(N, 1, 1, 1)

                # Apply erasing: set to constant value
                # Use where() to avoid in-place boolean indexing scatter overhead
                x1 = torch.where(rect, torch.as_tensor(self.value, device=device, dtype=x1.dtype), x1)
                x2 = torch.where(rect, torch.as_tensor(self.value, device=device, dtype=x2.dtype), x2)

        # Reshape back
        out1 = x1.reshape(B, F, R, C, H, W)
        out2 = x2.reshape(B, F, R, C, H, W)
        return out1, out2
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}, scale={self.scale}, ratio={self.ratio}, value={self.value}, generator={self.generator})"
        )