import draccus
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn


@dataclass
class PolicyConfig(draccus.ChoiceRegistry):
    pass


class BaseActor(nn.Module, ABC):
    @abstractmethod
    def __init__(self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, max_action: float):
        super().__init__()
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim
        self.max_action = max_action

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass


class BaseCritic(nn.Module, ABC):
    def __init__(self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int):
        super().__init__()
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass
