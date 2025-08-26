import draccus
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Iterable

import torch
import torch.nn as nn


@dataclass
class PolicyConfig(draccus.ChoiceRegistry):
    pass


class BaseActor(nn.Module, ABC):
    def __init__(self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int, max_action: float):
        super().__init__()
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim
        self.max_action = max_action

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def actor_loss_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()

    @property
    def critic_loss_parameters(self) -> Iterable[nn.Parameter]:
        return []


class BaseCritic(nn.Module, ABC):
    def __init__(self, obs_shape: Union[int, Tuple[int, ...]], action_dim: int):
        super().__init__()
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for updating the TD3 critic loss.
        """
        raise NotImplementedError

    @abstractmethod
    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for updating the TD3 actor loss.
        """
        raise NotImplementedError

    @property
    def actor_loss_parameters(self) -> Iterable[nn.Parameter]:
        return []

    @property
    def critic_loss_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()
