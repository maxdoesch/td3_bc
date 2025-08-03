import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .td3_bc import TD3BC_Base, TD3BC_Base_Config


@dataclass
class TD3BC_Config(TD3BC_Base_Config):
    policy_freq: int = 2


@dataclass
class TD3BC_Refine_Config(TD3BC_Base_Config):
    scaling_factor_lambda: float = 5.0


@dataclass
class TD3BC_Online_Config(TD3BC_Config):
    alpha_end: float = 0.2


class TD3BC(TD3BC_Base):
    def __init__(
        self,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        max_action: float,
        cfg: Optional[TD3BC_Config] = None,
        device: Optional[str] = None,
    ):
        if cfg is None:
            cfg = TD3BC_Config()

        super().__init__(obs_shape=obs_shape, action_dim=action_dim, max_action=max_action, cfg=cfg, device=device)

        self.policy_freq = cfg.policy_freq

        self.total_it = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float | np.ndarray]:
        metrics = {}
        start_time = time.time()
        self.total_it += 1

        critic_loss, avg_q1, avg_q2 = self.update_critic(**batch)

        metrics["train/critic_loss"] = critic_loss
        metrics["train/avg_q1"] = avg_q1
        metrics["train/avg_q2"] = avg_q2

        actor_loss = None
        bc_loss = None
        actions_taken = None
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actions_taken, actor_loss, bc_loss, _ = self.update_actor(batch["obs"], batch["action"])

            metrics["train/actor_loss"] = actor_loss
            metrics["train/bc_loss"] = bc_loss
            metrics["train/actions_taken"] = actions_taken

            # Update the frozen target models
            self.update_critic_target()
            self.update_actor_target()

        metrics["train/time"] = time.time() - start_time

        return metrics


class TD3BC_Refine(TD3BC_Base):
    def __init__(
        self,
        obs_shape: int,
        action_dim: int,
        max_action: float,
        cfg: Optional[TD3BC_Refine_Config] = None,
        device: Optional[str] = None,
    ):
        if cfg is None:
            cfg = TD3BC_Refine_Config()

        super().__init__(obs_shape=obs_shape, action_dim=action_dim, max_action=max_action, cfg=cfg, device=device)

        self.alpha = self.alpha / cfg.scaling_factor_lambda

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float | np.ndarray]:
        metrics = {}
        start_time = time.time()

        # Only update the policy, not the critic
        actions_taken, actor_loss, bc_loss, avg_q = self.update_actor(batch["obs"], batch["action"])
        metrics["train/actor_loss"] = actor_loss
        metrics["train/bc_loss"] = bc_loss
        metrics["train/actions_taken"] = actions_taken
        metrics["train/avg_q"] = avg_q

        # Update target network
        self.update_actor_target()

        metrics["train/time"] = time.time() - start_time

        return metrics


class TD3BC_Online(TD3BC):
    def __init__(
        self,
        obs_shape: int,
        action_dim: int,
        max_action: float,
        train_steps: int,
        cfg: Optional[TD3BC_Online_Config] = None,
        device: Optional[str] = None,
    ):
        if cfg is None:
            cfg = TD3BC_Online_Config()

        super().__init__(obs_shape=obs_shape, action_dim=action_dim, max_action=max_action, cfg=cfg, device=device)

        self.alpha_decay_rate = np.exp(np.log(cfg.alpha_end / self.alpha) / train_steps)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float | np.ndarray]:
        metrics = {}
        metrics = super().train_step(batch)
        metrics["train/alpha"] = self.alpha

        self.alpha *= self.alpha_decay_rate

        return metrics


if __name__ == "__main__":
    obs_shape = (3,)
    action_dim = 4
    max_action = 1.0

    cfg = TD3BC_Config()
    agent = TD3BC(obs_shape, action_dim, max_action, cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = {
        "obs": torch.randn(32, 3).to(device),
        "action": torch.randn(32, 4).to(device),
        "next_obs": torch.randn(32, 3).to(device),
        "reward": torch.randn(32, 1).to(device),
        "not_done": torch.ones(32, 1).to(device),
    }

    metrics = agent.train_step(batch)
    print("Training metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
