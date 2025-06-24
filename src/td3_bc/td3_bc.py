import os
import copy
import time
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional

import td3_bc.policies as policies


@dataclass
class TD3BC_Base_Config:
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    alpha: float = 0.4

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4


@dataclass
class TD3BC_Config(TD3BC_Base_Config):
    policy_freq: int = 2


@dataclass
class TD3BC_Refine_Config(TD3BC_Base_Config):
    scaling_factor_lambda: float = 5.0


@dataclass
class TD3BC_Online_Config(TD3BC_Config):
    alpha_end: float = 0.2


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        pass


class DummyAgent(BaseAgent):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def select_action(self, obs):
        assert obs.shape[-1] == self.obs_dim, "The dimension of obs and the internal obs_dim do not match."

        return np.random.randn(obs.shape[0], self.action_dim) * self.max_action


class TD3BC_Base(BaseAgent):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_action: float,
        cfg: Optional[TD3BC_Base_Config] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.actor, self.critic = policies.policy_factory("mlp", obs_dim, action_dim, max_action, self.device)
        self.actor_target, self.critic_target = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        if cfg is None:
            cfg = TD3BC_Base_Config()

        self.max_action = max_action
        self.discount = cfg.discount
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise * self.max_action
        self.noise_clip = cfg.noise_clip * self.max_action
        self.alpha = cfg.alpha

    @torch.inference_mode
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = self.actor(obs).cpu().numpy()
        return action

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float | np.ndarray]:
        pass

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        not_done: torch.Tensor,
    ) -> Tuple[float, float, float]:
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Compute critic loss
        critic_loss = functional.mse_loss(current_Q1, target_Q) + functional.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item(), current_Q1.mean().item(), current_Q2.mean().item()

    def update_actor(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[np.ndarray, float, float, float]:
        # Compute actor loss
        pi = self.actor(obs)
        q1_value = self.critic.q1(obs, pi)
        q1_value_norm = q1_value / q1_value.abs().mean().detach()

        bc_loss = functional.mse_loss(pi, action)
        actor_loss = -q1_value_norm.mean() + self.alpha * bc_loss

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return pi.detach().cpu().numpy(), actor_loss.item(), bc_loss.item(), q1_value.mean().item()

    def update_actor_target(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_critic_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, dir_path: str):
        file_path = os.path.join(dir_path, "td3_bc.pt")
        torch.save(
            {
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            },
            file_path,
        )

        logging.debug(f"Model parameters saved to: {file_path}.")

    def load(self, dir_path: str):
        file_path = os.path.join(dir_path, "td3_bc.pt")
        checkpoint = torch.load(file_path)

        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.actor_target = copy.deepcopy(self.actor)

        logging.debug(f"Model parameters loaded from: {file_path}.")


class TD3BC(TD3BC_Base):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_action: float,
        cfg: Optional[TD3BC_Config] = None,
        device: Optional[str] = None,
    ):
        if cfg is None:
            cfg = TD3BC_Config()

        super().__init__(obs_dim=obs_dim, action_dim=action_dim, max_action=max_action, cfg=cfg, device=device)

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
        obs_dim: int,
        action_dim: int,
        max_action: float,
        cfg: Optional[TD3BC_Refine_Config] = None,
        device: Optional[str] = None,
    ):
        if cfg is None:
            cfg = TD3BC_Refine_Config()

        super().__init__(obs_dim=obs_dim, action_dim=action_dim, max_action=max_action, cfg=cfg, device=device)

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
        obs_dim: int,
        action_dim: int,
        max_action: float,
        train_steps: int,
        cfg: Optional[TD3BC_Online_Config] = None,
        device: Optional[str] = None,
    ):
        if cfg is None:
            cfg = TD3BC_Online_Config()

        super().__init__(obs_dim=obs_dim, action_dim=action_dim, max_action=max_action, cfg=cfg, device=device)

        self.alpha_decay_rate = np.exp(np.log(cfg.alpha_end / self.alpha) / train_steps)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float | np.ndarray]:
        metrics = {}
        metrics = super().train_step(batch)
        metrics["train/alpha"] = self.alpha

        self.alpha *= self.alpha_decay_rate

        return metrics


def get_td3_bc_agent(
    obs_dim: int, action_dim: int, max_action: float, train_steps: int, cfg: TD3BC_Base_Config, device: str
) -> TD3BC_Base:
    if type(cfg) is TD3BC_Config:
        return TD3BC(obs_dim, action_dim, max_action, cfg, device)
    elif type(cfg) is TD3BC_Refine_Config:
        return TD3BC_Refine(obs_dim, action_dim, max_action, cfg, device)
    elif type(cfg) is TD3BC_Online_Config:
        return TD3BC_Online(obs_dim, action_dim, max_action, train_steps, cfg, device)
    else:
        raise ValueError(f"Unsupported configuration type: {type(cfg)}")


if __name__ == "__main__":
    cfg = TD3BC_Config()
    agent = TD3BC(3, 4, 1.0, cfg)

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
