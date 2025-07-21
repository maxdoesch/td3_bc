import os
import copy
import time
import wandb
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import td3_bc.policies as policies
import td3_bc.ftd.auxiliary_pred as aux
from td3_bc.td3_bc import TD3BC_Base, TD3BC_Base_Config


@dataclass
class TD3BC_FTD_Config(TD3BC_Base_Config):
    policy_config: policies.PolicyConfig = policies.FtdPolicyConfig()

    predictor_hidden_dim: int = 1024  # Hidden dimension for auxiliary predictors
    reward_factor: float = 1.0  # Scaling factor for the reward prediction loss
    inverse_factor: float = 1.0  # Scaling factor for the inverse dynamics prediction loss
    max_grad_norm: float = 5.0  # Maximum gradient norm for clipping predictor gradients, 0 means no clipping
    predictors_lr: float = 1e-4  # Learning rate for the auxiliary predictors

    # Update frequencies:
    policy_freq: int = 2  # Frequency of actor updates
    predictors_update_freq: int = 1  # Frequency of auxiliary predictors updates
    predictors_update_slow_freq: int = 50_000  # Frequency of slow updates for auxiliary predictors
    predictors_warmup_steps: int = 10_000  # Number of warmup steps before updating auxiliary predictors

    log_img_freq: int = 500  # Frequency of logging images to wandb


class TD3BC_FTD(TD3BC_Base):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        max_action: float,
        cfg: Optional[TD3BC_FTD_Config] = None,
        device: str | None = None,
    ):
        if cfg is None:
            cfg = TD3BC_FTD_Config()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.total_it = 0

        # === Configuration ===

        self.num_regions = cfg.policy_config.num_regions
        self.num_channels = cfg.policy_config.num_channels
        self.reward_factor = cfg.reward_factor
        self.inverse_factor = cfg.inverse_factor
        self.max_grad_norm = cfg.max_grad_norm
        self.max_action = max_action
        self.discount = cfg.discount
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise * self.max_action
        self.noise_clip = cfg.noise_clip * self.max_action
        self.alpha = cfg.alpha

        self.policy_freq = cfg.policy_freq
        self.predictors_update_freq = cfg.predictors_update_freq
        self.predictors_update_slow_freq = cfg.predictors_update_slow_freq
        self.predictors_warmup_steps = cfg.predictors_warmup_steps

        self.log_img_freq = cfg.log_img_freq

        # === Layers ===

        self.actor, self.critic = policies.get_policy(obs_shape, action_dim, max_action, self.device, cfg.policy_config)
        self.actor_target, self.critic_target = copy.deepcopy(self.actor), copy.deepcopy(self.critic)

        self.complete_selector = self.critic.shared_layers.selector_layers.to(self.device)

        # === Auxiliary Predictors ===

        self.reward_predictor = aux.RewardPredictor(self.critic.encoder, action_dim, cfg.predictor_hidden_dim).to(
            self.device
        )

        self.inverse_dynamic_predictor = aux.InverseDynamicPredictor(
            self.critic.encoder, action_dim, cfg.predictor_hidden_dim
        ).to(self.device)

        # === Optimizers ===

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=cfg.predictors_lr)
        self.inverse_dynamic_predictor_optimizer = torch.optim.Adam(
            self.inverse_dynamic_predictor.parameters(), lr=cfg.predictors_lr
        )

    def update_reward_predictor(self, obs, action, reward):
        """
        Update the reward predictor using the MSE loss between the predicted and actual rewards.
        :param obs: Observations of shape (batch_size, num_stack * (num_regions + 1) * num_channels, height, width).
        :param action: Actions of shape (batch_size, action_dim).
        :param reward: Actual rewards of shape (batch_size, 1).
        """
        predicted_reward = self.reward_predictor(obs, action)  # Shape: (batch_size, 1)
        predict_loss = self.reward_factor * F.mse_loss(reward, predicted_reward)

        self.reward_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(), self.max_grad_norm)
        self.reward_predictor_optimizer.step()

        return predict_loss.item()

    def update_inverse_dynamic_predictor(self, obs, action, next_obs):
        """
        Update the inverse dynamic predictor using the MSE loss between the predicted and actual actions.
        :param obs: Current observations of shape (batch_size, num_stack * (num_regions + 1) * num_channels, height, width).
        :param next_obs: Next observations of shape (batch_size, num_stack * (num_regions + 1) * num_channels, height, width).
        :param action: Actions of shape (batch_size, action_dim).
        """
        predicted_action = self.inverse_dynamic_predictor(obs, next_obs)  # Shape: (batch_size, action_dim)
        predict_loss = self.inverse_factor * F.mse_loss(action, predicted_action)

        self.inverse_dynamic_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(self.inverse_dynamic_predictor.parameters(), self.max_grad_norm)
        self.inverse_dynamic_predictor_optimizer.step()

        return predict_loss.item()

    def _obs_to_input(self, obs) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        elif isinstance(obs, torch.Tensor):
            obs = obs.to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}. Expected np.ndarray or torch.Tensor.")
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)  # Add batch dimension

        assert len(obs.shape) == 4, (
            f"Expected observation shape to be (batch_size, channels, height, width), got {obs.shape}"
        )
        assert obs.shape[0] == 1, f"Expected batch size of 1, got {obs.shape[0]}"

        return obs

    def select_image(self, obs):
        with torch.no_grad():
            current_obs = self._obs_to_input(obs)
            obs, logits = self.complete_selector(current_obs, return_all=True)
            selected_obs = torch.squeeze(obs)[-self.num_channels :].cpu().numpy()
            logits = logits.reshape(-1, self.num_regions)[-1].cpu().detach().tolist()
            print(f"Selected observation shape: {selected_obs.shape}")
            return logits, np.transpose(selected_obs * 255, (1, 2, 0)).astype(np.uint8)

    def save(self, dir_path: str):
        file_path = os.path.join(dir_path, "td3_bc_ftd.pt")
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "reward_predictor_state_dict": self.reward_predictor.state_dict(),
                "reward_predictor_optimizer_state_dict": self.reward_predictor_optimizer.state_dict(),
                "inverse_dynamic_predictor_state_dict": self.inverse_dynamic_predictor.state_dict(),
                "inverse_dynamic_predictor_optimizer_state_dict": self.inverse_dynamic_predictor_optimizer.state_dict(),
            },
            file_path,
        )

        logging.debug(f"Model parameters saved to: {file_path}.")

    def load(self, dir_path: str):
        file_path = os.path.join(dir_path, "td3_bc_ftd.pt")
        checkpoint = torch.load(file_path)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.critic_target = copy.deepcopy(self.critic)

        self.reward_predictor.load_state_dict(checkpoint["reward_predictor_state_dict"])
        self.reward_predictor_optimizer.load_state_dict(checkpoint["reward_predictor_optimizer_state_dict"])

        self.inverse_dynamic_predictor.load_state_dict(checkpoint["inverse_dynamic_predictor_state_dict"])
        self.inverse_dynamic_predictor_optimizer.load_state_dict(
            checkpoint["inverse_dynamic_predictor_optimizer_state_dict"]
        )

        logging.debug(f"Model parameters loaded from: {file_path}.")

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float | np.ndarray]:
        metrics = {}
        start_time = time.time()
        self.total_it += 1

        # Slow down the update frequency of the predictors
        if self.total_it > self.predictors_warmup_steps and self.total_it % self.predictors_update_slow_freq == 0:
            self.unsupervised_update_freq = self.unsupervised_update_freq + 1
            logging.debug(f"Slowing predictors update frequency to {self.unsupervised_update_freq}.")

        # Update critic
        critic_loss, avg_q1, avg_q2 = self.update_critic(**batch)

        metrics["train/critic_loss"] = critic_loss
        metrics["train/avg_q1"] = avg_q1
        metrics["train/avg_q2"] = avg_q2

        # Update actor
        if self.total_it % self.policy_freq == 0:
            actions_taken, actor_loss, bc_loss, _ = self.update_actor(batch["obs"], batch["action"])

            metrics["train/actor_loss"] = actor_loss
            metrics["train/bc_loss"] = bc_loss
            metrics["train/actions_taken"] = actions_taken

            # Update the frozen target models
            self.update_critic_target()
            self.update_actor_target()

        # Update auxiliary predictors
        if self.total_it > self.predictors_warmup_steps and self.total_it % self.predictors_update_freq == 0:
            if self.reward_factor != 0.0:
                reward_predictor_loss = self.update_reward_predictor(batch["obs"], batch["action"], batch["reward"])
                metrics["train/reward_predictor_loss"] = reward_predictor_loss

            if self.inverse_factor != 0.0:
                inverse_dynamic_loss = self.update_inverse_dynamic_predictor(
                    batch["obs"], batch["action"], batch["next_obs"]
                )
                metrics["train/inverse_dynamic_loss"] = inverse_dynamic_loss

        if self.total_it % self.log_img_freq == 0:
            metrics["train/raw_images"] = wandb.Image(batch["obs"][-1][:3])
            metrics["train/ftd_images"] = wandb.Image(self.select_image(batch["obs"][0])[1])

        metrics["train/time"] = time.time() - start_time

        return metrics
