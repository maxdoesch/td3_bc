from dataclasses import dataclass
import copy
import numpy as np
import torch
import torch.nn.functional as F

import td3_bc.ftd.auxiliary_pred as aux
from td3_bc.td3_bc import TD3BC_Base, TD3BC_Base_Config
import td3_bc.policies as policies


@dataclass
class TD3BC_FTD_Base_Config(TD3BC_Base_Config):
    num_regions: int = 9              # Maximum number of segmented regions
    num_channels: int = 3             # Number of input channels
    num_stack: int = 3                # Number of frames stacked together as a single observation
    num_selector_layers: int = 5      # Number of convolutional layers in the attention selector
    num_filters: int = 32             # Number of filters in the convolutional layers
    embed_dim: int = 128              # Dimension of the embedding space for attention
    num_attention_heads: int = 4      # Number of attention heads
    num_shared_layers: int = 11       # Number of shared convolutional layers
    num_head_layers: int = 0          # Number of hidden layers in the head CNN
    projection_dim: int = 100         # Dimension of the projection space for actor and critic; must match actor and critic input dim
    predictor_hidden_dim: int = 1024  # Hidden dimension for auxiliary predictors
    reward_factor: float = 1.0        # Scaling factor for the reward prediction loss
    inverse_factor: float = 1.0       # Scaling factor for the inverse dynamics prediction loss
    max_grad_norm: float = 5.0        # Maximum gradient norm for clipping predictor gradients, 0 means no clipping
    predictors_lr: float = 1e-4       # Learning rate for the auxiliary predictors

    def get_shared_layers_config(self):
        return policies.SharedFTDLayersConfig(
            num_regions=self.num_regions,
            num_channels=self.num_channels,
            num_stack=self.num_stack,
            num_selector_layers=self.num_selector_layers,
            num_filters=self.num_filters,
            embed_dim=self.embed_dim,
            num_attention_heads=self.num_attention_heads,
            num_shared_layers=self.num_shared_layers,
            num_head_layers=self.num_head_layers,
            projection_dim=self.projection_dim
        )


class TD3BC_FTD_Base(TD3BC_Base):

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        max_action: float,
        cfg: TD3BC_FTD_Base_Config = TD3BC_FTD_Base_Config(),
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # === Configuration ===

        self.num_regions = cfg.num_regions
        self.num_channels = cfg.num_channels
        self.reward_factor = cfg.reward_factor
        self.inverse_factor = cfg.inverse_factor
        self.max_grad_norm = cfg.max_grad_norm
        self.max_action = max_action
        self.discount = cfg.discount
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise * self.max_action
        self.noise_clip = cfg.noise_clip * self.max_action
        self.alpha = cfg.alpha

        # === Layers ===

        shared_layers_config = cfg.get_shared_layers_config()
        self.actor, self.critic = policies.policy_factory("ftd", obs_shape, action_dim, max_action, self.device, shared_layers_config)
        self.actor_target, self.critic_target = copy.deepcopy(self.actor), copy.deepcopy(self.critic)

        self.complete_selector = self.critic.shared_layers.selector_layers.to(self.device)

        # === Auxiliary Predictors ===

        self.reward_predictor = aux.RewardPredictor(self.critic.encoder,
                                                    action_dim,
                                                    cfg.predictor_hidden_dim).to(self.device)

        self.inverse_dynamic_predictor = aux.InverseDynamicPredictor(self.critic.encoder,
                                                                     action_dim,
                                                                     cfg.predictor_hidden_dim).to(self.device)

        # === Optimizers ===

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=cfg.predictors_lr)
        self.inverse_dynamic_predictor_optimizer = torch.optim.Adam(self.inverse_dynamic_predictor.parameters(), lr=cfg.predictors_lr)

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

        assert len(obs.shape) == 4, f"Expected observation shape to be (batch_size, channels, height, width), got {obs.shape}"
        assert obs.shape[0] == 1, f"Expected batch size of 1, got {obs.shape[0]}"

        return obs

    def select_image(self, obs):
        with torch.no_grad():
            current_obs = self._obs_to_input(obs)
            obs, logits = self.complete_selector(current_obs, return_all=True)
            selected_obs = torch.squeeze(obs)[-self.num_channels:].cpu().numpy()
            logits = logits.reshape(-1, self.num_regions)[-1].cpu().detach().tolist()
            print(f"Selected observation shape: {selected_obs.shape}")
            return logits, np.transpose(selected_obs * 255, (1, 2, 0)).astype(np.uint8)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float | np.ndarray]:
        pass
