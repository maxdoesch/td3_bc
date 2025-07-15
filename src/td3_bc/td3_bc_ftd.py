from dataclasses import dataclass
import copy
import numpy as np
import torch
import torch.nn.functional as F

import td3_bc.policies as policies
from td3_bc.td3_bc import TD3BC_Base, TD3BC_Base_Config
import td3_bc.ftd.auxiliary_pred as aux


@dataclass
class TD3BC_FTD_Base_Config(TD3BC_Base_Config):
    num_regions: int                  # Maximum number of segmented regions
    num_channels: int                 # Number of input channels
    num_stack: int                    # Number of frames stacked together as a single observation
    num_selector_layers: int          # Number of convolutional layers in the attention selector
    num_filters: int                  # Number of filters in the convolutional layers
    embed_dim: int                    # Dimension of the embedding space for attention
    num_attention_heads: int          # Number of attention heads
    num_shared_layers: int            # Number of shared convolutional layers
    num_head_layers: int              # Number of hidden layers in the head CNN
    projection_dim: int               # Dimension of the projection space for actor and critic; must match actor and critic input dim
    predictor_hidden_dim: int         # Hidden dimension for auxiliary predictors
    reward_factor: float              # Scaling factor for the reward prediction loss
    inverse_factor: float             # Scaling factor for the inverse dynamics prediction loss
    max_grad_norm: float              # Maximum gradient norm for clipping predictor gradients, 0 means no clipping
    predictors_lr: float              # Learning rate for the auxiliary predictors
    reward_accumulate_steps: int = 1  # Number of steps to accumulate for Reward prediction - Not Used
    inv_accumulate_steps: int = 1     # Number of steps to accumulate for inverse dynamics prediction - Not Used

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
        self.reward_accumulate_steps = cfg.reward_accumulate_steps
        self.inv_accumulate_steps = cfg.inv_accumulate_steps
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

        self.reward_predictor = aux.RewardPredictor(self.critic.encoder, (action_dim,), cfg.reward_accumulate_steps,
                                                    cfg.predictor_hidden_dim, cfg.num_filters).to(self.device)

        self.inverse_dynamic_predictor = aux.InverseDynamicPredictor(self.critic.encoder, (action_dim,),
                                                                     cfg.inv_accumulate_steps,
                                                                     cfg.predictor_hidden_dim, cfg.num_filters).to(self.device)

        # === Optimizers ===

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=cfg.predictors_lr)
        self.inverse_dynamic_predictor_optimizer = torch.optim.Adam(self.inverse_dynamic_predictor.parameters(), lr=cfg.predictors_lr)

    def update_reward_predictor(self, obs, action, reward):
        """
        Update the reward predictor using the sampled observations, actions, and rewards.
        **Work in progress**
        """
        concatenated_action = torch.concatenate(action, dim=1)
        concatenated_reward = torch.sum(torch.concatenate(reward, dim=1), dim=1, keepdim=True)
        predicted_reward = self.reward_predictor(obs, concatenated_action)
        predict_loss = self.reward_factor * \
            F.mse_loss(concatenated_reward, predicted_reward)

        self.reward_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(), self.max_grad_norm)
        self.reward_predictor_optimizer.step()

        return predict_loss.item()

    def update_inverse_dynamic_predictor(self, obs, action, next_obs):
        """
        Update the inverse dynamic predictor using the sampled observations, actions, and next observations.
        **Work in progress**
        """
        concatenated_pre_action = torch.concatenate(action[:-1], dim=1) if len(action) > 1 else torch.tensor([]).to(self.device)
        predicted_action = self.inverse_dynamic_predictor(obs, next_obs, concatenated_pre_action)
        predict_loss = self.inverse_factor * \
            F.mse_loss(action[-1], predicted_action)

        self.inverse_dynamic_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.inverse_dynamic_predictor.parameters(), self.max_grad_norm)
        self.inverse_dynamic_predictor_optimizer.step()

        return predict_loss.item()

    def _obs_to_input(self, obs) -> torch.Tensor:
        _obs = torch.FloatTensor(obs).to(self.device)
        if len(_obs.shape) == 3:
            _obs = _obs.unsqueeze(0)  # Add batch dimension
        return _obs

    def select_image(self, obs):
        with torch.no_grad():
            current_obs = self._obs_to_input(obs)
            obs, logits = self.complete_selector(current_obs, return_all=True)
            selected_obs = torch.squeeze(obs)[-self.num_channels:].cpu().numpy()
            logits = logits.reshape(-1, self.num_regions)[-1].cpu().detach().tolist()
            return logits, np.transpose(selected_obs * 255, (1, 2, 0)).astype(np.uint8)
