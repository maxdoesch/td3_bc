from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from td3_bc.td3_bc import TD3BC_Base, TD3BC_Base_Config
from td3_bc.buffer import ReplayBuffer
from td3_bc.ftd.image_attention import ImageAttentionSelectorLayers
import td3_bc.ftd.auxiliary_pred as aux
import td3_bc.ftd.modules as m


@dataclass
class TD3BC_FTD_Base_Config(TD3BC_Base_Config):
    num_regions: int                # Maximum number of segmented regions
    num_channels: int               # Number of input channels
    num_stack: int                  # Number of frames stacked together as a single observation
    num_selector_layers: int        # Number of convolutional layers in the attention selector
    num_filters: int                # Number of filters in the convolutional layers
    embed_dim: int                  # Dimension of the embedding space for attention
    num_attention_heads: int        # Number of attention heads
    num_shared_layers: int          # Number of shared convolutional layers
    num_head_layers: int            # Number of hidden layers in the head CNN
    projection_dim: int             # Dimension of the projection space for actor and critic; must match actor and critic input dim
    predictor_hidden_dim: int       # Hidden dimension for auxiliary predictors
    reward_accumulate_steps: int    # Number of steps to accumulate rewards for prediction
    inv_accumulate_steps: int       # Number of steps to accumulate actions for inverse dynamics prediction
    reward_factor: float            # Scaling factor for the reward prediction loss
    inverse_factor: float           # Scaling factor for the inverse dynamics prediction loss
    max_grad_norm: float            # Maximum gradient norm for clipping predictor gradients, 0 means no clipping
    selector_lr: float              # Learning rate for the attention selector and auxiliary predictors


class TD3BC_FTD_Base(TD3BC_Base):

    def __init__(
        self,
        obs_shape: int | tuple[int, ...],
        action_dim: int,
        max_action: float,
        cfg: TD3BC_FTD_Base_Config | None = None,
        device: str | None = None,
    ):
        if cfg is None:
            cfg = TD3BC_FTD_Base_Config()
        super().__init__(obs_shape, action_dim, max_action, cfg, device)

        # === Configuration ===
        
        self.num_regions = cfg.num_regions
        self.num_channels = cfg.num_channels
        self.reward_accumulate_steps = cfg.reward_accumulate_steps
        self.inv_accumulate_steps = cfg.inv_accumulate_steps
        self.reward_factor = cfg.reward_factor
        self.inverse_factor = cfg.inverse_factor
        self.max_grad_norm = cfg.max_grad_norm
        
        # === Layers ===

        selector_layers = ImageAttentionSelectorLayers(
            obs_shape, cfg.num_regions, cfg.num_channels,
            cfg.num_stack, cfg.num_selector_layers,
            cfg.num_filters, cfg.embed_dim, cfg.num_attention_heads)

        self.complete_selector = selector_layers.to(self.device)

        selector_cnn = m.SelectorCNN(
            selector_layers, obs_shape, cfg.num_regions,
            cfg.num_channels, cfg.num_stack,
            cfg.num_shared_layers, cfg.num_filters).to(self.device)

        head_cnn = m.HeadCNN(selector_cnn.out_shape,
                                  cfg.num_head_layers, cfg.num_filters).to(self.device)

        actor_projection = m.RLProjection(
            head_cnn.out_shape, cfg.projection_dim)

        critic_projection = m.RLProjection(
            head_cnn.out_shape, cfg.projection_dim)

        self.actor_encoder = m.Encoder(  # Use this for actor
            selector_cnn,
            head_cnn,
            actor_projection
        )

        self.critic_encoder = m.Encoder(  # Use this for critic
            selector_cnn,
            head_cnn,
            critic_projection
        )

        # === Auxiliary Predictors ===
        self.reward_predictor = aux.RewardPredictor(self.critic_encoder, (action_dim,), cfg.reward_accumulate_steps,
                                                    cfg.predictor_hidden_dim, cfg.num_filters).to(self.device)

        self.inverse_dynamic_predictor = aux.InverseDynamicPredictor(self.critic_encoder, (action_dim,),
                                                                     cfg.inv_accumulate_steps,
                                                                     cfg.predictor_hidden_dim, cfg.num_filters).to(self.device)

        # === Optimizers ===
        self.selector_optimizer = torch.optim.Adam(self.complete_selector.parameters(), lr=cfg.selector_lr)
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=cfg.selector_lr)
        self.inverse_dynamic_predictor_optimizer = torch.optim.Adam(self.inverse_dynamic_predictor.parameters(), lr=cfg.selector_lr)

    
    def update_reward_predictor(self, replay_buffer: ReplayBuffer):
        """
        Update the reward predictor using the sampled observations, actions, and rewards.
        **Work in progress**
        """
        sample_dict = replay_buffer.sample(self.reward_accumulate_steps)
        obs, action, reward = sample_dict['obs'], sample_dict['action'], sample_dict['reward']
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

    def update_inverse_dynamic_predictor(self, replay_buffer: ReplayBuffer):
        """
        Update the inverse dynamic predictor using the sampled observations, actions, and next observations.
        **Work in progress**
        """
        sample_dict = replay_buffer.sample(self.inv_accumulate_steps)
        obs, action, next_obs = sample_dict['obs'], sample_dict['action'], sample_dict['next_obs']
        concatenated_pre_action = torch.concatenate(action[:-1], dim=1) if len(action) > 1 else torch.tensor([]).cuda()
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
            obs, logits= self.complete_selector(current_obs, return_all=True)
            selected_obs = torch.squeeze(obs)[-self.num_channels:].cpu().numpy()
            logits = logits.reshape(-1, self.num_regions)[-1].cpu().detach().tolist()
            return logits, np.transpose(selected_obs * 255, (1, 2, 0)).astype(np.uint8)