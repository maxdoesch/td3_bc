from typing import Callable, Tuple

from .policy import PolicyConfig, BaseActor, BaseCritic
from .mlp import MlpPolicyConfig, MlpActor, MlpCritic
from .cnn import CnnPolicyConfig, CnnEncoder, CnnActor, CnnCritic
from .ftd import FtdPolicyConfig, SharedFTDLayers, FTDActor, FTDCritic

POLICY_REGISTRY = {}


def register_policy(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        if name in POLICY_REGISTRY:
            raise ValueError(f"Policy {name} already registered")
        POLICY_REGISTRY[name] = fn
        return fn

    return decorator


@register_policy("mlp")
def build_mlp_policy(
    obs_shape: int | tuple[int, int, int],
    action_dim: int,
    max_action: float,
    device: str,
    cfg: MlpPolicyConfig,
) -> Tuple[BaseActor, BaseCritic]:
    actor = MlpActor(
        obs_shape,
        action_dim,
        hidden_dim=cfg.actor_hidden_dim,
        n_layers=cfg.actor_n_layers,
        max_action=max_action,
    ).to(device)
    critic = MlpCritic(
        obs_shape,
        action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        n_layers=cfg.critic_n_layers,
    ).to(device)
    return actor, critic


@register_policy("cnn")
def build_cnn_policy(
    obs_shape: tuple[int, int, int],
    action_dim: int,
    max_action: float,
    device: str,
    cfg: CnnPolicyConfig,
) -> Tuple[BaseActor, BaseCritic]:
    shared_encoder = CnnEncoder(obs_shape, hidden_dim=cfg.encoder_hidden_dim).to(device)
    actor = CnnActor(
        shared_encoder,
        obs_shape,
        action_dim,
        hidden_dim=cfg.actor_hidden_dim,
        n_layers=cfg.actor_n_layers,
        max_action=max_action,
    ).to(device)
    critic = CnnCritic(
        shared_encoder,
        obs_shape,
        action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        n_layers=cfg.critic_n_layers,
    ).to(device)
    return actor, critic


@register_policy("ftd")
def build_ftd_policy(
    obs_shape: tuple[int, int, int],
    action_dim: int,
    max_action: float,
    device: str,
    cfg: FtdPolicyConfig,
) -> Tuple[BaseActor, BaseCritic]:
    shared_layers = SharedFTDLayers(obs_shape, cfg.shared_layers_cfg).to(device)
    actor = FTDActor(shared_layers, obs_shape, action_dim, max_action, cfg.actor_cfg).to(device)
    critic = FTDCritic(shared_layers, obs_shape, action_dim, cfg.critic_cfg).to(device)
    return actor, critic


def get_policy(
    obs_shape: int | tuple[int, int, int],
    action_dim: int,
    max_action: float,
    device: str,
    cfg: PolicyConfig,
) -> Tuple[BaseActor, BaseCritic]:
    name = PolicyConfig.get_choice_name(type(cfg))
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown Policy Configuration type: {type(cfg)}")
    builder = POLICY_REGISTRY[name]
    return builder(obs_shape, action_dim, max_action, device, cfg)
