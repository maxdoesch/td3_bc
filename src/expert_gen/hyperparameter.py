#import flax.nnx as nn
import torch.nn as nn
from stable_baselines3.common.utils import get_linear_fn

HYPERPARAMETERS = {
    "ppo": {
        "dmc_cheetah_run_1-v1": {
            "normalize": True,
            "n_envs": 16,
            "policy": "MlpPolicy",
            "n_timesteps": 2_000_000,
            "batch_size": 256,                     # SB3 default
            "n_steps": 512,                         # SB3 default
            "gamma": 0.98,                          # SB3 default
            "learning_rate": get_linear_fn(start=5e-4, end=1e-4, end_fraction=1.0),
            "ent_coef": 0.0,                      # SB3 default
            "clip_range": 0.2,                    # SB3 default
            "n_epochs": 10,                       # SB3 default
            "gae_lambda": 0.9,                   # SB3 default
            "max_grad_norm": 0.5,                 # SB3 default
            "vf_coef": 0.5,                       # SB3 default
            "use_sde": True,                     # SB3 default
            "sde_sample_freq": 4,                # SB3 default
            "normalize_advantage": True,          # SB3 default
            "policy_kwargs": {
                "log_std_init": -0.5,
                "ortho_init": True,               # SB3 default for PPO
                "activation_fn": nn.Tanh,         # SB3 default for PPO
                "net_arch": dict(pi=[64, 64], vf=[64, 64]),  # SB3 default
            },
            "normalize_kwargs": {
                "norm_obs": True,
                "norm_reward": True,
            },
            "env_kwargs": {
                "action_repeat": 2,
            },
        },
        "dmc_hopper_run_1-v1": {
            "normalize": True,
            "n_envs": 1,
            "policy": "MlpPolicy",
            "n_timesteps": 1_000_000,
            "batch_size": 32,
            "n_steps": 512,
            "gamma": 0.999,
            "learning_rate": 9.80828e-05,
            "ent_coef": 0.00229519,
            "clip_range": 0.2,
            "n_epochs": 5,
            "gae_lambda": 0.99,
            "max_grad_norm": 0.7,
            "vf_coef": 0.835671,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
        "dmc_humanoid_run_1-v1": {
            "normalize": True,
            "n_envs": 1,
            "policy": "MlpPolicy",
            "n_timesteps": 10_000_000,
            "batch_size": 256,
            "n_steps": 512,
            "gamma": 0.95,
            "learning_rate": 3.56987e-05,
            "ent_coef": 0.00238306,
            "clip_range": 0.3,
            "n_epochs": 5,
            "gae_lambda": 0.9,
            "max_grad_norm": 2,
            "vf_coef": 0.431892,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
        "dmc_reacher_easy_1-v1": dict(
            n_envs=16,
            normalize=True,
            normalize_kwargs=dict(norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0),
            n_steps=1024,  # per env → 16*1024 =  rollout batch
            batch_size=2048,
            n_epochs=10,
            learning_rate=get_linear_fn(start=1e-4, end=3e-5, end_fraction=1.0),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,  # optional: 1e-3 if exploration is too timid
            max_grad_norm=0.5,
            use_sde=True,  # helps on small state tasks
            sde_sample_freq=4,
            policy="MlpPolicy",
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), ortho_init=True),
            n_timesteps=1_000_000,
            env_kwargs=dict(action_repeat=4),
        ),
        "dmc_cartpole_swingup_1-v1": dict(
            n_envs=16,
            normalize_advantage=True,
            normalize=True,
            normalize_kwargs=dict(norm_obs=True, norm_reward=True),
            n_steps=512,  # per env → 16*1024 =  rollout batch
            batch_size=256,
            n_epochs=10,
            learning_rate=get_linear_fn(start=3e-4, end=3e-5, end_fraction=1.0),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,  # optional: 1e-3 if exploration is too timid
            max_grad_norm=0.5,
            use_sde=True,  # helps on small state tasks
            sde_sample_freq=4,
            policy="MlpPolicy",
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=nn.Tanh,
                ortho_init=True),
            n_timesteps=3_000_000,
            env_kwargs=dict(action_repeat=4),
        ),
    },
    "td3": {
        "dmc_reacher_easy_1-v1": dict(
            n_envs=1,
            policy="MlpPolicy",
            learning_starts=100,
            n_timesteps=1_000_000,
            env_kwargs=dict(action_repeat=4),
        ),
        "dmc_cheetah_run_1-v1": dict(
            n_envs=1,
            policy="MlpPolicy",
            learning_starts=100,
            n_timesteps=1_000_000,
            env_kwargs=dict(action_repeat=4),
        )
    },
}
