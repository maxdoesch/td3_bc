import os
import re
import shutil
import torch
from typing import Dict, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import draccus
import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
from tqdm import tqdm
import wandb

from src.td3_bc.buffer import ReplayBuffer
import src.td3_bc.td3_bc as td3_bc
from src.td3_bc.evaluator import Evaluator

@dataclass
class ModeConfig(draccus.ChoiceRegistry):
    name: str
    td3_config: td3_bc.TD3BC_Base_Config

@ModeConfig.register_subclass("pretrain")
@dataclass
class PretrainConfig(ModeConfig):
    name: str = "pretrain"
    td3_config: td3_bc.TD3BC_Config = td3_bc.TD3BC_Config()

    dataset_path: Optional[str] = None

@ModeConfig.register_subclass("refine")
@dataclass
class RefineConfig(ModeConfig):
    name: str = "refine"
    td3_config: td3_bc.TD3BC_Refine_Config = td3_bc.TD3BC_Refine_Config()

    dataset_path: Optional[str] = None

@ModeConfig.register_subclass("online")
@dataclass
class OnlineConfig(ModeConfig):
    name: str = "online"
    td3_config: td3_bc.TD3BC_Online_Config = td3_bc.TD3BC_Online_Config()

    warmup_episodes: int = 5000
    expl_noise: float = 0.1

@dataclass
class TrainerConfig():
    train_mode: ModeConfig

    train_steps: int = 50_000
    eval_freq: int = 500
    checkpoint_freq: int = 5_000
    eval_episodes: int = 16
    batch_size: int = 256
    debug: bool = False
    seed: int = 0

    wandb_project: str = "pushing_offline_rl"

    checkpoint_path: str = "./checkpoints"
    experiment_name: Optional[str] = None

    pretrain_path: Optional[str] = None

    env_name: Optional[str] = "MountainCarContinuous-v0"
    num_envs: Optional[int] = 1

class Trainer(ABC):
    def __init__(self, cfg: TrainerConfig, envs: Optional[VectorEnv] = None):
        self.cfg = cfg

        if envs:
            self.envs = envs
        elif cfg.env_name:
            self.envs = gym.make_vec(self.cfg.env_name, num_envs=self.cfg.num_envs, vectorization_mode="sync")
        else:
            raise ValueError(f'No environment specified.')

        self.obs_dim = self.envs.single_observation_space.shape[0]
        self.action_dim = self.envs.single_action_space.shape[0]
        self.max_action = self.envs.single_action_space.high[0]

        self.agent = td3_bc.get_td3_bc_agent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            train_steps=cfg.train_steps,
            cfg=cfg.train_mode.td3_config,
        )

        self.evaluator = Evaluator(self.envs, self.agent, self.cfg.eval_episodes)

        self.buffer = ReplayBuffer(obs_dim=self.obs_dim, action_dim=self.action_dim)

        self.experiment_name = self._get_experiment_name()
        self.checkpoint_base_dir = self._create_checkpoint_dir(self.experiment_name)
        self._save_config()

    def _save_config(self):
        cfg_path = os.path.join(self.checkpoint_base_dir, 'config.yaml')
        with open(cfg_path, 'w') as f:
            draccus.dump(self.cfg, f)

    def _get_experiment_name(self) -> str:
        if self.cfg.experiment_name:
            return self.cfg.experiment_name

        experiment_info = [
            (int(m.group(1)), d)
            for d in os.listdir(self.cfg.checkpoint_path)
            if os.path.isdir(os.path.join(self.cfg.checkpoint_path, d)) 
            and (m := re.match(r"experiment_(\d+)", d))
        ]

        experiment_info.sort()
        last_id = experiment_info[-1][0] if experiment_info else 0
        last_exp_path = os.path.join(self.cfg.checkpoint_path, experiment_info[-1][1]) if experiment_info else ""

        if self.cfg.train_mode.name == 'pretrain':
            return f"experiment_{last_id + 1}"
        elif self.cfg.train_mode.name in ('refine', 'online'):
            subdir = self.cfg.train_mode.name
            if last_exp_path and subdir in os.listdir(last_exp_path):
                return f"experiment_{last_id + 1}"
            else:
                return f"experiment_{last_id}"

        return f"experiment_{last_id + 1}"

    def _create_checkpoint_dir(self, experiment_name: str) -> str:
        experiment_path = os.path.join(self.cfg.checkpoint_path, experiment_name)
        checkpoint_dir = os.path.join(experiment_path, self.cfg.train_mode.name)

        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

        os.makedirs(checkpoint_dir, exist_ok=True)

        return checkpoint_dir

    @abstractmethod
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Abstract method to get a batch of data for training.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def initialize_replay_buffer(self):
        """
        Abstract method to initialize the replay buffer.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def train(self):
        print(f"{'-' * 40}")
        print(f"Starting training: Mode={self.cfg.train_mode.name} | Experiment: {self.experiment_name} | Seed={self.cfg.seed}")
        print(f"Save run to {self.checkpoint_base_dir}")
        print(f"{'-' * 40}")

        if self.cfg.pretrain_path:
            if os.path.exists(self.cfg.pretrain_path):
                self.agent.load(self.cfg.pretrain_path)
            else:
                raise ValueError(f'Path: {self.cfg.pretrain_path} is not a valid path.')

        run_name = f"{self.experiment_name}_{self.cfg.train_mode.name}"
        wandb.init(
            project=self.cfg.wandb_project,
            group=self.experiment_name,
            name=run_name,
            mode="disabled" if self.cfg.debug else "online",
            config=self.cfg,
        )

        self.initialize_replay_buffer()

        for i in tqdm(range(self.cfg.train_steps), desc="Training Steps"):
            batch = self.get_batch(self.cfg.batch_size)
            metrics = self.agent.train_step(batch)

            wandb.log(metrics, step=i)

            if (i + 1) % self.cfg.eval_freq == 0 or i == self.cfg.train_steps - 1:
                eval_metrics = self.evaluator.evaluate()
                wandb.log(eval_metrics, step=i)

            if (i + 1) % self.cfg.checkpoint_freq == 0 or i == self.cfg.train_steps - 1:
                checkpoint_dir = os.path.join(self.checkpoint_base_dir, f'checkpoint_{i+1}')
                os.makedirs(checkpoint_dir, exist_ok=True)

                self.agent.save(checkpoint_dir)


        wandb.finish()

class OfflineTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig, dataset: Dict, envs: Optional[VectorEnv] = None):
        super().__init__(cfg, envs)

        if dataset:
            self.buffer.convert_dict(dataset)
        elif self.cfg.train_mode.dataset_path:
            if os.path.exists(self.cfg.train_mode.dataset_path):
                raise NotImplementedError("load from file")
            else: 
                raise NotImplementedError("load from minari")
        else:
            raise ValueError(f"Dataset must be provided for offline training mode '{cfg.name}'.")

    def initialize_replay_buffer(self):
        obs_mean, obs_std = self.buffer.compute_dataset_statistics()
        self.buffer.set_dataset_statistics(obs_mean=obs_mean, obs_std=obs_std)

        experiment_path = os.path.join(self.cfg.checkpoint_path, self.experiment_name)
        self.buffer.save_statistics(path=experiment_path)

        self.evaluator.set_obs_statistics(obs_mean, obs_std)

        print(f"Observations normalized and dataset statistics saved to {experiment_path}")

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.buffer.sample(batch_size)

class OnlineTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig, envs: Optional[VectorEnv] = None):
        super().__init__(cfg, envs)
        
        self.obs = np.zeros((self.envs.num_envs, self.obs_dim))
        self.episode_starts = np.ones(self.envs.num_envs, dtype=np.bool)

    def step_and_add(self) -> np.ndarray:
        actions = (
            self.agent.select_action(self.obs) + 
            np.random.normal(0, self.cfg.train_mode.expl_noise, size=self.action_dim)
        ).clip(-self.max_action, self.max_action)

        next_obs, rewards, terminated, truncated, _ = self.envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        mask = ~self.episode_starts
        self.buffer.add(self.obs[mask], actions[mask], next_obs[mask], rewards[mask], dones[mask])

        self.episode_starts = dones
        self.obs = next_obs

        return dones

    def fill_replay_buffer(self):
        print(f'Filling replay buffer with {self.cfg.train_mode.warmup_episodes} warmup episodes.')
        n_envs = self.envs.num_envs

        episode_counts = np.zeros(n_envs, dtype=int)
        episode_targets = np.array(
            [(self.cfg.train_mode.warmup_episodes + i) // n_envs for i in range(n_envs)],
            dtype=int,
        )

        self.obs, _ = self.envs.reset()
        self.episode_starts = np.zeros(self.envs.num_envs, dtype=np.bool)

        while (episode_counts < episode_targets).any():
            dones = self.step_and_add()

            mask = dones & (episode_counts < episode_targets)
            episode_counts[mask] += 1

    def initialize_replay_buffer(self):
        experiment_path = os.path.join(self.cfg.checkpoint_path, self.experiment_name)
        obs_mean, obs_std = self.buffer.load_statistics(path=experiment_path)

        self.evaluator.set_obs_statistics(obs_mean, obs_std)

        self.fill_replay_buffer()

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        self.step_and_add()

        return self.buffer.sample(batch_size)

def get_trainer(cfg: TrainerConfig, dataset: Dict, envs: Optional[VectorEnv] = None) -> Trainer:
    trainer_map = {
        "pretrain": OfflineTrainer,
        "refine": OfflineTrainer,
        "online": OnlineTrainer
    }
    if cfg.train_mode.name not in trainer_map:
        raise ValueError(f"Unknown training mode: {cfg.train_mode.name}")
    return trainer_map[cfg.train_mode.name](cfg, dataset, envs) if cfg.train_mode.name != "online" else trainer_map[cfg.train_mode.name](cfg, envs)

def main():
    episode_length = 100

    cfg = PretrainConfig()
    envs = gym.make_vec(cfg.env_name, num_envs=cfg.num_envs, vectorization_mode="sync")
    episodes = 3
    episode_length = 1000
    dict_dataset = {
        "obs": [np.random.randn(episode_length + 1, envs.single_observation_space.shape[0]) for _ in range(episodes)],
        "acts": [np.ones((episode_length, envs.single_action_space.shape[0])) for _ in range(episodes)],
        "rews": [np.random.randn(episode_length) for _ in range(episodes)],
    }

    trainer = get_trainer(cfg, dict_dataset, envs)
    trainer.train()

if __name__ == "__main__":
    main()