import os
import re
import shutil
import torch
import logging
import minari
from typing import Dict, Optional, Union, List
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

@ModeConfig.register_subclass("refine")
@dataclass
class RefineConfig(ModeConfig):
    name: str = "refine"
    td3_config: td3_bc.TD3BC_Refine_Config = td3_bc.TD3BC_Refine_Config()

@ModeConfig.register_subclass("online")
@dataclass
class OnlineConfig(ModeConfig):
    name: str = "online"
    td3_config: td3_bc.TD3BC_Online_Config = td3_bc.TD3BC_Online_Config()

    warmup_steps: int = 5000
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

    seeds: Union[List, int] = 0
    n_seeds: int = 1

    wandb_project: str = "pushing_offline_rl"

    checkpoint_dir: str = "./checkpoints"
    experiment_name: Optional[str] = None

    pretrain_dir: Optional[str] = None
    pretrain_checkpoint: Optional[int] = None

    dataset_path: Optional[str] = None

    env_name: Optional[str] = None
    num_envs: Optional[int] = 1

    @property
    def dataset_statistics_path(self) -> str:
        return os.path.join(self.experiment_dir, "dataset_statistics.json")


    def _get_last_experiment_id(self) -> int:
        if not os.path.exists(self.checkpoint_dir):
            return 0

        experiment_ids = [
            int(m.group(1))
            for d in os.listdir(self.checkpoint_dir)
            if os.path.isdir(os.path.join(self.checkpoint_dir, d)) 
            and (m := re.match(r"experiment_(\d+)", d))
        ]
        return max(experiment_ids, default=0)

    def _set_experiment_name(self):
        if self.experiment_name:
            return

        last_id = self._get_last_experiment_id()
        candidate_id = last_id + 1
        candidate_name = f"experiment_{candidate_id}"

        # Check if mode-specific subdir already exists in last experiment
        if self.train_mode.name in ('refine', 'online'):
            last_exp_dir = os.path.join(self.checkpoint_dir, f"experiment_{last_id}")
            if os.path.isdir(last_exp_dir) and self.train_mode.name not in os.listdir(last_exp_dir):
                candidate_name = f"experiment_{last_id}"

        self.experiment_name = candidate_name

    def _create_checkpoint_dir(self):
        self.experiment_dir = os.path.join(self.checkpoint_dir, self.experiment_name)
        self.checkpoint_mode_dir = os.path.join(self.experiment_dir, self.train_mode.name)

        if os.path.exists(self.checkpoint_mode_dir):
            shutil.rmtree(self.checkpoint_mode_dir)

        os.makedirs(self.checkpoint_mode_dir, exist_ok=True)

    def _save_config(self):
        cfg_path = os.path.join(self.checkpoint_mode_dir, 'config.yaml')
        with open(cfg_path, 'w') as f:
            draccus.dump(self, f)

    def __post_init__(self):
        if isinstance(self.seeds, int):
            self.seeds = [self.seeds + i * 171 for i in range(self.n_seeds)]

    def initialize_config(self):
        self._set_experiment_name()
        self._create_checkpoint_dir()
        self._save_config()
    
def normalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-5):
    return (array - mean) / (std + eps)

class Trainer(ABC):
    def __init__(self, cfg: TrainerConfig, envs: Optional[VectorEnv] = None):
        self.cfg = cfg
        self.cfg.initialize_config()

        if envs:
            self.envs = envs
        elif cfg.env_name:
            self.envs = gym.make_vec(self.cfg.env_name, num_envs=self.cfg.num_envs, vectorization_mode="sync")
        else:
            raise ValueError(f'No environment specified.')

        self.obs_dim = self.envs.single_observation_space.shape[0]
        self.action_dim = self.envs.single_action_space.shape[0]
        self.max_action = self.envs.single_action_space.high[0]

        self.agent: td3_bc.TD3BC_Base = None

        self.evaluator: Evaluator = None

        self.buffer: ReplayBuffer = None

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
    
    def _set_seed(self, seed: int):
        """
        Set the random seed for reproducibility.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.envs.reset(seed=seed)
    
    def _load_agent(self, seed: int):
        self.agent = td3_bc.get_td3_bc_agent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            train_steps=self.cfg.train_steps,
            cfg=self.cfg.train_mode.td3_config,
        )

        pretrain_path = None
        if self.cfg.pretrain_dir:
            if self.cfg.pretrain_checkpoint:
                pretrain_path = os.path.join(
                    self.cfg.pretrain_dir, f'seed_{seed}', f'checkpoint_{self.cfg.pretrain_checkpoint}'
                )
            else:
                pretrain_path = os.path.join(
                    self.cfg.pretrain_dir, f'seed_{seed}', 'checkpoint_latest'
                )

            if os.path.exists(pretrain_path):
                self.agent.load(pretrain_path)
                logging.info(f"Loaded pretrained agent from {pretrain_path}")
            else:
                raise ValueError(f'Invalid pretrain path: {pretrain_path}')
        else:
            logging.info("No pretraining directory specified, starting from scratch.")

    def train(self):
        for seed in self.cfg.seeds:
            logging.info(f"{'-' * 40}")
            logging.info(f"Starting training: Mode={self.cfg.train_mode.name} | Experiment: {self.cfg.experiment_name} | Seed={seed}")
            logging.info(f"Save run to {self.cfg.checkpoint_mode_dir}")
            logging.info(f"{'-' * 40}")
            
            group_name = f"{self.cfg.experiment_name}_{self.cfg.train_mode.name}"
            run_name = f"{group_name}_seed_{seed}"
            wandb.init(
                project=self.cfg.wandb_project,
                group=group_name,
                name=run_name,
                mode="disabled" if self.cfg.debug else "online",
                config=self.cfg,
            )

            self._set_seed(seed)

            self._load_agent(seed)

            self.initialize_replay_buffer()

            self.evaluator = Evaluator(
                self.envs,
                self.agent,
                n_eval_episodes=self.cfg.eval_episodes,
                dataset_statistics_path=self.cfg.dataset_statistics_path,
                render=False,
            )

            for i in tqdm(range(self.cfg.train_steps), desc="Training Steps"):
                batch = self.get_batch(self.cfg.batch_size)
                metrics = self.agent.train_step(batch)

                wandb.log(metrics, step=i)

                if (i + 1) % self.cfg.eval_freq == 0 or i == self.cfg.train_steps - 1 or i == 0:
                    eval_metrics = self.evaluator.evaluate()
                    wandb.log(eval_metrics, step=i)

                if (i + 1) % self.cfg.checkpoint_freq == 0 or i == self.cfg.train_steps - 1:
                    checkpoint_dir = os.path.join(self.cfg.checkpoint_mode_dir, f'seed_{seed}', f'checkpoint_{i+1}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.agent.save(checkpoint_dir)

                    latest_checkpoint = os.path.join(self.cfg.checkpoint_mode_dir, f'seed_{seed}', 'checkpoint_latest')
                    os.makedirs(latest_checkpoint, exist_ok=True)
                    self.agent.save(latest_checkpoint)

            wandb.finish()

class OfflineTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig, dataset: Optional[Dict] = None, envs: Optional[VectorEnv] = None):
        super().__init__(cfg, envs)

        self.dataset = dataset

    def _fill_replay_buffer(self):
        if self.dataset:
            self.buffer.convert_dict(self.dataset)
            logging.info(f"Replay buffer filled with transitions from provided dict dataset.")
        elif self.cfg.dataset_path:
            if os.path.exists(self.cfg.dataset_path):
                raise NotImplementedError("load from file")
            else:     
                dataset = minari.load_dataset(self.cfg.dataset_path, download=True)
                self.buffer.convert_minari(dataset)
                logging.info(f"Replay buffer filled with transitions from Minari dataset at {self.cfg.dataset_path}.")
        else:
            raise ValueError(f"Dataset must be provided for offline training mode '{self.cfg.name}'.")

    def initialize_replay_buffer(self):
        self.buffer = ReplayBuffer(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self._fill_replay_buffer()

        obs_mean, obs_std = self.buffer.compute_dataset_statistics()
        self.buffer.set_dataset_statistics(obs_mean=obs_mean, obs_std=obs_std)

        self.buffer.save_statistics(self.cfg.dataset_statistics_path)

        logging.info(f"Observations normalized and dataset statistics saved to {self.cfg.experiment_dir}")

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.buffer.sample(batch_size)

class OnlineTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig, envs: Optional[VectorEnv] = None):
        super().__init__(cfg, envs)
        
        self.obs = np.zeros((self.envs.num_envs, self.obs_dim))
        self.episode_starts = np.ones(self.envs.num_envs, dtype=np.bool)

        self.obs_mean = np.zeros(self.obs_dim)
        self.obs_std = np.ones(self.obs_dim)

    def step_and_add(self) -> np.ndarray:

        norm_obs = normalize(self.obs, self.obs_mean, self.obs_std)

        actions = (
            self.agent.select_action(norm_obs) + 
            np.random.normal(0, self.cfg.train_mode.expl_noise, size=self.action_dim)
        ).clip(-self.max_action, self.max_action)

        next_obs, rewards, terminated, truncated, _ = self.envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        mask = ~self.episode_starts
        self.buffer.add(self.obs[mask], actions[mask], next_obs[mask], rewards[mask], dones[mask])

        self.episode_starts = dones
        self.obs = next_obs

        return dones

    def _fill_replay_buffer(self):
        logging.info(f'Filling replay buffer with {self.cfg.train_mode.warmup_steps} warmup steps.')
        n_envs = self.envs.num_envs

        self.obs, _ = self.envs.reset()
        self.episode_starts = np.zeros(n_envs, dtype=np.bool)

        for _ in range(0, self.cfg.train_mode.warmup_steps, n_envs):
            self.step_and_add()

        self.envs.reset()
        self.episode_starts = np.zeros(n_envs, dtype=np.bool)

    def initialize_replay_buffer(self):
        self.buffer = ReplayBuffer(obs_dim=self.obs_dim, action_dim=self.action_dim)

        if self.cfg.pretrain_dir.endswith('/'):
            self.cfg.pretrain_dir = self.cfg.pretrain_dir[:-1]

        pretrain_dataset_statistics_path = os.path.join(os.path.dirname(self.cfg.pretrain_dir), 'dataset_statistics.json')
        if not os.path.exists(self.cfg.dataset_statistics_path):
            shutil.copyfile(pretrain_dataset_statistics_path, self.cfg.dataset_statistics_path)
        self.buffer.load_statistics(self.cfg.dataset_statistics_path)

        self.obs_mean, self.obs_std = self.buffer.get_dataset_statistics()

        self._fill_replay_buffer()

        logging.info(f'Loaded dataset statistics from {self.cfg.dataset_statistics_path} and filled replay buffer with {self.cfg.train_mode.warmup_steps} warmup steps.')

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        self.step_and_add()

        return self.buffer.sample(batch_size)

def get_trainer(cfg: TrainerConfig, dataset: Optional[Dict] = None, envs: Optional[VectorEnv] = None) -> Trainer:
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