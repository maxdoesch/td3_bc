import os
import re
import shutil
import json
import torch
from typing import Dict, Optional
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
class TrainerConfig(draccus.ChoiceRegistry):
    name: str

    train_steps: int = 200_000
    eval_freq: int = 100
    eval_episodes: int = 10
    batch_size: int = 256
    debug: bool = False
    seed: int = 0

    wandb_project: str = "pushing_offline_rl"

    checkpoint_path: str = "./checkpoints"
    experiment_name: Optional[str] = None


@TrainerConfig.register_subclass("pretrain")
@dataclass
class PretrainConfig(TrainerConfig):
    name: str = "pretrain"
    td3_config: td3_bc.TD3BC_Config = td3_bc.TD3BC_Config()


@TrainerConfig.register_subclass("refine")
@dataclass
class RefineConfig(TrainerConfig):
    name: str = "refine"
    td3_config: td3_bc.TD3BC_Refine_Config = td3_bc.TD3BC_Refine_Config()


@TrainerConfig.register_subclass("online")
@dataclass
class OnlineConfig(TrainerConfig):
    name: str = "online"
    td3_config: td3_bc.TD3BC_Online_Config = td3_bc.TD3BC_Online_Config()

    initial_samples: int = 5000
    expl_noise: float = 0.1


class Trainer(ABC):
    def __init__(self, cfg: TrainerConfig, envs: VectorEnv):
        self.envs = envs
        self.cfg = cfg

        state_dim = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.shape[0]
        max_action = self.envs.single_action_space.high[0]

        self.td3_bc_agent = td3_bc.get_td3_bc_agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            train_steps=cfg.train_steps,
            cfg=cfg.td3_config,
        )

        self.evaluator = Evaluator(envs, self.td3_bc_agent, self.cfg.eval_episodes)

        self.buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)

        self.experiment_name = self._get_experiment_name()
        self.checkpoint_dir = self._create_checkpoint_dir(self.experiment_name)

    def _get_experiment_name(self) -> str:
        if self.cfg.experiment_name:
            return self.cfg.experiment_name

        experiment_dirs = [
            d
            for d in os.listdir(self.cfg.checkpoint_path)
            if os.path.isdir(os.path.join(self.cfg.checkpoint_path, d)) and d.startswith("experiment_")
        ]

        ids = [int(m.group(1)) for d in experiment_dirs if (m := re.match(r"experiment_(\d+)", d))]
        return f"experiment_{max(ids, default=0) + 1}"

    def _create_checkpoint_dir(self, experiment_name: str) -> str:
        experiment_path = os.path.join(self.cfg.checkpoint_path, experiment_name)
        checkpoint_dir = os.path.join(experiment_path, self.cfg.name)

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
        print(f"Starting training: Mode={self.cfg.name} | Experiment: {self.experiment_name} | Seed={self.cfg.seed}")
        print(f"Save run to {self.checkpoint_dir}")
        print(f"{'-' * 40}")

        run_name = f"{self.experiment_name}_{self.cfg.name}"
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
            metrics = self.td3_bc_agent.train_step(batch)

            wandb.log(metrics, step=i)

            if (i + 1) % self.cfg.eval_freq == 0 or i == self.cfg.train_steps - 1:
                eval_metrics = self.evaluator.evaluate()
                wandb.log(eval_metrics, step=i)

        wandb.finish()


class OfflineTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig, env: VectorEnv, dataset: Dict):
        super().__init__(cfg, env)

        if not dataset:
            raise ValueError("Dataset must be provided for offline training.")

        self.buffer.convert_dict(dataset)

    def initialize_replay_buffer(self):
        state_mean, state_std = self.buffer.compute_dataset_statistics()

        dataset_statistics = {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
        }

        experiment_path = os.path.join(self.cfg.checkpoint_path, self.experiment_name)
        stats_path = os.path.join(experiment_path, "dataset_statistics.json")

        with open(stats_path, "w") as f:
            json.dump(dataset_statistics, f, indent=4)
        print(f"Dataset statistics saved to {stats_path}")

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.buffer.sample(batch_size)


class OnlineTrainer(Trainer):
    def __init__(self, cfg: TrainerConfig, env: VectorEnv):
        super().__init__(cfg, env)

    def initialize_replay_buffer(self):
        experiment_path = os.path.join(self.cfg.checkpoint_path, self.experiment_name)
        stats_path = os.path.join(experiment_path, "dataset_statistics.json")

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                dataset_statistics = json.load(f)
            state_mean = np.array(dataset_statistics["state_mean"])
            state_std = np.array(dataset_statistics["state_std"])
            self.buffer.normalize_states(state_mean, state_std)
        else:
            print(f"Dataset statistics not found at {stats_path}. Replay buffer will not be normalized.")

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        pass


def get_trainer(cfg: TrainerConfig, env: VectorEnv, dataset: Optional[Dict] = None) -> Trainer:
    if cfg.name == "pretrain":
        return OfflineTrainer(cfg, env, dataset)
    elif cfg.name == "refine":
        return OfflineTrainer(cfg, env, dataset)
    elif cfg.name == "online":
        return OnlineTrainer(cfg, env)
    else:
        raise ValueError(f"Unknown mode: {cfg.name}")


def main():
    episode_length = 100

    cfg = PretrainConfig()
    envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3, vectorization_mode="sync")
    episodes = 3
    episode_length = 1000
    dict_dataset = {
        "obs": [np.random.randn(episode_length + 1, envs.single_observation_space.shape[0]) for _ in range(episodes)],
        "acts": [np.random.randn(episode_length, envs.single_action_space.shape[0]) for _ in range(episodes)],
        "rews": [np.random.randn(episode_length) for _ in range(episodes)],
    }

    trainer = get_trainer(cfg, envs, dict_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
