import os
import yaml
import draccus
import gymnasium as gym
from typing import Dict
from dataclasses import dataclass

from src.td3_bc.trainer import TrainerConfig
from src.td3_bc.evaluator import Evaluator
import src.td3_bc.td3_bc as td3_bc

@dataclass
class EvalConfig:
    checkpoint_mode_path: str
    checkpoint_step: int
    n_eval_episodes: int = 10
    num_envs: int = 1
    render: bool = False

    def __post_init__(self):
        config_path = os.path.join(self.checkpoint_mode_path, 'config.yaml')
        with open(config_path, 'r') as f:
            self.trainer_config: TrainerConfig = draccus.load(TrainerConfig, f)

    @property
    def checkpoint_path(self):
        return os.path.join(self.checkpoint_mode_path, f'checkpoint_{self.checkpoint_step}')
    
    @property
    def checkpoint_base_path(self):
        return os.path.dirname(self.checkpoint_mode_path)

@draccus.wrap()
def main(cfg: EvalConfig):

    envs = gym.make_vec(cfg.trainer_config.env_name, cfg.num_envs, vectorization_mode='sync', render_mode='human' if cfg.render else None)

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    max_action = envs.single_action_space.high[0]

    agent = td3_bc.get_td3_bc_agent(obs_dim=obs_dim, action_dim=action_dim, max_action=max_action, train_steps=cfg.trainer_config.train_steps, cfg=cfg.trainer_config.train_mode.td3_config)
    agent.load(cfg.checkpoint_path)

    dataset_statistics_path = os.path.join(cfg.checkpoint_base_path, 'dataset_statistics.json')

    evaluator = Evaluator(envs, agent, cfg.n_eval_episodes, dataset_statistics_path=dataset_statistics_path, render=cfg.render)
    metrics = evaluator.evaluate()

    for key, item in metrics.items():
        print(f'{key}: {item:.2f}')

if __name__ == "__main__":
    main()