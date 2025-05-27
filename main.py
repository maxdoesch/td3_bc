import draccus
import gymnasium as gym
import numpy as np

from config.train_config import TrainConfig
from src.td3_bc.trainer import get_trainer


@draccus.wrap()
def main(cfg: TrainConfig):
    envs = gym.make_vec(cfg.env_name, num_envs=cfg.num_envs, vectorization_mode="sync")

    episodes = 3
    episode_length = 1000
    dict_dataset = {
        "obs": [np.random.randn(episode_length + 1, envs.single_observation_space.shape[0]) for _ in range(episodes)],
        "acts": [np.random.randn(episode_length, envs.single_action_space.shape[0]) for _ in range(episodes)],
        "rews": [np.random.randn(episode_length) for _ in range(episodes)],
    }

    trainer = get_trainer(cfg.trainer_config, envs, dict_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
