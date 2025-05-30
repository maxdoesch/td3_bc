import draccus
import gymnasium as gym
import numpy as np

from src.td3_bc.trainer import get_trainer, TrainerConfig
from misc.mountain_car_expert import HardcodedMountainCarPolicy, generate_expert_episodes

@draccus.wrap()
def main(cfg: TrainerConfig):
    envs = gym.make_vec(cfg.env_name, cfg.num_envs, vectorization_mode="sync")

    dict_dataset = generate_expert_episodes(envs, n_episodes=64)

    trainer = get_trainer(cfg, dict_dataset, envs)
    trainer.train()

if __name__ == "__main__":
    main()
