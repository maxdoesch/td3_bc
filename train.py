import draccus
import gymnasium as gym
import numpy as np

from src.td3_bc.trainer import get_trainer, TrainerConfig

@draccus.wrap()
def main(cfg: TrainerConfig):
    trainer = get_trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
