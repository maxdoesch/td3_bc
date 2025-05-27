from dataclasses import dataclass
from typing import Optional

import src.td3_bc.trainer as trainer


@dataclass
class TrainConfig:
    dataset_path: Optional[str] = None
    trainer_config: trainer.TrainerConfig = trainer.PretrainConfig()
    env_name: str = "MountainCarContinuous-v0"
    num_envs: int = 3
