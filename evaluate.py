import os
import torch
import numpy as np
import draccus
import gymnasium as gym
from dataclasses import dataclass
from typing import Union

from td3_bc.trainer import TrainerConfig
from td3_bc.evaluator import Evaluator
import td3_bc.td3_bc as td3_bc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_normalized_score(score: np.ndarray, ref_min_score: float, ref_max_score: float) -> np.ndarray:
    return (score - ref_min_score) / (ref_max_score - ref_min_score)


@dataclass
class EvalConfig:
    checkpoint_mode_path: str
    checkpoint_step: Union[int, str] = "latest"
    n_eval_episodes: int = 10
    num_envs: int = 1
    seed: int = 43
    render: bool = False

    def __post_init__(self):
        config_path = os.path.join(self.checkpoint_mode_path, "config.yaml")
        with open(config_path, "r") as f:
            self.trainer_config: TrainerConfig = draccus.load(TrainerConfig, f)

    @property
    def checkpoint_base_path(self):
        # remove trailing slash
        if self.checkpoint_mode_path.endswith("/"):
            self.checkpoint_mode_path = self.checkpoint_mode_path[:-1]
        return os.path.dirname(self.checkpoint_mode_path)


@draccus.wrap()
def main(cfg: EvalConfig):
    envs = gym.make_vec(
        cfg.trainer_config.env_name,
        cfg.num_envs,
        vectorization_mode="sync",
        render_mode="human" if cfg.render else None,
    )

    obs_shape = envs.single_observation_space.shape
    action_dim = envs.single_action_space.shape[0]
    max_action = envs.single_action_space.high[0]

    all_metrics = []

    for i, orig_seed in enumerate(cfg.trainer_config.seeds):
        seed = cfg.seed + i * 1034

        print(f"{'-' * 40}")
        print(f"Starting evaluation of checkpoint {cfg.checkpoint_mode_path} with seed {seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        envs.reset(seed=seed)

        agent = td3_bc.get_td3_bc_agent(
            obs_shape=obs_shape,
            action_dim=action_dim,
            max_action=max_action,
            train_steps=cfg.trainer_config.train_steps,
            cfg=cfg.trainer_config.train_mode.td3_config,
            device=device,
        )
        checkpoint_path = os.path.join(
            cfg.checkpoint_mode_path, f"seed_{orig_seed}", f"checkpoint_{cfg.checkpoint_step}"
        )
        agent.load(checkpoint_path)

        dataset_statistics_path = os.path.join(cfg.checkpoint_base_path, "dataset_statistics.json")
        evaluator = Evaluator(
            envs, agent, cfg.n_eval_episodes, dataset_statistics_path=dataset_statistics_path, render=cfg.render
        )

        metrics = evaluator.evaluate()
        all_metrics.append(metrics)

        print(f"Evaluation results for checkpoint {cfg.checkpoint_mode_path} with seed {seed}:")
        for key, item in metrics.items():
            print(f"{key}: {item:.2f}")
    # hopper
    ref_min_score = -20.272305
    ref_max_score = 3234.3

    # halfcheetah
    # ref_min_score = -280.178953
    # ref_max_score = 12135.0
    normalized_scores = np.array(
        [get_normalized_score(metric["eval/mean_reward"], ref_min_score, ref_max_score) for metric in all_metrics]
    )
    normalized_stds = np.array(
        [get_normalized_score(metric["eval/std_reward"], ref_min_score, ref_max_score) for metric in all_metrics]
    )

    mean_normalized_scores = np.mean(normalized_scores)

    print(normalized_stds)
    print((normalized_scores - mean_normalized_scores) ** 2)
    std_normalized_scores = np.sqrt(np.mean(normalized_stds**2 + (normalized_scores - mean_normalized_scores) ** 2))

    print(f"Mean normalized score: {mean_normalized_scores:.4f} ± {std_normalized_scores:.4f}")


if __name__ == "__main__":
    main()
