import draccus
from functools import partial
import gymnasium as gym
from typing import List, Dict
import numpy as np
import wandb

from td3_bc.trainer import get_trainer, TrainerConfig
from td3_bc.evaluator import RewardAndLengthMetric

import dmc_env  # noqa: F401
from dmc_env.segmentation import MobileSAMV2Config
from dmc_env.wrappers import FTDObservationWrapper, FTDObservationWrapperConfig, ResizeObservation, FrameStack, ZoomObservationWrapper

class ObservationMetric(RewardAndLengthMetric):
    def __init__(self, fps: int = 30):
        super().__init__()
        self.fps = fps
        self.episode_obs: List[List[np.ndarray]] = []
        self.current_obs: List[List[np.ndarray]] = []

    def reset(self) -> None:
        super().reset()
        self.episode_obs = []
        self.current_obs = []

    def step(self, obs: np.ndarray, rewards: np.ndarray, dones: np.ndarray, infos: List[Dict]) -> None:
        super().step(obs, rewards, dones, infos)

        self.current_obs = [[] for _ in range(len(obs))] if len(self.current_obs) == 0 else self.current_obs

        for i, obs in enumerate(obs):
            self.current_obs[i].append(obs[0, -1])

    def on_episode_end(self, env_idx: int) -> None:
        super().on_episode_end(env_idx)

        frames = np.stack(self.current_obs[env_idx], axis=0)

        self.episode_obs.append(frames)

        self.current_obs[env_idx] = []

    def compute(self) -> Dict[str, float]:
        metrics = super().compute()

        if len(self.episode_obs) > 0:
            metrics["eval/episode"] = wandb.Video(self.episode_obs[0], fps=self.fps, format="mp4")

        return metrics


def make_vec(env_id: str, frame_stack: int, **env_kwargs):
    action_repeat = env_kwargs.pop("action_repeat", 1)
    env = gym.make(
        env_id,
        obs_type="pixels",
        channels_first=True,
        height=256,
        width=256,
        is_train=False,
        action_repeat=action_repeat,
        **env_kwargs,
    )
    env = ZoomObservationWrapper(env, scale=0.7, keep_size=True, channels_first=True)
    env_config = FTDObservationWrapperConfig(
        sam_config=MobileSAMV2Config(image_size=256, confidence_threshold=0.5),
        add_original_frame=True,
    )
    env = FTDObservationWrapper(env, config=env_config)
    env = ResizeObservation(env, shape=(64, 64), is_channels_first=True)
    env = FrameStack(env, k=frame_stack)
    return env


@draccus.wrap()
def main(cfg: TrainerConfig):
    envs = gym.vector.SyncVectorEnv(
        [
            partial(make_vec, env_id=cfg.env_name, frame_stack=cfg.frame_stack, **cfg.env_kwargs)
            for _ in range(cfg.num_envs)
        ]
    )

    trainer = get_trainer(cfg, envs=envs, eval_metric=ObservationMetric())
    trainer.train()


if __name__ == "__main__":
    main()
