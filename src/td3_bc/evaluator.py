import numpy as np
from typing import Dict, List, Optional
import gymnasium as gym
from gymnasium.vector import VectorEnv
from abc import ABC, abstractmethod

import src.td3_bc.td3_bc as td3_bc


class Metric(ABC):
    def __init__(self, n_envs: int) -> None:
        self.n_envs: int = n_envs
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, rewards: np.ndarray, dones: np.ndarray, infos: List[Dict]) -> None:
        pass

    @abstractmethod
    def on_episode_end(self, env_idx: int) -> None:
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        pass


class RewardAndLengthMetric(Metric):
    def reset(self) -> None:
        self.current_rewards: np.ndarray = np.zeros(self.n_envs)
        self.current_lengths: np.ndarray = np.zeros(self.n_envs, dtype=int)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def step(self, rewards: np.ndarray, dones: np.ndarray, infos: List[Dict]) -> None:
        self.current_rewards += rewards
        self.current_lengths += 1

    def on_episode_end(self, env_idx: int) -> None:
        self.episode_rewards.append(self.current_rewards[env_idx])
        self.episode_lengths.append(self.current_lengths[env_idx])

        self.current_rewards[env_idx] = 0
        self.current_lengths[env_idx] = 0

    def compute(self) -> Dict[str, float]:
        return {
            "eval/mean_reward": float(np.mean(self.episode_rewards)),
            "eval/std_reward": float(np.std(self.episode_rewards)),
            "eval/mean_length": float(np.mean(self.episode_lengths)),
            "eval/std_length": float(np.std(self.episode_lengths)),
        }


class Evaluator:
    def __init__(
        self,
        envs: VectorEnv,
        agent: td3_bc.BaseAgent,
        n_eval_episodes: int = 10,
        render: bool = False,
        metric: Optional[Metric] = None,
    ) -> None:
        self.envs: VectorEnv = envs
        self.agent: td3_bc.BaseAgent = agent
        self.n_eval_episodes: int = n_eval_episodes
        self.render: bool = render

        self.n_envs: int = self.envs.num_envs
        self.metric: Metric = metric or RewardAndLengthMetric(self.n_envs)

    def evaluate(self) -> Dict[str, float]:
        self.metric.reset()
        episode_counts: np.ndarray = np.zeros(self.n_envs, dtype=int)
        episode_targets: np.ndarray = np.array(
            [(self.n_eval_episodes + i) // self.n_envs for i in range(self.n_envs)],
            dtype=int,
        )

        observations, _ = self.envs.reset()

        while (episode_counts < episode_targets).any():
            actions: np.ndarray = self.agent.select_action(observations)
            new_observations, rewards, terminated, truncated, infos = self.envs.step(actions)

            dones: np.ndarray = np.logical_or(terminated, truncated)
            self.metric.step(rewards, dones, infos)

            for i in range(self.n_envs):
                if dones[i] and episode_counts[i] < episode_targets[i]:
                    episode_counts[i] += 1
                    self.metric.on_episode_end(i)

            observations = new_observations

            if self.render:
                self.envs.render()

        return self.metric.compute()


if __name__ == "__main__":
    render = False

    envs = gym.make_vec(
        "MountainCarContinuous-v0", num_envs=1, vectorization_mode="sync", render_mode="human" if render else None
    )

    agent = td3_bc.DummyAgent(
        envs.single_observation_space.shape[0],
        envs.single_action_space.shape[0],
        max_action=envs.single_action_space.high[0],
    )

    evaluator = Evaluator(envs, agent, render=render)

    metrics = evaluator.evaluate()
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
