import os
import argparse
import imageio
import numpy as np
import gymnasium as gym

from sbx import PPO
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from minari import DataCollector, delete_dataset, list_local_datasets

import dmc_envs  # noqa: F401

# Metadata
CODE_PERMALINK = "https://github.com/maxdoesch/td3_bc"
AUTHOR = "Maximilian Doesch"
AUTHOR_EMAIL = "doesch.maximilian@gmail.com"

# Skill level thresholds
SKILL_LEVEL = {"expert": 1.0, "medium": 0.4, "simple": 0.2}


class GetStateFromInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        _, info = env.reset()
        state = info["state"]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=state.shape, dtype=state.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["pixels"] = obs
        return info["state"], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["pixels"] = obs
        return info["state"], reward, terminated, truncated, info


def generate_expert_dataset(env_id: str, total_steps: int, expert_path: str, skill_level: str, save_to_gif: bool):
    # Determine checkpoint
    checkpoints_dir = os.path.join(expert_path, "checkpoints")
    checkpoints = os.listdir(checkpoints_dir)
    last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
    training_steps = int(last_checkpoint.split("_")[2].split(".")[0])
    checkpoint = int(SKILL_LEVEL.get(skill_level, 0.95) * training_steps)

    checkpoint_path = os.path.join(checkpoints_dir, f"ppo_model_{checkpoint}_steps.zip")
    vecnorm_path = os.path.join(expert_path, "vecnormalize_checkpoints", f"vecnormalize_step_{checkpoint}.pkl")

    # Build env
    env = gym.make(env_id, obs_type="pixels", height=96, width=96)

    env_dc = DataCollector(
        env,
        record_infos=False,
    )

    env = GetStateFromInfo(env_dc)

    if os.path.exists(vecnorm_path):
        env = NormalizeObservation(env)
        env.update_running_mean = False
        env.obs_rms = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env])).obs_rms
    else:
        print("Warning: VecNormalize file not found, proceeding without normalization.")

    # Load agent
    model = PPO.load(checkpoint_path)

    # Handle dataset naming and duplication
    dataset_id = f"dmc_distraction/{env_id}/{skill_level}-v0"
    if dataset_id in list_local_datasets():
        delete_dataset(dataset_id)

    # Rollout
    obs, _ = env.reset()
    observations = []
    cumulative_reward = 0.0
    cumulative_rewards = []
    episode_count = 0

    if save_to_gif:
        gif_dir = os.path.join(expert_path, "rollout")
        os.makedirs(gif_dir, exist_ok=True)

    for _ in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        cumulative_reward += reward
        if save_to_gif:
            observations.append(info["pixels"].copy())

        if terminated or truncated:
            cumulative_rewards.append(cumulative_reward)
            cumulative_reward = 0.0
            obs, _ = env.reset()

            if save_to_gif and observations:
                gif_path = os.path.join(gif_dir, f"episode_{skill_level}_{episode_count}.gif")
                imageio.mimsave(gif_path, observations, fps=30)
                observations = []

            episode_count += 1

    # Final stats and dataset save
    print(f"[{skill_level.upper()}] Episodes: {episode_count}, Avg Reward: {np.mean(cumulative_rewards):.2f}")

    env_dc.create_dataset(
        dataset_id=dataset_id,
        eval_env=gym.make(env_id, obs_type="pixels"),
        algorithm_name="ppo",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        code_permalink=CODE_PERMALINK,
        description=f"Expert dataset for {env_id} at skill level {skill_level}",
    )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a Minari expert dataset from a PPO-trained agent.")
    parser.add_argument("--env-id", type=str, default="dmc_distraction_cheetah_run_1-v1")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--expert-path", type=str, default="checkpoints/expert_models")
    parser.add_argument("--save-to-gif", action="store_true")
    args = parser.parse_args()

    for level in SKILL_LEVEL:
        print(f"--- Generating dataset for skill level: {level} ---")
        generate_expert_dataset(
            env_id=args.env_id,
            total_steps=args.total_steps,
            expert_path=args.expert_path,
            skill_level=level,
            save_to_gif=args.save_to_gif,
        )


if __name__ == "__main__":
    main()
