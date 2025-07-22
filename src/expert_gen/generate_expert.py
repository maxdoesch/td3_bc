import os
import argparse
import imageio
import numpy as np
import gymnasium as gym
import tqdm

from sbx import PPO
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from minari import DataCollector, list_local_datasets

import dmc_envs  # noqa: F401
from td3_bc.segmentation import MobileSAMV2Config
from td3_bc.wrappers import FTDObservationWrapper, FTDObservationWrapperConfig, ResizeObservation
import td3_bc.utils as utils

# Metadata
CODE_PERMALINK = "https://github.com/maxdoesch/td3_bc"
AUTHOR = "Maximilian Doesch"
AUTHOR_EMAIL = "doesch.maximilian@gmail.com"

# Skill level thresholds
SKILL_LEVEL = {"expert": 1.0, "medium": 0.4, "simple": 0.2}

RAW_IMG_RESOLUTION = 256


class CombineStackedFrames(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        C, H, W = env.observation_space.shape

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(H, W * (C // 3), 3), dtype=np.uint8)

    def observation(self, observation):
        observation = utils.combine_stacked_frames(observation)

        return observation


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


def generate_expert_dataset(
    env_id: str,
    dataset_id: str,
    image_size: int,
    gen_segmentation: bool,
    total_steps: int,
    expert_path: str,
    skill_level: str,
    save_to_gif: bool,
):
    # Determine checkpoint
    checkpoints_dir = os.path.join(expert_path, "checkpoints")
    checkpoints = os.listdir(checkpoints_dir)
    last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
    training_steps = int(last_checkpoint.split("_")[2].split(".")[0])
    checkpoint = int(SKILL_LEVEL.get(skill_level, 0.95) * training_steps)

    checkpoint_path = os.path.join(checkpoints_dir, f"ppo_model_{checkpoint}_steps.zip")
    vecnorm_path = os.path.join(expert_path, "vecnormalize_checkpoints", f"vecnormalize_step_{checkpoint}.pkl")

    # Build env
    env = gym.make(
        env_id,
        obs_type="pixels",
        height=RAW_IMG_RESOLUTION,
        width=RAW_IMG_RESOLUTION,
        channels_first=True if gen_segmentation else False,
        is_train=True,
    )

    if gen_segmentation:
        sam_config = MobileSAMV2Config(image_size=RAW_IMG_RESOLUTION, confidence_threshold=0.5)
        env_config = FTDObservationWrapperConfig(
            sam_config=sam_config,
            add_original_frame=True,
        )
        env = FTDObservationWrapper(env, config=env_config)
        env = ResizeObservation(env, shape=(image_size, image_size), is_channels_first=True)
        env = CombineStackedFrames(env)

    env_dc = DataCollector(env, record_infos=False, data_format="arrow")

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
    if dataset_id is None:
        dataset_id = f"dmc_distraction/{env_id}/{skill_level}-v0"
    if dataset_id in list_local_datasets():
        raise ValueError(f"Dataset ID '{dataset_id}' already exists. Please choose a different ID.")

    # Rollout
    obs, _ = env.reset()
    observations = []
    cumulative_reward = 0.0
    cumulative_rewards = []
    episode_count = 0

    if save_to_gif:
        gif_dir = os.path.join(expert_path, "rollout")
        os.makedirs(gif_dir, exist_ok=True)

    for _ in tqdm.tqdm(range(total_steps), desc=f"Generating dataset for {skill_level} skill level"):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"Reward: {reward}, Cumulative Reward: {cumulative_reward}")

        cumulative_reward += reward
        if save_to_gif:
            observations.append(info["pixels"].copy())

        if terminated or truncated:
            cumulative_rewards.append(cumulative_reward)
            cumulative_reward = 0.0
            obs, _ = env.reset()

            if save_to_gif and observations:
                gif_path = os.path.join(gif_dir, f"episode_{skill_level}_{episode_count}.gif")
                imageio.mimsave(gif_path, observations, fps=20)
                observations = []

            episode_count += 1

    if save_to_gif and observations:
        gif_path = os.path.join(gif_dir, f"episode_{skill_level}_{episode_count}.gif")
        imageio.mimsave(gif_path, observations, fps=20)
        observations = []

    # Final stats and dataset save
    print(f"[{skill_level.upper()}] Episodes: {episode_count}, Avg Reward: {np.mean(cumulative_rewards):.2f}")

    env_dc.create_dataset(
        dataset_id=dataset_id,
        eval_env=env_dc,
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
    parser.add_argument("--dataset-id", type=str)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--expert-path", type=str, default="checkpoints/expert_models")
    parser.add_argument("--gen-segmentation", action="store_true", help="Generate segmentation masks in the dataset.")
    parser.add_argument("--save-to-gif", action="store_true")
    args = parser.parse_args()

    for level in SKILL_LEVEL:
        print(f"--- Generating dataset for skill level: {level} ---")
        generate_expert_dataset(
            env_id=args.env_id,
            dataset_id=args.dataset_id,
            image_size=args.image_size,
            gen_segmentation=args.gen_segmentation,
            total_steps=args.total_steps,
            expert_path=args.expert_path,
            skill_level=level,
            save_to_gif=args.save_to_gif,
        )


if __name__ == "__main__":
    main()
