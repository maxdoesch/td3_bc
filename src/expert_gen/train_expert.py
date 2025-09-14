import os
import argparse
import numpy as np
from typing import Dict
import gymnasium as gym
import wandb

# from sbx import PPO, TD3
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback

from expert_gen.hyperparameter import HYPERPARAMETERS
import dmc_env  # noqa: F401


def make_env(env_id, env_kwargs: Dict):
    return gym.make(env_id, obs_type="state", **env_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train an expert agent using PPO.")
    parser.add_argument("--env-id", type=str, default="dmc_cheetah_run_1-v1", help="Environment ID to train on.")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use (default: ppo).")
    parser.add_argument("--eval-envs", type=int, default=1, help="Number of evaluation environments.")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Evaluation frequency.")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Episodes per evaluation.")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="Checkpoint frequency.")
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/expert_models", help="Output directory for logs and models."
    )
    args = parser.parse_args()

    hparams = HYPERPARAMETERS[args.algorithm][args.env_id]

    run = wandb.init(
        project="dmc-expert-gen",
        name=f"{args.algorithm}-{args.env_id}",
        config={"env_id": args.env_id, "hyperparameters": hparams},
        sync_tensorboard=True,
        monitor_gym=True,
    )

    run_path = os.path.join(args.output_dir, f"run-{args.env_id}-{run.id}")

    # Training environment
    train_env = DummyVecEnv([lambda: make_env(args.env_id, hparams["env_kwargs"]) for _ in range(hparams["n_envs"])])
    train_env = VecMonitor(train_env, filename=os.path.join(run_path, "monitor.csv"))
    if hparams.get("normalize", False):
        train_env = VecNormalize(venv=train_env, gamma=hparams["gamma"], **hparams["normalize_kwargs"])

    # Evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(args.env_id, hparams["env_kwargs"]) for _ in range(args.eval_envs)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(run_path, "eval_monitor.csv"))
    if hparams.get("normalize", False):
        eval_env = VecNormalize(venv=eval_env, gamma=hparams["gamma"], **hparams["normalize_kwargs"])
        eval_env.training = False
        eval_env.norm_reward = False
        eval_env.obs_rms = train_env.obs_rms

    # Callbacks
    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(run_path, "best"),
                log_path=os.path.join(run_path, "eval_logs"),
                eval_freq=args.eval_freq // hparams["n_envs"],
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
            ),
            CheckpointCallback(
                save_freq=args.checkpoint_freq // hparams["n_envs"],
                save_path=os.path.join(run_path, "checkpoints"),
                save_vecnormalize=hparams.get("normalize", False),
            ),
            WandbCallback(gradient_save_freq=100),
        ]
    )

    # Model training
    if args.algorithm == "ppo":
        model = PPO(
            policy=hparams["policy"],
            env=train_env,
            learning_rate=hparams["learning_rate"],
            n_steps=hparams["n_steps"],
            batch_size=hparams["batch_size"],
            n_epochs=hparams["n_epochs"],
            gamma=hparams["gamma"],
            gae_lambda=hparams["gae_lambda"],
            ent_coef=hparams["ent_coef"],
            clip_range=hparams["clip_range"],
            max_grad_norm=hparams["max_grad_norm"],
            use_sde=hparams.get("use_sde", False),
            sde_sample_freq=hparams.get("sde_sample_freq", 4),
            policy_kwargs=hparams["policy_kwargs"],
            vf_coef=hparams["vf_coef"],
            verbose=1,
            tensorboard_log=run_path,
        )
    elif args.algorithm == "td3":
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3(
            policy=hparams["policy"],
            env=train_env,
            learning_starts=hparams["learning_starts"],
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=run_path,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    model.learn(
        total_timesteps=hparams["n_timesteps"],
        callback=callbacks,
    )

    train_env.close()
    eval_env.close()
    run.finish()


if __name__ == "__main__":
    main()
