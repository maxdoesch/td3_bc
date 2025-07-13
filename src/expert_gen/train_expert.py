import os
import wandb
import argparse
import gymnasium as gym
from sbx import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback

from expert_gen.hyperparameter import HYPERPARAMETERS
import dmc_envs


class VecNormalizeCallback(BaseCallback):
    def __init__(self, vecnormalize_env, save_path, save_freq, verbose=0):
        super().__init__(verbose)
        self.vecnormalize_env = vecnormalize_env
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"vecnormalize_step_{self.n_calls}.pkl")
            self.vecnormalize_env.save(path)
            if self.verbose > 0:
                print(f"Saved VecNormalize to {path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train an expert agent using PPO.")
    parser.add_argument("--env-id", type=str, default="dmc_cheetah_run_1-v1",
                        help="Environment ID to train on.")
    parser.add_argument("--eval-envs", type=int, default=1,
                        help="Number of evaluation environments.")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Frequency of evaluation during training.")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Number of episodes to evaluate during each evaluation.")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Frequency of saving model checkpoints.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/expert_models",)
    args = parser.parse_args()

    run = wandb.init(
        project="dmc-expert-gen",
        name=f"ppo-{args.env_id}",
        config={
            "env_id": args.env_id,
            "hyperparameters": HYPERPARAMETERS.get(args.env_id, {}),
        },
        sync_tensorboard=True,
        monitor_gym=True,
    )

    def make_env():
        env = gym.make(args.env_id, obs_type="state")
        return env

    run_path = os.path.join(args.output_dir, f'run-{args.env_id}-{run.id}')
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(os.path.join(run_path, "vecnormalize_checkpoints"), exist_ok=True)

    train_env = DummyVecEnv([make_env for _ in range(HYPERPARAMETERS[args.env_id]["n_envs"])])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True) if HYPERPARAMETERS[args.env_id]['normalize'] else train_env
    train_env = VecMonitor(train_env, f"{run_path}/monitor.csv")

    eval_env = DummyVecEnv([make_env for _ in range(args.eval_envs)])
    if HYPERPARAMETERS[args.env_id]['normalize']:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        eval_env.training = False
        eval_env.norm_reward = False
        eval_env.obs_rms = train_env.obs_rms
    eval_env = VecMonitor(eval_env, f"{run_path}/eval_monitor.csv")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{run_path}/best",
        log_path=f"{run_path}/eval_logs",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=f"{run_path}/checkpoints",
        name_prefix="ppo_model",
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
    )

    vecnormalize_callback = VecNormalizeCallback(
        vecnormalize_env=train_env,
        save_path=os.path.join(run_path, "vecnormalize_checkpoints"),
        save_freq=args.checkpoint_freq,
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        wandb_callback,
        vecnormalize_callback,
    ])

    model = PPO(
        policy=HYPERPARAMETERS[args.env_id]["policy"],
        env=train_env,
        learning_rate=HYPERPARAMETERS[args.env_id]["learning_rate"],
        n_steps=HYPERPARAMETERS[args.env_id]["n_steps"],
        batch_size=HYPERPARAMETERS[args.env_id]["batch_size"],
        n_epochs=HYPERPARAMETERS[args.env_id]["n_epochs"],
        gamma=HYPERPARAMETERS[args.env_id]["gamma"],
        gae_lambda=HYPERPARAMETERS[args.env_id]["gae_lambda"],
        ent_coef=HYPERPARAMETERS[args.env_id]["ent_coef"],
        clip_range=HYPERPARAMETERS[args.env_id]["clip_range"],
        max_grad_norm=HYPERPARAMETERS[args.env_id]["max_grad_norm"],
        policy_kwargs=HYPERPARAMETERS[args.env_id]["policy_kwargs"],
        vf_coef=HYPERPARAMETERS[args.env_id]["vf_coef"],
        verbose=1,
        tensorboard_log=run_path,
    )

    model.learn(
        total_timesteps=HYPERPARAMETERS[args.env_id]["n_timesteps"],
        callback=callbacks,
    )

    train_env.close()
    eval_env.close()

    run.finish()


if __name__ == "__main__":
    main()