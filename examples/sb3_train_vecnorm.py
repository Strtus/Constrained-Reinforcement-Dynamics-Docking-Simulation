"""
Stable-Baselines3 PPO training with VecNormalize on SpacecraftRvDEnvironment.

Features:
- Vectorized env with make_vec_env
- VecNormalize for observation and return normalization
- Periodic evaluation (deterministic)
- Checkpoints and normalization stats saving

This is a lightweight but practical training harness.
"""

import os
import sys
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environment import SpacecraftRvDEnvironment  # noqa: E402


def make_env(max_steps: int = 600, enable_csv_logging: bool = False):
    return SpacecraftRvDEnvironment({
        "max_steps": max_steps,
        "enable_csv_logging": enable_csv_logging,
    })


def train(
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    n_steps: int = 1024,
    batch_size: int = 256,
    seed: int = 42,
    log_dir: str = "runs/ppo_vecnorm",
    device: str = "cpu",
    eval_freq: int = 10_000,
    save_freq: int = 20_000,
    max_steps: int = 600,
):
    os.makedirs(log_dir, exist_ok=True)

    # Vec env
    env = make_vec_env(lambda: make_env(max_steps=max_steps), n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Separate eval env (no reward normalization during eval)
    eval_env = make_vec_env(lambda: make_env(max_steps=max_steps), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    np.random.seed(seed)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        ent_coef=0.01,
        device=device,
        verbose=1,
    )

    # Callbacks: eval + checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=eval_freq // n_envs,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=os.path.join(log_dir, "ckpt"),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # Save model and VecNormalize stats
    model.save(os.path.join(log_dir, "final_model"))
    env.save(os.path.join(log_dir, "vecnormalize.pkl"))

    print(f"Training finished. Artifacts saved under: {log_dir}")


if __name__ == "__main__":
    # Default: quick-ish run to validate; adjust total_timesteps for longer training
    train(
        total_timesteps=50_000,
        n_envs=2,
        n_steps=512,
        batch_size=256,
        device="cpu",
        eval_freq=5_000,
        save_freq=10_000,
        max_steps=600,
    )
