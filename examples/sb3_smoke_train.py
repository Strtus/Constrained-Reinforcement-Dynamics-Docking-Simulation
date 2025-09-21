"""
Stable-Baselines3 PPO smoke test on SpacecraftRvDEnvironment.

Runs a very short training to verify integration.
"""

import os
import sys

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from environment import SpacecraftRvDEnvironment


def make_env():
    # Keep episodes short and logging off for speed
    return SpacecraftRvDEnvironment({
        "max_steps": 200,
        "enable_csv_logging": False,
    })


def main():
    np.random.seed(0)

    # Vectorize with 1 env for simplicity
    env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=32,
        batch_size=32,
        learning_rate=3e-4,
        verbose=1,
        device="cpu",
    )

    # Train a tiny number of steps just to validate wiring
    model.learn(total_timesteps=256)

    # Brief evaluation rollout on a raw (non-vectorized) env for Gymnasium 5-tuple API
    eval_env = make_env()
    obs, _ = eval_env.reset()
    total_r = 0.0
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_r += float(reward)
        if terminated or truncated:
            break

    print("PPO smoke total reward:", total_r)


if __name__ == "__main__":
    main()
