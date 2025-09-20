"""
Minimal smoke test for SpacecraftRvDEnvironment

Runs a short episode with random actions to verify imports and basic stepping.
"""

import os
import sys
import numpy as np

# Ensure src/ is on sys.path to import the environment module without loading package __init__
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environment import SpacecraftRvDEnvironment


def main():
    # Deterministic seed for reproducibility
    np.random.seed(42)

    env = SpacecraftRvDEnvironment({
        "enable_csv_logging": False,
        "max_steps": 200,  # short
    })

    obs, info = env.reset()
    print("Obs shape:", obs.shape)
    print("Action space:", env.action_space)

    total_reward = 0.0
    steps = 50
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if t % 10 == 0:
            print(f"t={t:03d} | dist={info['distance_to_target']:.1f} m | vel={info['relative_velocity_magnitude']:.3f} m/s | fuel={info['fuel_remaining']:.2f}")
        if terminated or truncated:
            print("Episode ended early:", info.get("termination_reason"))
            break

    env.render()
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()
