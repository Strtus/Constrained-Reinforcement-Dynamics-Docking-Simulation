"""
GRPO smoke training on SpacecraftRvDEnvironment.

Uses in-repo GRPOAgent and TrainingPipeline for a very short run
to validate that the native algorithm wiring works end-to-end.
"""

import os
import sys

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from environment import SpacecraftRvDEnvironment  # noqa: E402
from agent import GRPOAgent  # noqa: E402
from training import TrainingPipeline, TrainingConfiguration  # noqa: E402


def main():
    # Short training to validate GRPO wiring
    cfg = TrainingConfiguration(
        total_episodes=5,
        evaluation_frequency=2,
        checkpoint_frequency=0,
        max_episode_steps=200,
        curriculum_learning=False,
        log_interval=1,
    )

    pipeline = TrainingPipeline(
        agent_class=GRPOAgent,
        environment_class=SpacecraftRvDEnvironment,
        config=cfg,
    )

    results = pipeline.train_agent(save_path=None)
    fm = results["final_metrics"]
    print(
        "GRPO smoke done | success_rate=%.3f, collision_rate=%.3f, fuel_mean=%.3f"
        % (fm.success_rate, fm.collision_rate, fm.fuel_consumption_mean)
    )


if __name__ == "__main__":
    main()
