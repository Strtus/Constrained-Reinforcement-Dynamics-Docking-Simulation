"""
SafeRL - 航天器安全强化学习对接仿真框架
===============================================

一个专门用于航天器交会对接任务的安全强化学习框架，采用GRPO算法实现AI驱动的自主控制。

主要模块:
- environment: 6DOF航天器动力学环境
- agent: 基于GRPO的安全RL智能体
- simulator: 航天器物理仿真器
- visualization: 决策可视化和解释性工具
"""

__version__ = "1.0.0"
__author__ = "SafeRL Team"
__license__ = "MIT"

# Export the updated environment class; keep a backward-compatible alias
from .environment import SpacecraftRvDEnvironment as SpacecraftDockingEnv
from .environment import SpacecraftRvDEnvironment
from .agent import GRPOAgent
from .simulator import SpacecraftSimulator

__all__ = [
    "SpacecraftDockingEnv",
    "SpacecraftRvDEnvironment",
    "GRPOAgent", 
    "SpacecraftSimulator"
]