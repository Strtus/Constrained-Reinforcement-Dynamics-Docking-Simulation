"""
GRPO智能体训练脚本
==================

使用GRPO算法训练航天器对接智能体，包含超参数配置、训练循环和性能评估。
"""

import os
import sys
import numpy as np
import ray
from ray import tune
from ray.rllib.env.env_context import EnvContext
import logging
import json
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SpacecraftDockingEnv, DockingConfig
from src.agent import GRPOAgent, GRPOConfig
from src.simulator import SpacecraftParameters, OrbitParameters

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_env(env_config: EnvContext = None) -> SpacecraftDockingEnv:
    """创建环境实例"""
    docking_config = DockingConfig(
        max_distance=100.0,
        docking_tolerance=0.1,
        attitude_tolerance=0.1,
        max_velocity=0.5,
        max_angular_velocity=0.1,
        fault_probability=0.2,
        max_episode_steps=1000
    )
    
    return SpacecraftDockingEnv(config=docking_config)


def setup_training_config() -> Dict[str, Any]:
    """设置训练配置"""
    
    # GRPO配置
    grpo_config = GRPOConfig(
        # 网络架构
        hidden_dims=[256, 256, 128],
        use_transformer=True,
        transformer_layers=2,
        attention_heads=8,
        
        # 训练超参数
        learning_rate=1e-4,
        batch_size=4096,
        train_batch_size=32768,
        gamma=0.995,
        lambda_gae=0.95,
        entropy_coeff=0.01,
        value_loss_coeff=0.5,
        
        # 安全约束参数
        safety_threshold=0.1,
        constraint_penalty=100.0,
        lagrange_multiplier_lr=1e-3,
        
        # GRPO特定参数
        guided_reward_coeff=0.3,
        safety_bonus_coeff=0.2,
        exploration_bonus_coeff=0.1,
        
        # 训练配置
        num_workers=4,
        num_envs_per_worker=8,
        training_steps=1_000_000,
        evaluation_episodes=100
    )
    
    return grpo_config


def train_agent():
    """训练GRPO智能体"""
    
    print("=" * 60)
    print("SafeRL - 航天器安全强化学习训练")
    print("=" * 60)
    
    # 初始化Ray
    ray.shutdown()  # 确保清理之前的会话
    ray.init(ignore_reinit_error=True)
    
    try:
        # 注册环境
        from ray.tune.registry import register_env
        register_env("SpacecraftDockingEnv", create_env)
        
        # 创建GRPO智能体
        config = setup_training_config()
        agent = GRPOAgent(config)
        
        # 设置算法
        env_config = {}
        agent.setup_algorithm(env_config)
        
        print(f"开始训练 - 目标步数: {config.training_steps:,}")
        print(f"并行工作器: {config.num_workers}")
        print(f"每个工作器环境数: {config.num_envs_per_worker}")
        
        # 训练循环
        best_reward = -np.inf
        checkpoint_dir = f"checkpoints/grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        training_results = []
        
        for iteration in range(1000):  # 最大迭代次数
            # 训练一步
            result = agent.train_step()
            
            # 记录结果
            episode_reward_mean = result.get('episode_reward_mean', 0)
            episode_len_mean = result.get('episode_len_mean', 0)
            training_iteration = result.get('training_iteration', iteration)
            
            # 安全指标
            safety_metrics = agent.get_safety_metrics()
            
            training_results.append({
                'iteration': iteration,
                'episode_reward_mean': episode_reward_mean,
                'episode_len_mean': episode_len_mean,
                'constraint_violations': safety_metrics['constraint_violations'],
                'fault_recoveries': safety_metrics['fault_recoveries']
            })
            
            # 打印进度
            if iteration % 10 == 0:
                print(f"迭代 {iteration:4d} | "
                      f"平均奖励: {episode_reward_mean:8.2f} | "
                      f"平均步数: {episode_len_mean:6.1f} | "
                      f"约束违反: {safety_metrics['constraint_violations']:4d} | "
                      f"故障恢复: {safety_metrics['fault_recoveries']:4d}")
            
            # 保存最佳模型
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                checkpoint_path = agent.save_checkpoint(
                    os.path.join(checkpoint_dir, f"best_model_iter_{iteration}")
                )
                print(f"新的最佳模型已保存: {checkpoint_path}")
            
            # 定期评估
            if iteration % 50 == 0 and iteration > 0:
                print("\n" + "-" * 40)
                print("执行评估...")
                
                eval_results = agent.evaluate(num_episodes=20)
                eval_reward = eval_results.get('evaluation', {}).get('episode_reward_mean', 0)
                safety_results = eval_results.get('safety_metrics', {})
                
                print(f"评估奖励: {eval_reward:.2f}")
                print(f"约束违反率: {safety_results.get('constraint_violation_rate', 0):.3f}")
                print(f"故障恢复率: {safety_results.get('fault_recovery_rate', 0):.3f}")
                print("-" * 40 + "\n")
                
                # 重置安全指标
                agent.reset_safety_metrics()
            
            # 检查收敛条件
            if len(training_results) >= 100:
                recent_rewards = [r['episode_reward_mean'] for r in training_results[-50:]]
                if np.std(recent_rewards) < 10 and np.mean(recent_rewards) > 800:
                    print(f"\n训练收敛！平均奖励: {np.mean(recent_rewards):.2f}")
                    break
        
        # 保存最终模型和训练结果
        final_checkpoint = agent.save_checkpoint(
            os.path.join(checkpoint_dir, "final_model")
        )
        
        # 保存训练结果
        results_file = os.path.join(checkpoint_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print(f"\n训练完成！")
        print(f"最终模型: {final_checkpoint}")
        print(f"训练结果: {results_file}")
        print(f"最佳奖励: {best_reward:.2f}")
        
        return agent, checkpoint_dir
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise
    finally:
        ray.shutdown()


def evaluate_trained_agent(checkpoint_path: str, num_episodes: int = 100):
    """评估训练好的智能体"""
    
    print("=" * 60)
    print("SafeRL - 智能体性能评估")
    print("=" * 60)
    
    # 初始化Ray
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    
    try:
        # 注册环境
        from ray.tune.registry import register_env
        register_env("SpacecraftDockingEnv", create_env)
        
        # 创建智能体并加载模型
        config = setup_training_config()
        agent = GRPOAgent(config)
        agent.setup_algorithm()
        agent.load_checkpoint(checkpoint_path)
        
        print(f"开始评估 - 测试回合数: {num_episodes}")
        
        # 创建测试环境
        test_env = create_env()
        
        # 评估指标
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        fuel_efficiency = []
        safety_violations = 0
        fault_recovery_count = 0
        
        for episode in range(num_episodes):
            obs, info = test_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_violations = 0
            initial_fuel = info.get('fuel_remaining', 100)
            
            done = False
            while not done:
                # 获取动作（确定性策略）
                action, _ = agent.get_action(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # 记录安全指标
                if info.get('constraint_violations', 0) > episode_violations:
                    episode_violations = info.get('constraint_violations', 0)
                
                if any(h < 1.0 for h in info.get('thruster_health', [1.0]*8)):
                    fault_recovery_count += 1
            
            # 记录回合结果
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 检查成功对接
            if info.get('is_docked', False):
                success_count += 1
            
            # 燃料效率
            fuel_used = initial_fuel - info.get('fuel_remaining', 0)
            fuel_efficiency.append(fuel_used)
            
            # 安全违反
            safety_violations += episode_violations
            
            if (episode + 1) % 10 == 0:
                print(f"已完成 {episode + 1:3d}/{num_episodes} 回合")
        
        # 计算评估结果
        results = {
            'num_episodes': num_episodes,
            'success_rate': success_count / num_episodes,
            'average_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'average_episode_length': np.mean(episode_lengths),
            'fuel_efficiency': np.mean(fuel_efficiency),
            'safety_violation_rate': safety_violations / num_episodes,
            'fault_recovery_rate': fault_recovery_count / num_episodes,
            'best_episode_reward': np.max(episode_rewards),
            'worst_episode_reward': np.min(episode_rewards)
        }
        
        # 打印评估结果
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"成功率:           {results['success_rate']:.1%}")
        print(f"平均奖励:         {results['average_reward']:.2f} ± {results['reward_std']:.2f}")
        print(f"平均回合长度:     {results['average_episode_length']:.1f} 步")
        print(f"平均燃料消耗:     {results['fuel_efficiency']:.2f} kg")
        print(f"安全违反率:       {results['safety_violation_rate']:.3f}")
        print(f"故障恢复率:       {results['fault_recovery_rate']:.3f}")
        print(f"最佳回合奖励:     {results['best_episode_reward']:.2f}")
        print(f"最差回合奖励:     {results['worst_episode_reward']:.2f}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SafeRL 训练和评估脚本")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train",
                       help="运行模式: train(训练) 或 evaluate(评估)")
    parser.add_argument("--checkpoint", type=str, 
                       help="评估模式下的模型检查点路径")
    parser.add_argument("--episodes", type=int, default=100,
                       help="评估回合数")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        trained_agent, checkpoint_dir = train_agent()
        print(f"\n训练完成！检查点目录: {checkpoint_dir}")
        
    elif args.mode == "evaluate":
        if not args.checkpoint:
            print("评估模式需要指定 --checkpoint 参数")
            sys.exit(1)
        
        if not os.path.exists(args.checkpoint):
            print(f"检查点路径不存在: {args.checkpoint}")
            sys.exit(1)
        
        results = evaluate_trained_agent(args.checkpoint, args.episodes)
        
        # 保存评估结果
        results_dir = os.path.dirname(args.checkpoint)
        results_file = os.path.join(results_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n评估结果已保存: {results_file}")