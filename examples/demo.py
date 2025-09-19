"""
快速演示脚本
============

展示SafeRL框架的基本使用方法，包括环境创建、智能体初始化和简单训练。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SpacecraftDockingEnv, DockingConfig
from src.simulator import SpacecraftSimulator, SpacecraftParameters
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_environment():
    """演示环境基本功能"""
    print("=" * 50)
    print("SafeRL 环境演示")
    print("=" * 50)
    
    # 创建环境
    config = DockingConfig(
        max_distance=50.0,
        docking_tolerance=0.1,
        fault_probability=0.3,  # 增加故障概率用于演示
        max_episode_steps=200
    )
    
    env = SpacecraftDockingEnv(config)
    
    print(f"状态空间维度: {env.observation_space.shape}")
    print(f"动作空间维度: {env.action_space.shape}")
    print(f"动作空间范围: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    
    # 运行一个回合
    obs, info = env.reset()
    print(f"\n初始距离: {info['distance_to_target']:.2f}m")
    print(f"初始燃料: {info['fuel_remaining']:.1f}kg")
    
    episode_rewards = []
    distances = []
    fuel_levels = []
    
    for step in range(100):
        # 简单的PD控制器作为基线
        distance = info['distance_to_target']
        velocity = info['relative_velocity']
        
        # 比例-微分控制
        kp, kd = 0.1, 0.5
        thrust_magnitude = kp * distance - kd * velocity
        thrust_magnitude = np.clip(thrust_magnitude, -1.0, 1.0)
        
        # 构造动作：向目标方向推进
        action = np.array([thrust_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_rewards.append(reward)
        distances.append(info['distance_to_target'])
        fuel_levels.append(info['fuel_remaining'])
        
        # 打印关键步骤
        if step % 20 == 0:
            print(f"步骤 {step:3d}: 距离={distance:6.2f}m, 速度={velocity:6.3f}m/s, "
                  f"奖励={reward:7.2f}, 燃料={info['fuel_remaining']:5.1f}kg")
        
        if terminated or truncated:
            break
    
    print(f"\n回合结束:")
    print(f"总奖励: {sum(episode_rewards):.2f}")
    print(f"最终距离: {distances[-1]:.2f}m")
    print(f"剩余燃料: {fuel_levels[-1]:.1f}kg")
    print(f"对接成功: {'是' if info.get('is_docked', False) else '否'}")
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    steps = range(len(episode_rewards))
    
    axes[0, 0].plot(steps, episode_rewards)
    axes[0, 0].set_title('瞬时奖励')
    axes[0, 0].set_xlabel('步数')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(steps, distances)
    axes[0, 1].set_title('距离目标')
    axes[0, 1].set_xlabel('步数')
    axes[0, 1].set_ylabel('距离 (m)')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(steps, fuel_levels)
    axes[1, 0].set_title('燃料消耗')
    axes[1, 0].set_xlabel('步数')
    axes[1, 0].set_ylabel('剩余燃料 (kg)')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(steps, np.cumsum(episode_rewards))
    axes[1, 1].set_title('累积奖励')
    axes[1, 1].set_xlabel('步数')
    axes[1, 1].set_ylabel('累积奖励')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print(f"\n结果图表已保存: demo_results.png")
    
    return env


def demo_simulator():
    """演示物理仿真器"""
    print("\n" + "=" * 50)
    print("SafeRL 仿真器演示")
    print("=" * 50)
    
    # 创建仿真器
    simulator = SpacecraftSimulator()
    
    print(f"轨道高度: {simulator.orbit.altitude/1000:.1f}km")
    print(f"航天器质量: {simulator.spacecraft.mass:.0f}kg")
    
    # 仿真一段轨道
    states = []
    controls = []
    
    for i in range(100):
        # 小推力机动
        control = np.array([1.0, 0.0, 0.0, 0.1, 0.0, 0.0])  # 小推力和力矩
        
        state, info = simulator.step(control, dt=1.0)
        states.append(state.copy())
        controls.append(control.copy())
        
        if i % 20 == 0:
            orbital_elements = info['orbital_elements']
            print(f"时间 {i:3d}s: 高度={orbital_elements['altitude']/1000:.2f}km, "
                  f"偏心率={orbital_elements['eccentricity']:.6f}")
    
    # 分析轨道变化
    states = np.array(states)
    positions = states[:, 0:3]
    velocities = states[:, 3:6]
    
    altitudes = [np.linalg.norm(pos) - simulator.physics.EARTH_RADIUS for pos in positions]
    speeds = [np.linalg.norm(vel) for vel in velocities]
    
    print(f"\n轨道机动结果:")
    print(f"初始高度: {altitudes[0]/1000:.2f}km")
    print(f"最终高度: {altitudes[-1]/1000:.2f}km")
    print(f"高度变化: {(altitudes[-1] - altitudes[0])/1000:.2f}km")
    print(f"初始速度: {speeds[0]:.1f}m/s")
    print(f"最终速度: {speeds[-1]:.1f}m/s")
    
    # 可视化轨道
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    time_points = np.arange(len(altitudes))
    
    axes[0].plot(time_points, np.array(altitudes)/1000)
    axes[0].set_title('轨道高度变化')
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('高度 (km)')
    axes[0].grid(True)
    
    axes[1].plot(time_points, speeds)
    axes[1].set_title('轨道速度变化')
    axes[1].set_xlabel('时间 (s)')
    axes[1].set_ylabel('速度 (m/s)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('simulator_demo.png', dpi=150, bbox_inches='tight')
    print(f"仿真结果图表已保存: simulator_demo.png")
    
    return simulator


def demo_fault_injection():
    """演示故障注入功能"""
    print("\n" + "=" * 50)
    print("SafeRL 故障注入演示")
    print("=" * 50)
    
    # 创建环境
    config = DockingConfig(fault_probability=0.8)  # 高故障概率
    env = SpacecraftDockingEnv(config)
    
    obs, info = env.reset()
    
    fault_history = []
    performance_metrics = []
    
    for step in range(50):
        # 随机动作
        action = env.action_space.sample() * 0.1  # 小幅动作
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录推进器健康状态
        thruster_health = info.get('thruster_health', [1.0]*8)
        fault_history.append(thruster_health.copy())
        
        # 记录性能
        performance_metrics.append({
            'distance': info['distance_to_target'],
            'fuel': info['fuel_remaining'],
            'violations': info['constraint_violations']
        })
        
        # 检查故障
        faulty_thrusters = [i for i, h in enumerate(thruster_health) if h < 1.0]
        if faulty_thrusters and step % 10 == 0:
            print(f"步骤 {step}: 检测到推进器故障 {faulty_thrusters}, "
                  f"健康度: {[f'{h:.2f}' for i, h in enumerate(thruster_health) if i in faulty_thrusters]}")
        
        if terminated or truncated:
            break
    
    # 分析故障影响
    fault_history = np.array(fault_history)
    avg_health = np.mean(fault_history, axis=1)
    
    print(f"\n故障分析:")
    print(f"平均推进器健康度: {np.mean(avg_health):.3f}")
    print(f"故障发生频率: {np.sum(avg_health < 1.0) / len(avg_health):.1%}")
    print(f"最严重故障: {np.min(fault_history):.3f}")
    
    # 可视化故障影响
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    steps = range(len(fault_history))
    
    # 推进器健康度
    for i in range(8):
        axes[0, 0].plot(steps, fault_history[:, i], alpha=0.7, label=f'推进器{i}')
    axes[0, 0].set_title('推进器健康状态')
    axes[0, 0].set_xlabel('步数')
    axes[0, 0].set_ylabel('健康度')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True)
    
    # 平均健康度
    axes[0, 1].plot(steps, avg_health)
    axes[0, 1].set_title('平均推进器健康度')
    axes[0, 1].set_xlabel('步数')
    axes[0, 1].set_ylabel('平均健康度')
    axes[0, 1].grid(True)
    
    # 距离变化
    distances = [m['distance'] for m in performance_metrics]
    axes[1, 0].plot(steps, distances)
    axes[1, 0].set_title('距离目标（存在故障）')
    axes[1, 0].set_xlabel('步数')
    axes[1, 0].set_ylabel('距离 (m)')
    axes[1, 0].grid(True)
    
    # 燃料消耗
    fuel_levels = [m['fuel'] for m in performance_metrics]
    axes[1, 1].plot(steps, fuel_levels)
    axes[1, 1].set_title('燃料消耗（存在故障）')
    axes[1, 1].set_xlabel('步数')
    axes[1, 1].set_ylabel('剩余燃料 (kg)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('fault_demo.png', dpi=150, bbox_inches='tight')
    print(f"故障分析图表已保存: fault_demo.png")
    
    return fault_history, performance_metrics


def demo_safety_constraints():
    """演示安全约束功能"""
    print("\n" + "=" * 50)
    print("SafeRL 安全约束演示")
    print("=" * 50)
    
    # 创建两个环境：有约束和无约束
    config_safe = DockingConfig(fault_probability=0.1)
    config_unsafe = DockingConfig(fault_probability=0.1, max_velocity=10.0)  # 放宽速度限制
    
    env_safe = SpacecraftDockingEnv(config_safe)
    env_unsafe = SpacecraftDockingEnv(config_unsafe)
    
    def run_episode(env, name):
        obs, info = env.reset()
        violations = []
        distances = []
        velocities = []
        
        for step in range(100):
            # 激进的控制策略
            action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 最大推力
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            violations.append(info['constraint_violations'])
            distances.append(info['distance_to_target'])
            velocities.append(info['relative_velocity'])
            
            if terminated or truncated:
                break
        
        print(f"\n{name}环境结果:")
        print(f"约束违反次数: {violations[-1]}")
        print(f"最大速度: {max(velocities):.3f}m/s")
        print(f"最终距离: {distances[-1]:.2f}m")
        
        return violations, distances, velocities
    
    # 运行比较实验
    safe_violations, safe_distances, safe_velocities = run_episode(env_safe, "安全约束")
    unsafe_violations, unsafe_distances, unsafe_velocities = run_episode(env_unsafe, "无安全约束")
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    steps_safe = range(len(safe_distances))
    steps_unsafe = range(len(unsafe_distances))
    
    # 距离比较
    axes[0].plot(steps_safe, safe_distances, label='安全约束', linewidth=2)
    axes[0].plot(steps_unsafe, unsafe_distances, label='无安全约束', linewidth=2)
    axes[0].set_title('距离对比')
    axes[0].set_xlabel('步数')
    axes[0].set_ylabel('距离 (m)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 速度比较
    axes[1].plot(steps_safe, safe_velocities, label='安全约束', linewidth=2)
    axes[1].plot(steps_unsafe, unsafe_velocities, label='无安全约束', linewidth=2)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='安全速度限制')
    axes[1].set_title('速度对比')
    axes[1].set_xlabel('步数')
    axes[1].set_ylabel('速度 (m/s)')
    axes[1].legend()
    axes[1].grid(True)
    
    # 约束违反比较
    axes[2].plot(steps_safe, safe_violations, label='安全约束', linewidth=2)
    axes[2].plot(steps_unsafe, unsafe_violations, label='无安全约束', linewidth=2)
    axes[2].set_title('约束违反次数')
    axes[2].set_xlabel('步数')
    axes[2].set_ylabel('累积违反次数')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('safety_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n安全约束对比图表已保存: safety_demo.png")


if __name__ == "__main__":
    print("SafeRL - 航天器安全强化学习框架演示")
    print("=====================================")
    
    try:
        # 1. 环境演示
        env = demo_environment()
        
        # 2. 仿真器演示
        simulator = demo_simulator()
        
        # 3. 故障注入演示
        fault_history, performance_metrics = demo_fault_injection()
        
        # 4. 安全约束演示
        demo_safety_constraints()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("生成的文件:")
        print("- demo_results.png: 环境基本功能演示")
        print("- simulator_demo.png: 物理仿真器演示")
        print("- fault_demo.png: 故障注入功能演示")
        print("- safety_demo.png: 安全约束功能演示")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()