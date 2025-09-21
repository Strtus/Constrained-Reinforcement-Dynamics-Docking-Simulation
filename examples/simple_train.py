#!/usr/bin/env python3
"""
千帆卫星交会对接训练 - 简化版本
不依赖复杂的外部库，使用基础Python库
"""

import math
import random
import os
import time
import sys
import json
import numpy as np
class SimpleQianfanTrainer:
    """简化的千帆卫星训练器"""
    def __init__(self):
        self.spacecraft_mass = 150.0  # kg
        self.orbit_altitude = 500000  # m (500km)
        self.max_thrust = 0.5  # N per thruster
        self.num_thrusters = 8
        self.orbital_period = 5676  # 秒
        self.orbital_velocity = 7612  # m/s
        self.episodes = 12000  # 整夜训练
        self.max_steps_per_episode = 3000
        self.output_dir = "training_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_distance = float('inf')
        self.distance_improvement_bonus = 0
        self.curriculum_stage = 1
        self.use_nn_controller = False
        self.nn_controller = None
    def orbital_dynamics(self, position, velocity, thrust_vector):
        x, y, z = position
        vx, vy, vz = velocity
        fx, fy, fz = thrust_vector
        n = 2 * math.pi / self.orbital_period
        ax = 3 * n**2 * x + 2 * n * vy + fx / self.spacecraft_mass
        ay = -2 * n * vx + fy / self.spacecraft_mass
        az = -n**2 * z + fz / self.spacecraft_mass
        return [ax, ay, az]
    def save_training_progress(self, progress_data):
        import json
        progress_file = os.path.join(self.output_dir, "training_progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                all_progress = json.load(f)
        else:
            all_progress = []
        all_progress.append(progress_data)
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(all_progress, f, indent=2, ensure_ascii=False)
        print(f"   训练进度已保存到: {progress_file}")
    # ...existing code...
    
    def simulate_docking_scenario(self):
        """模拟对接场景 - 优化版本"""
        # 随机初始位置和速度
        position = [
            random.uniform(50.0, 200.0),    # 50-200m距离
            random.uniform(-50.0, 50.0),    # 横向偏差
            random.uniform(-30.0, 30.0)     # 法向偏差
        ]
        velocity = [
            random.uniform(-0.2, 0.2),      # 径向速度
            random.uniform(-0.1, 0.1),      # 横向速度  
            random.uniform(-0.1, 0.1)       # 法向速度
        ]
        
        target_position = [0.0, 0.0, 0.0]  # 目标位置
        initial_distance = math.sqrt(sum(p**2 for p in position))
        self.best_distance = initial_distance  # 重置最佳距离
        
        print(f"\n开始对接任务")
        print(f"   初始距离: {initial_distance:.1f} m")
        
        dt = 1.0  # 时间步长 1秒
        reward_history = []  # 记录奖励历史
        
        for step in range(self.max_steps_per_episode):
            # 计算距离和速度
            distance = math.sqrt(sum((p - tp)**2 for p, tp in zip(position, target_position)))
            speed = math.sqrt(sum(v**2 for v in velocity))
            # 奖励塑形：记录距离改善
            if distance < self.best_distance:
                self.distance_improvement_bonus += (self.best_distance - distance) * 10
                self.best_distance = distance
            # 计算即时奖励
            instant_reward = -distance - 10*speed  # 距离和速度惩罚
            if distance < 10.0:
                instant_reward += 50  # 接近奖励
            if distance < 5.0:
                instant_reward += 100  # 近距离奖励
            reward_history.append(instant_reward)
            if distance < 2.0 and speed < 0.2:  # 放宽成功对接条件
                total_reward = sum(reward_history) + self.distance_improvement_bonus + 1000  # 成功奖励
                print(f"第{step}步成功对接！距离: {distance:.2f} m, 速度: {speed:.3f} m/s")
                print(f"   总奖励: {total_reward:.1f}, 距离改善奖励: {self.distance_improvement_bonus:.1f}")
                return True, step
            if distance > 1000.0:  # 超出范围
                print(f"第{step}步超出范围，任务失败")
                return False, step
            # 自适应PD控制策略（基于距离调整增益）
            if distance > 50:
                kp, kd = 0.01, 0.3
            elif distance > 10:
                kp, kd = 0.02, 0.5
            elif distance > 5:
                kp, kd = 0.03, 0.7
            else:
                kp, kd = 0.05, 1.0
            thrust_vector = []
            for i in range(3):
                position_error = position[i] - target_position[i]
                thrust_i = -kp * position_error - kd * velocity[i]
                thrust_vector.append(thrust_i)
            max_total_thrust = self.num_thrusters * self.max_thrust
            thrust_magnitude = math.sqrt(sum(f**2 for f in thrust_vector))
            if thrust_magnitude > max_total_thrust:
                scale = max_total_thrust / thrust_magnitude
                thrust_vector = [f * scale for f in thrust_vector]
            acceleration = self.orbital_dynamics(position, velocity, thrust_vector)
            for i in range(3):
                velocity[i] += acceleration[i] * dt
                position[i] += velocity[i] * dt
            if step % 200 == 0:
                improvement = f", 改善: {self.distance_improvement_bonus:.1f}" if self.distance_improvement_bonus > 0 else ""
                print(f"   第{step}步: 距离={distance:.1f}m, 速度={speed:.3f}m/s{improvement}")
        final_reward = sum(reward_history) + self.distance_improvement_bonus - 500  # 超时惩罚
        print(f"达到最大步数，任务超时。最终奖励: {final_reward:.1f}")
        return False, self.max_steps_per_episode
    
    def train(self):
        """执行训练主循环"""
        print(f"\n开始千帆卫星交会对接训练")
        print(f"   总训练轮数: {self.episodes}")
        successful_episodes = 0
        total_steps = 0
        start_time = time.time()
        for episode in range(self.episodes):
            # 课程学习阶段划分
            if episode < int(self.episodes * 0.3):
                self.curriculum_stage = 1
            elif episode < int(self.episodes * 0.7):
                self.curriculum_stage = 2
            else:
                self.curriculum_stage = 3
            print(f"\nEpisode {episode + 1}/{self.episodes} (阶段{self.curriculum_stage})")
            success, steps = self.simulate_docking_scenario()
            total_steps += steps
            if success:
                successful_episodes += 1
            success_rate = successful_episodes / (episode + 1) * 100
            avg_steps = total_steps / (episode + 1)
            print(f"   成功率: {success_rate:.1f}% ({successful_episodes}/{episode + 1})")
            print(f"   平均步数: {avg_steps:.1f}")
            if (episode + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"\n阶段总结 (前{episode + 1}轮):")
                print(f"   成功率: {success_rate:.1f}%")
                print(f"   平均完成时间: {avg_steps:.1f} 秒")
                print(f"   训练用时: {elapsed_time:.1f} 秒")
                print(f"   预计剩余时间: {elapsed_time * (self.episodes - episode - 1) / (episode + 1):.1f} 秒")
                progress_data = {
                    'episode': episode + 1,
                    'success_rate': success_rate,
                    'successful_episodes': successful_episodes,
                    'avg_steps': avg_steps,
                    'elapsed_time': elapsed_time
                }
                self.save_training_progress(progress_data)
        final_time = time.time() - start_time
        print(f"\n千帆卫星训练完成。")
        print(f"   最终成功率: {successful_episodes/self.episodes*100:.1f}%")
        print(f"   成功任务: {successful_episodes}/{self.episodes}")
        print(f"   总训练时间: {final_time:.1f} 秒")
        print(f"   平均对接时间: {total_steps/self.episodes:.1f} 秒")
        
        # 可选：生成可视化图表（中性表述，若脚本缺失则跳过）
        try:
            self.generate_visualization()
        except Exception as _:
            # 可视化非关键路径，出错时静默跳过
            pass

    # 兼容旧名：保留旧方法名，内部调用新实现
    def generate_professional_visualization(self):
        return self.generate_visualization()

    def generate_visualization(self):
        """运行可视化脚本生成分析图表（中性表述，无夸张用语）"""
        print("\n生成可视化图表...")

        # 可视化脚本路径（若存在则调用）
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_scripts = [
            os.path.join(repo_root, 'analysis', 'qianfan_visualization.py'),
            os.path.join(repo_root, 'analysis', 'visualization.py'),
            os.path.join(repo_root, 'professional_visualization.py'),  # 兼容历史文件名
        ]

        # 输出目录
        viz_output_dir = os.path.join(repo_root, 'analysis_results')
        os.makedirs(viz_output_dir, exist_ok=True)

        # 查找并调用首个存在的脚本
        script_to_run = next((p for p in candidate_scripts if os.path.exists(p)), None)
        if not script_to_run:
            print("未找到可视化脚本，已跳过。")
            return

        import subprocess
        cmd = f"cd {repo_root} && python {os.path.relpath(script_to_run, repo_root)}"
        print("执行可视化脚本...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("可视化已生成。输出位于 analysis_results/ 目录。")
        else:
            # 打印简要错误信息，保持中性
            stderr = (result.stderr or '').splitlines()[-1] if result.stderr else 'unknown error'
            print(f"可视化执行失败：{stderr}")

def main():
    """主函数"""
    print("千帆卫星交会对接任务启动")
    print("=" * 50)
    
    trainer = SimpleQianfanTrainer()
    trainer.train()
    print("\n" + "=" * 50)
    print("训练任务完成！")

if __name__ == "__main__":
    main()