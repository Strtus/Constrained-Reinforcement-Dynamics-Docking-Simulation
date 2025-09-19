#!/usr/bin/env python3
"""
千帆卫星交会对接训练 - 简化版本
不依赖复杂的外部库，使用基础Python库
"""

import math
import random
import time
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class SimpleQianfanTrainer:
    """简化的千帆卫星训练器"""
    
    def __init__(self):
        self.spacecraft_mass = 150.0  # kg
        self.orbit_altitude = 500000  # m (500km)
        self.max_thrust = 0.5  # N per thruster
        self.num_thrusters = 8
        
        # 轨道参数
        self.orbital_period = 5676  # 秒
        self.orbital_velocity = 7612  # m/s
        
        # 训练参数
        self.episodes = 10  # 先测试10轮
        self.max_steps_per_episode = 500  # 减少到500步
        
        print("千帆卫星交会对接训练器初始化完成")
        print(f"   卫星质量: {self.spacecraft_mass} kg")
        print(f"   轨道高度: {self.orbit_altitude/1000} km")
        print(f"   推力器: {self.num_thrusters} x {self.max_thrust} N")
    
    def orbital_dynamics(self, position, velocity, thrust_vector):
        """Hill-Clohessy-Wiltshire轨道动力学简化版"""
        x, y, z = position
        vx, vy, vz = velocity
        fx, fy, fz = thrust_vector
        
        # 轨道角速度
        n = 2 * math.pi / self.orbital_period
        
        # Hill-Clohessy-Wiltshire方程
        ax = 3 * n**2 * x + 2 * n * vy + fx / self.spacecraft_mass
        ay = -2 * n * vx + fy / self.spacecraft_mass
        az = -n**2 * z + fz / self.spacecraft_mass
        
        return [ax, ay, az]
    
    def simulate_docking_scenario(self):
        """模拟对接场景"""
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
        
        print(f"\n开始对接任务")
        print(f"   初始距离: {math.sqrt(sum(p**2 for p in position)):.1f} m")
        
        dt = 1.0  # 时间步长 1秒
        
        for step in range(self.max_steps_per_episode):
            # 计算距离和速度
            distance = math.sqrt(sum((p - tp)**2 for p, tp in zip(position, target_position)))
            speed = math.sqrt(sum(v**2 for v in velocity))
            
            if distance < 1.0 and speed < 0.1:  # 成功对接（距离+速度条件）
                print(f"第{step}步成功对接！距离: {distance:.2f} m, 速度: {speed:.3f} m/s")
                return True, step
            
            if distance > 1000.0:  # 超出范围
                print(f"第{step}步超出范围，任务失败")
                return False, step
            
            # 改进的PD控制策略（比例+微分）
            kp = 0.05  # 位置增益（降低）
            kd = 0.3   # 速度增益（添加阻尼）
            
            thrust_vector = []
            for i in range(3):
                # PD控制：F = -kp*(x-x_target) - kd*v
                position_error = position[i] - target_position[i]
                thrust_i = -kp * position_error - kd * velocity[i]
                thrust_vector.append(thrust_i)
            
            # 限制推力
            max_total_thrust = self.num_thrusters * self.max_thrust
            thrust_magnitude = math.sqrt(sum(f**2 for f in thrust_vector))
            if thrust_magnitude > max_total_thrust:
                scale = max_total_thrust / thrust_magnitude
                thrust_vector = [f * scale for f in thrust_vector]
            
            # 更新动力学
            acceleration = self.orbital_dynamics(position, velocity, thrust_vector)
            
            # 数值积分
            for i in range(3):
                velocity[i] += acceleration[i] * dt
                position[i] += velocity[i] * dt
            
            # 每100步打印状态
            if step % 100 == 0:
                print(f"   第{step}步: 距离={distance:.1f}m, 速度={speed:.3f}m/s")
        
        print(f"达到最大步数，任务超时")
        return False, self.max_steps_per_episode
    
    def train(self):
        """执行训练"""
        print(f"\n开始千帆卫星交会对接训练")
        print(f"   总训练轮数: {self.episodes}")
        
        successful_episodes = 0
        total_steps = 0
        
        start_time = time.time()
        
        for episode in range(self.episodes):
            print(f"\nEpisode {episode + 1}/{self.episodes}")
            
            success, steps = self.simulate_docking_scenario()
            total_steps += steps
            
            if success:
                successful_episodes += 1
            
            # 计算成功率
            success_rate = successful_episodes / (episode + 1) * 100
            avg_steps = total_steps / (episode + 1)
            
            print(f"   成功率: {success_rate:.1f}% ({successful_episodes}/{episode + 1})")
            print(f"   平均步数: {avg_steps:.1f}")
            
            # 每10轮总结一次
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"\n阶段总结 (前{episode + 1}轮):")
                print(f"   成功率: {success_rate:.1f}%")
                print(f"   平均完成时间: {avg_steps:.1f} 秒")
                print(f"   训练用时: {elapsed_time:.1f} 秒")
        
        # 最终总结
        final_time = time.time() - start_time
        print(f"\n千帆卫星训练完成！")
        print(f"   最终成功率: {successful_episodes/self.episodes*100:.1f}%")
        print(f"   成功任务: {successful_episodes}/{self.episodes}")
        print(f"   总训练时间: {final_time:.1f} 秒")
        print(f"   平均对接时间: {total_steps/self.episodes:.1f} 秒")

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