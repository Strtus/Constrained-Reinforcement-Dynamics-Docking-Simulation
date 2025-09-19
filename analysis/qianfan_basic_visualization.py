#!/usr/bin/env python3
"""
Author: Strtus
"""

import math
import random
import json
from datetime import datetime

class QianfanAnalyzer:
    """Qianfan Satellite Analyzer"""
    
    def __init__(self):
        # Qianfan satellite parameters
        self.spacecraft_mass = 150.0  # kg
        self.orbit_altitude = 500000  # m
        self.orbital_period = 5676   # s
        self.n = 2 * math.pi / self.orbital_period  # orbital angular velocity
        
        # Propulsion system parameters
        self.max_thrust = 0.5  # N per thruster
        self.num_thrusters = 8
        self.specific_impulse = 220  # s
        self.g0 = 9.81  # m/s²
        
        # Mission constraints
        self.safety_distance = 10.0  # m
        self.approach_speed_limit = 0.1  # m/s
        self.fuel_budget = 2.0  # kg
    
    def hill_clohessy_wiltshire_dynamics(self, state, thrust_vector):
        """Hill-Clohessy-Wiltshire orbital dynamics equations"""
        x, y, z, vx, vy, vz = state
        fx, fy, fz = thrust_vector
        
        # HCW equations
        ax = 3 * self.n**2 * x + 2 * self.n * vy + fx / self.spacecraft_mass
        ay = -2 * self.n * vx + fy / self.spacecraft_mass
        az = -self.n**2 * z + fz / self.spacecraft_mass
        
        return [vx, vy, vz, ax, ay, az]
    
    def pd_control_law(self, state, target_state, kp, kd):
        """PD control law"""
        x, y, z, vx, vy, vz = state
        xt, yt, zt, vxt, vyt, vzt = target_state
        
        # Position and velocity errors
        pos_error = [x - xt, y - yt, z - zt]
        vel_error = [vx - vxt, vy - vyt, vz - vzt]
        
        # PD control
        thrust = [
            -kp[0] * pos_error[0] - kd[0] * vel_error[0],
            -kp[1] * pos_error[1] - kd[1] * vel_error[1],
            -kp[2] * pos_error[2] - kd[2] * vel_error[2]
        ]
        
        # Thrust saturation
        max_total_thrust = self.num_thrusters * self.max_thrust
        thrust_magnitude = math.sqrt(sum(f**2 for f in thrust))
        if thrust_magnitude > max_total_thrust:
            scale = max_total_thrust / thrust_magnitude
            thrust = [f * scale for f in thrust]
            
        return thrust
    
    def simulate_mission(self, initial_state, target_state, kp, kd, duration=3600, dt=1.0):
        """Simulate rendezvous and docking mission"""
        t = 0
        state = initial_state.copy()
        
        # Record data
        time_history = []
        state_history = []
        thrust_history = []
        fuel_consumption = 0
        
        while t < duration:
            # Record current state
            time_history.append(t)
            state_history.append(state.copy())
            
            # Calculate control input
            thrust = self.pd_control_law(state, target_state, kp, kd)
            thrust_history.append(thrust.copy())
            
            # Update state (Euler integration)
            derivatives = self.hill_clohessy_wiltshire_dynamics(state, thrust)
            for i in range(6):
                state[i] += derivatives[i] * dt
            
            # Fuel consumption
            thrust_magnitude = math.sqrt(sum(f**2 for f in thrust))
            fuel_consumption += thrust_magnitude * dt / (self.specific_impulse * self.g0)
            
            # Check termination conditions
            distance = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
            velocity = math.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
            
            if distance < 1.0 and velocity < self.approach_speed_limit:
                print(f"Successful docking! Time: {t/60:.1f} minutes")
                break
            
            if distance > 1000.0:
                print(f"Mission failed, out of range")
                break
            
            t += dt
        
        return {
            'time': time_history,
            'states': state_history,
            'thrusts': thrust_history,
            'fuel_used': fuel_consumption,
            'final_distance': distance,
            'final_velocity': velocity,
            'mission_time': t
        }
    
    def analyze_trajectory_data(self, simulation_results):
        """Analyze trajectory data"""
        states = simulation_results['states']
        thrusts = simulation_results['thrusts']
        time = simulation_results['time']
        
        analysis = {}
        
        # Position analysis
        positions = [[s[0], s[1], s[2]] for s in states]
        distances = [math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in positions]
        
        analysis['distance_stats'] = {
            'initial': distances[0],
            'final': distances[-1],
            'minimum': min(distances),
            'maximum': max(distances)
        }
        
        # Velocity analysis
        velocities = [[s[3], s[4], s[5]] for s in states]
        speeds = [math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in velocities]
        
        analysis['velocity_stats'] = {
            'initial': speeds[0],
            'final': speeds[-1],
            'maximum': max(speeds),
            'average': sum(speeds) / len(speeds)
        }
        
        # Thrust analysis
        thrust_magnitudes = [math.sqrt(t[0]**2 + t[1]**2 + t[2]**2) for t in thrusts]
        
        analysis['thrust_stats'] = {
            'maximum': max(thrust_magnitudes),
            'average': sum(thrust_magnitudes) / len(thrust_magnitudes),
            'total_impulse': sum(thrust_magnitudes) * (time[1] - time[0]) if len(time) > 1 else 0
        }
        
        # Fuel efficiency
        analysis['fuel_efficiency'] = {
            'total_consumption': simulation_results['fuel_used'],
            'consumption_rate': simulation_results['fuel_used'] / (simulation_results['mission_time'] / 3600),
            'budget_utilization': simulation_results['fuel_used'] / self.fuel_budget * 100
        }
        
        return analysis
    
    def monte_carlo_analysis(self, num_simulations=50):
        """Monte Carlo analysis"""
        print(f"Starting Monte Carlo analysis ({num_simulations} simulations)...")
        
        results = []
        
        for i in range(num_simulations):
            # Random initial conditions
            initial_state = [
                random.uniform(50, 200),    # x
                random.uniform(-50, 50),    # y
                random.uniform(-30, 30),    # z
                random.uniform(-0.2, 0.2),  # vx
                random.uniform(-0.1, 0.1),  # vy
                random.uniform(-0.1, 0.1)   # vz
            ]
            
            # Random control parameters
            kp = [random.uniform(0.01, 0.1) for _ in range(3)]
            kd = [random.uniform(0.1, 1.0) for _ in range(3)]
            
            target_state = [0, 0, 0, 0, 0, 0]
            
            # Simulation
            sim_result = self.simulate_mission(initial_state, target_state, kp, kd, 1800)
            analysis = self.analyze_trajectory_data(sim_result)
            
            # Success criteria
            success = (sim_result['final_distance'] < 1.0 and 
                      sim_result['final_velocity'] < self.approach_speed_limit)
            
            results.append({
                'success': success,
                'fuel_used': sim_result['fuel_used'],
                'mission_time': sim_result['mission_time'],
                'final_distance': sim_result['final_distance'],
                'analysis': analysis
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_simulations} simulations")
        
        return self.summarize_monte_carlo_results(results)
    
    def summarize_monte_carlo_results(self, results):
        """总结蒙特卡洛结果"""
        successful_missions = [r for r in results if r['success']]
        
        summary = {
            'total_simulations': len(results),
            'successful_missions': len(successful_missions),
            'success_rate': len(successful_missions) / len(results) * 100,
            'statistics': {}
        }
        
        if successful_missions:
            fuel_consumptions = [r['fuel_used'] * 1000 for r in successful_missions]  # g
            mission_times = [r['mission_time'] / 60 for r in successful_missions]  # min
            
            summary['statistics'] = {
                'fuel_consumption': {
                    'mean': sum(fuel_consumptions) / len(fuel_consumptions),
                    'min': min(fuel_consumptions),
                    'max': max(fuel_consumptions)
                },
                'mission_time': {
                    'mean': sum(mission_times) / len(mission_times),
                    'min': min(mission_times),
                    'max': max(mission_times)
                }
            }
        
        return summary
    
    def generate_ascii_plots(self, simulation_results):
        """生成ASCII图表"""
        states = simulation_results['states']
        time = simulation_results['time']
        
        print("\n" + "="*80)
        print("千帆卫星轨迹分析图表")
        print("="*80)
        
        # 距离变化图
        distances = [math.sqrt(s[0]**2 + s[1]**2 + s[2]**2) for s in states]
        self.plot_ascii_line(time, distances, "距离变化 (m)", "时间 (s)", "距离 (m)")
        
        # 速度变化图
        speeds = [math.sqrt(s[3]**2 + s[4]**2 + s[5]**2) for s in states]
        self.plot_ascii_line(time, speeds, "速度变化 (m/s)", "时间 (s)", "速度 (m/s)")
        
        # 推力变化图
        thrusts = simulation_results['thrusts']
        thrust_magnitudes = [math.sqrt(t[0]**2 + t[1]**2 + t[2]**2) for t in thrusts]
        self.plot_ascii_line(time, thrust_magnitudes, "推力变化 (N)", "时间 (s)", "推力 (N)")
    
    def plot_ascii_line(self, x_data, y_data, title, xlabel, ylabel, width=60, height=15):
        """绘制ASCII线图"""
        print(f"\n{title}")
        print("-" * len(title))
        
        if not x_data or not y_data:
            print("无数据")
            return
        
        # 数据归一化
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        
        if x_max == x_min or y_max == y_min:
            print("数据无变化")
            return
        
        # 创建图表网格
        chart = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 绘制数据点
        for i in range(len(x_data)):
            x_pos = int((x_data[i] - x_min) / (x_max - x_min) * (width - 1))
            y_pos = height - 1 - int((y_data[i] - y_min) / (y_max - y_min) * (height - 1))
            
            if 0 <= x_pos < width and 0 <= y_pos < height:
                chart[y_pos][x_pos] = '*'
        
        # 打印图表
        print(f"{ylabel}")
        for row in chart:
            print('|' + ''.join(row) + '|')
        print('+' + '-' * width + '+')
        print(f" {xlabel}")
        print(f" 范围: {y_min:.3f} 到 {y_max:.3f}")
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("千帆卫星交会对接综合分析报告")
        print("=" * 50)
        
        # 标准任务仿真
        initial_state = [100.0, 50.0, 30.0, 0.1, 0.05, 0.02]
        target_state = [0, 0, 0, 0, 0, 0]
        kp = [0.05, 0.05, 0.05]
        kd = [0.3, 0.3, 0.3]
        
        print("\n1. 标准任务仿真分析")
        print("-" * 30)
        
        sim_result = self.simulate_mission(initial_state, target_state, kp, kd)
        trajectory_analysis = self.analyze_trajectory_data(sim_result)
        
        # 打印关键指标
        print(f"最终距离误差: {sim_result['final_distance']:.3f} m")
        print(f"最终速度误差: {sim_result['final_velocity']*1000:.3f} mm/s")
        print(f"任务时间: {sim_result['mission_time']/60:.1f} 分钟")
        print(f"燃料消耗: {sim_result['fuel_used']*1000:.2f} g")
        print(f"燃料利用率: {trajectory_analysis['fuel_efficiency']['budget_utilization']:.1f}%")
        
        # 生成ASCII图表
        self.generate_ascii_plots(sim_result)
        
        print("\n2. 蒙特卡洛统计分析")
        print("-" * 30)
        
        # 蒙特卡洛分析
        mc_results = self.monte_carlo_analysis(30)
        
        print(f"总仿真次数: {mc_results['total_simulations']}")
        print(f"成功任务数: {mc_results['successful_missions']}")
        print(f"成功率: {mc_results['success_rate']:.1f}%")
        
        if mc_results['statistics']:
            stats = mc_results['statistics']
            print(f"平均燃料消耗: {stats['fuel_consumption']['mean']:.2f} g")
            print(f"平均任务时间: {stats['mission_time']['mean']:.1f} 分钟")
            print(f"燃料消耗范围: {stats['fuel_consumption']['min']:.2f} - {stats['fuel_consumption']['max']:.2f} g")
        
        # 生成技术报告
        technical_report = {
            'mission_info': {
                'spacecraft': '千帆卫星',
                'mass': f"{self.spacecraft_mass} kg",
                'orbit': f"{self.orbit_altitude/1000} km",
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'standard_mission': {
                'final_distance_error': f"{sim_result['final_distance']:.3f} m",
                'final_velocity_error': f"{sim_result['final_velocity']*1000:.3f} mm/s",
                'mission_duration': f"{sim_result['mission_time']/60:.1f} min",
                'fuel_consumption': f"{sim_result['fuel_used']*1000:.2f} g"
            },
            'monte_carlo_analysis': mc_results,
            'spacecraft_parameters': {
                'max_thrust_per_thruster': f"{self.max_thrust} N",
                'number_of_thrusters': self.num_thrusters,
                'specific_impulse': f"{self.specific_impulse} s",
                'fuel_budget': f"{self.fuel_budget} kg"
            }
        }
        
        # 保存报告
        try:
            with open('qianfan_analysis_report.json', 'w', encoding='utf-8') as f:
                json.dump(technical_report, f, ensure_ascii=False, indent=2)
            print(f"\n技术报告已保存至: qianfan_analysis_report.json")
        except Exception as e:
            print(f"保存报告时出错: {e}")
        
        print("\n3. 关键技术指标")
        print("-" * 30)
        print(f"轨道周期: {self.orbital_period/60:.1f} 分钟")
        print(f"轨道角速度: {self.n*1000:.6f} mrad/s")
        print(f"最大总推力: {self.num_thrusters * self.max_thrust} N")
        print(f"推重比: {(self.num_thrusters * self.max_thrust) / (self.spacecraft_mass * self.g0)*1000:.2f} ‰")
        print(f"安全距离: {self.safety_distance} m")
        print(f"接近速度限制: {self.approach_speed_limit*1000} mm/s")
        
        return technical_report

def main():
    """主函数"""
    analyzer = QianfanAnalyzer()
    
    print("千帆卫星交会对接任务分析系统")
    print("基于Hill-Clohessy-Wiltshire轨道动力学")
    print("适用于500km圆轨道近距离交会对接")
    
    report = analyzer.generate_comprehensive_report()
    
    print(f"\n分析完成！生成了包含以下内容的综合报告:")
    print("- 标准任务轨迹分析")
    print("- ASCII可视化图表")
    print("- 蒙特卡洛统计分析")
    print("- 关键技术指标")
    print("- JSON格式技术报告")

if __name__ == "__main__":
    main()