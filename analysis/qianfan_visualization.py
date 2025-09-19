#!/usr/bin/env python3
"""
千帆卫星交会对接任务关键参数可视化分析
==========================================

生成顶级期刊水准的图表，展示：
1. 轨道动力学演化
2. 控制性能分析
3. 燃料消耗优化
4. 安全约束验证
5. 多场景对比分析

Author: Strtus
Date: 2025-09-19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import math
import json
from datetime import datetime

# Set scientific journal style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class QianfanVisualizationSuite:
    """Qianfan Satellite Visualization Analysis Suite"""
    
    def __init__(self):
        """Initialize Qianfan satellite parameters"""
        # Qianfan satellite parameters
        self.spacecraft_mass = 150.0  # kg
        self.orbit_altitude = 500000  # m
        self.orbital_period = 5676   # s
        self.n = 2 * np.pi / self.orbital_period  # orbital angular velocity
        
        # Propulsion system parameters
        self.max_thrust = 0.5  # N per thruster
        self.num_thrusters = 8
        self.specific_impulse = 220  # s (cold gas thrusters)
        self.g0 = 9.81  # m/s²
        
        # Mission constraints
        self.safety_distance = 10.0  # m
        self.approach_speed_limit = 0.1  # m/s
        self.fuel_budget = 2.0  # kg
        
        # Chart configuration
        self.fig_size = (16, 12)
        self.dpi = 300
        
    def hill_clohessy_wiltshire_dynamics(self, t, state, thrust_vector):
        """Hill-Clohessy-Wiltshire orbital dynamics equations"""
        x, y, z, vx, vy, vz = state
        fx, fy, fz = thrust_vector
        
        # HCW方程
        ax = 3 * self.n**2 * x + 2 * self.n * vy + fx / self.spacecraft_mass
        ay = -2 * self.n * vx + fy / self.spacecraft_mass
        az = -self.n**2 * z + fz / self.spacecraft_mass
        
        return [vx, vy, vz, ax, ay, az]
    
    def optimal_control_law(self, state, target_state, gains):
        """Optimal control law (LQR-based)"""
        kp_x, kp_y, kp_z = gains['position']
        kd_x, kd_y, kd_z = gains['velocity']
        
        x, y, z, vx, vy, vz = state
        xt, yt, zt, vxt, vyt, vzt = target_state
        
        # 位置和速度误差
        pos_error = np.array([x - xt, y - yt, z - zt])
        vel_error = np.array([vx - vxt, vy - vyt, vz - vzt])
        
        # PD控制
        thrust = -np.array([kp_x, kp_y, kp_z]) * pos_error - np.array([kd_x, kd_y, kd_z]) * vel_error
        
        # 推力限制
        max_total_thrust = self.num_thrusters * self.max_thrust
        thrust_magnitude = np.linalg.norm(thrust)
        if thrust_magnitude > max_total_thrust:
            thrust = thrust * (max_total_thrust / thrust_magnitude)
            
        return thrust
    
    def simulate_rendezvous_mission(self, initial_state, target_state, gains, duration=3600):
        """Simulate rendezvous and docking mission"""
        t_span = (0, duration)
        t_eval = np.linspace(0, duration, 3600)
        
        states = []
        thrusts = []
        fuel_consumption = 0
        
        def dynamics_with_control(t, state):
            thrust = self.optimal_control_law(state, target_state, gains)
            thrusts.append(thrust.copy())
            return self.hill_clohessy_wiltshire_dynamics(t, state, thrust)
        
        # 数值积分
        sol = solve_ivp(dynamics_with_control, t_span, initial_state, 
                       t_eval=t_eval, method='RK45', rtol=1e-8)
        
        return sol, np.array(thrusts)
    
    def plot_orbital_trajectory_3d(self, solution, thrusts, save_path=None):
        """Plot 3D orbital trajectory"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        t = solution.t / 60  # 转换为分钟
        
        # 轨迹颜色按时间变化
        scatter = ax.scatter(x, y, z, c=t, cmap='viridis', s=20, alpha=0.8)
        
        # Start and end point markers
        ax.scatter([x[0]], [y[0]], [z[0]], color='red', s=100, 
                  marker='o', label='Start Position')
        ax.scatter([0], [0], [0], color='gold', s=200, 
                  marker='*', label='Target Position')
        
        # 安全区域
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = self.safety_distance * np.outer(np.cos(u), np.sin(v))
        y_sphere = self.safety_distance * np.outer(np.sin(u), np.sin(v))
        z_sphere = self.safety_distance * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='orange')
        
        # Set coordinate axes
        ax.set_xlabel('Radial Distance (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tangential Distance (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Normal Distance (m)', fontsize=12, fontweight='bold')
        ax.set_title('Qianfan Satellite 3D Rendezvous Trajectory\n(Hill-Clohessy-Wiltshire Coordinate System)', 
                    fontsize=14, fontweight='bold')
        
        # Color bar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=30)
        cbar.set_label('Time (minutes)', fontsize=12, fontweight='bold')
        
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_control_performance_analysis(self, solution, thrusts, save_path=None):
        """Plot control performance analysis chart"""
        fig = plt.figure(figsize=self.fig_size)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        t = solution.t / 60  # minutes
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        vx, vy, vz = solution.y[3], solution.y[4], solution.y[5]
        
        # Distance evolution
        ax1 = fig.add_subplot(gs[0, 0])
        distance = np.sqrt(x**2 + y**2 + z**2)
        ax1.plot(t, distance, 'b-', linewidth=2, label='Actual Distance')
        ax1.axhline(y=self.safety_distance, color='orange', linestyle='--', 
                   linewidth=2, label='Safety Distance')
        ax1.axhline(y=1.0, color='red', linestyle='--', 
                   linewidth=2, label='Docking Threshold')
        ax1.set_xlabel('Time (minutes)', fontweight='bold')
        ax1.set_ylabel('Distance (m)', fontweight='bold')
        ax1.set_title('Distance Convergence Characteristics', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Velocity evolution
        ax2 = fig.add_subplot(gs[0, 1])
        velocity = np.sqrt(vx**2 + vy**2 + vz**2)
        ax2.plot(t, velocity * 1000, 'g-', linewidth=2, label='Relative Velocity')
        ax2.axhline(y=self.approach_speed_limit * 1000, color='red', 
                   linestyle='--', linewidth=2, label='Velocity Limit')
        ax2.set_xlabel('Time (minutes)', fontweight='bold')
        ax2.set_ylabel('Velocity (mm/s)', fontweight='bold')
        ax2.set_title('Velocity Control Characteristics', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Thrust history
        ax3 = fig.add_subplot(gs[1, :])
        thrust_magnitude = np.linalg.norm(thrusts, axis=1)
        thrust_x, thrust_y, thrust_z = thrusts[:, 0], thrusts[:, 1], thrusts[:, 2]
        
        ax3.plot(t, thrust_x * 1000, 'r-', linewidth=1.5, label='Radial Thrust', alpha=0.8)
        ax3.plot(t, thrust_y * 1000, 'g-', linewidth=1.5, label='Tangential Thrust', alpha=0.8)
        ax3.plot(t, thrust_z * 1000, 'b-', linewidth=1.5, label='Normal Thrust', alpha=0.8)
        ax3.plot(t, thrust_magnitude * 1000, 'k--', linewidth=2, label='Total Thrust')
        
        ax3.axhline(y=self.max_thrust * self.num_thrusters * 1000, 
                   color='red', linestyle=':', linewidth=2, label='Thrust Limit')
        ax3.set_xlabel('Time (minutes)', fontweight='bold')
        ax3.set_ylabel('Thrust (mN)', fontweight='bold')
        ax3.set_title('Thrust Control Output', fontweight='bold')
        ax3.legend(ncol=3)
        ax3.grid(True, alpha=0.3)
        
        # Fuel consumption analysis
        ax4 = fig.add_subplot(gs[2, 0])
        dt = np.diff(solution.t)
        thrust_dt = thrust_magnitude[:-1] * dt
        fuel_rate = thrust_dt / (self.specific_impulse * self.g0)
        cumulative_fuel = np.cumsum(np.concatenate([[0], fuel_rate])) * 1000  # g
        
        ax4.plot(t, cumulative_fuel, 'purple', linewidth=2)
        ax4.axhline(y=self.fuel_budget * 1000, color='red', linestyle='--', 
                   linewidth=2, label='Fuel Budget')
        ax4.set_xlabel('Time (minutes)', fontweight='bold')
        ax4.set_ylabel('Fuel Consumption (g)', fontweight='bold')
        ax4.set_title('Fuel Efficiency Analysis', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Control accuracy analysis
        ax5 = fig.add_subplot(gs[2, 1])
        position_error = distance
        velocity_error = velocity
        
        ax5.semilogy(t, position_error, 'b-', linewidth=2, label='Position Error')
        ax5_twin = ax5.twinx()
        ax5_twin.semilogy(t, velocity_error * 1000, 'r-', linewidth=2, label='Velocity Error')
        
        ax5.set_xlabel('Time (minutes)', fontweight='bold')
        ax5.set_ylabel('Position Error (m)', color='blue', fontweight='bold')
        ax5_twin.set_ylabel('Velocity Error (mm/s)', color='red', fontweight='bold')
        ax5.set_title('Control Accuracy Metrics', fontweight='bold')
        
        # Legend
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Qianfan Satellite Rendezvous and Docking Control Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_phase_portrait_analysis(self, solution, save_path=None):
        """Plot phase diagram analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        vx, vy, vz = solution.y[3], solution.y[4], solution.y[5]
        t = solution.t / 60
        
        # X-axis phase plot
        axes[0, 0].plot(x, vx, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].scatter([x[0]], [vx[0]], color='red', s=50, zorder=5, label='Start')
        axes[0, 0].scatter([0], [0], color='gold', s=100, marker='*', zorder=5, label='Target')
        axes[0, 0].set_xlabel('Radial Position (m)', fontweight='bold')
        axes[0, 0].set_ylabel('Radial Velocity (m/s)', fontweight='bold')
        axes[0, 0].set_title('Radial Motion Phase Portrait', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Y-axis phase plot
        axes[0, 1].plot(y, vy, 'g-', linewidth=2, alpha=0.8)
        axes[0, 1].scatter([y[0]], [vy[0]], color='red', s=50, zorder=5, label='Start')
        axes[0, 1].scatter([0], [0], color='gold', s=100, marker='*', zorder=5, label='Target')
        axes[0, 1].set_xlabel('Tangential Position (m)', fontweight='bold')
        axes[0, 1].set_ylabel('Tangential Velocity (m/s)', fontweight='bold')
        axes[0, 1].set_title('Tangential Motion Phase Portrait', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Z-axis phase plot
        axes[1, 0].plot(z, vz, 'purple', linewidth=2, alpha=0.8)
        axes[1, 0].scatter([z[0]], [vz[0]], color='red', s=50, zorder=5, label='Start')
        axes[1, 0].scatter([0], [0], color='gold', s=100, marker='*', zorder=5, label='Target')
        axes[1, 0].set_xlabel('Normal Position (m)', fontweight='bold')
        axes[1, 0].set_ylabel('Normal Velocity (m/s)', fontweight='bold')
        axes[1, 0].set_title('Normal Motion Phase Portrait', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Energy analysis
        kinetic_energy = 0.5 * self.spacecraft_mass * (vx**2 + vy**2 + vz**2)
        potential_energy = 0.5 * self.spacecraft_mass * self.n**2 * (3*x**2 - z**2)
        total_energy = kinetic_energy + potential_energy
        
        axes[1, 1].plot(t, kinetic_energy, 'r-', linewidth=2, label='Kinetic Energy')
        axes[1, 1].plot(t, potential_energy, 'b-', linewidth=2, label='Potential Energy')
        axes[1, 1].plot(t, total_energy, 'k--', linewidth=2, label='Total Energy')
        axes[1, 1].set_xlabel('Time (minutes)', fontweight='bold')
        axes[1, 1].set_ylabel('Energy (J)', fontweight='bold')
        axes[1, 1].set_title('System Energy Evolution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Qianfan Satellite Phase Space Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_monte_carlo_analysis(self, num_simulations=100, save_path=None):
        """Monte Carlo analysis: multi-scenario performance evaluation"""
        success_rates = []
        fuel_consumptions = []
        mission_times = []
        
        # Control gain ranges
        gain_ranges = {
            'position': [(0.01, 0.1), (0.01, 0.1), (0.01, 0.1)],
            'velocity': [(0.1, 1.0), (0.1, 1.0), (0.1, 1.0)]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for sim in range(num_simulations):
            # Random initial conditions
            initial_state = [
                np.random.uniform(50, 200),    # x
                np.random.uniform(-50, 50),    # y
                np.random.uniform(-30, 30),    # z
                np.random.uniform(-0.2, 0.2),  # vx
                np.random.uniform(-0.1, 0.1),  # vy
                np.random.uniform(-0.1, 0.1)   # vz
            ]
            
            # Random control gains
            gains = {
                'position': [np.random.uniform(r[0], r[1]) for r in gain_ranges['position']],
                'velocity': [np.random.uniform(r[0], r[1]) for r in gain_ranges['velocity']]
            }
            
            target_state = [0, 0, 0, 0, 0, 0]
            
            try:
                sol, thrusts = self.simulate_rendezvous_mission(
                    initial_state, target_state, gains, duration=1800)
                
                # Performance evaluation
                final_distance = np.sqrt(sol.y[0][-1]**2 + sol.y[1][-1]**2 + sol.y[2][-1]**2)
                success = final_distance < 1.0
                
                # Fuel consumption
                thrust_magnitude = np.linalg.norm(thrusts, axis=1)
                dt = np.diff(sol.t)
                fuel_used = np.sum(thrust_magnitude[:-1] * dt) / (self.specific_impulse * self.g0)
                
                success_rates.append(1 if success else 0)
                fuel_consumptions.append(fuel_used * 1000)  # g
                mission_times.append(sol.t[-1] / 60)  # min
                
            except:
                success_rates.append(0)
                fuel_consumptions.append(self.fuel_budget * 1000)
                mission_times.append(30)  # assumed failure time
        
        # Success rate distribution
        axes[0, 0].hist(success_rates, bins=2, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Mission Outcome', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title(f'Mission Success Rate: {np.mean(success_rates)*100:.1f}%', fontweight='bold')
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_xticklabels(['Failed', 'Success'])
        
        # Fuel consumption distribution
        axes[0, 1].hist(fuel_consumptions, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(x=self.fuel_budget * 1000, color='red', linestyle='--', 
                          linewidth=2, label='Budget Limit')
        axes[0, 1].set_xlabel('Fuel Consumption (g)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Fuel Consumption Distribution', fontweight='bold')
        axes[0, 1].legend()
        
        # Mission time distribution
        axes[1, 0].hist(mission_times, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Mission Time (minutes)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Mission Time Distribution', fontweight='bold')
        
        # Performance correlation
        successful_indices = [i for i, s in enumerate(success_rates) if s == 1]
        if successful_indices:
            success_fuel = [fuel_consumptions[i] for i in successful_indices]
            success_time = [mission_times[i] for i in successful_indices]
            
            axes[1, 1].scatter(success_fuel, success_time, alpha=0.6, color='blue')
            axes[1, 1].set_xlabel('Fuel Consumption (g)', fontweight='bold')
            axes[1, 1].set_ylabel('Mission Time (minutes)', fontweight='bold')
            axes[1, 1].set_title('Time-Fuel Relationship for Successful Missions', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Qianfan Satellite Monte Carlo Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
        # Return statistical results
        return {
            'success_rate': np.mean(success_rates),
            'avg_fuel_consumption': np.mean(fuel_consumptions),
            'avg_mission_time': np.mean(mission_times)
        }
    
    def generate_comprehensive_report(self, output_dir='./analysis_results/'):
        """Generate comprehensive analysis report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard mission scenario
        initial_state = [100.0, 50.0, 30.0, 0.1, 0.05, 0.02]
        target_state = [0, 0, 0, 0, 0, 0]
        gains = {
            'position': [0.05, 0.05, 0.05],
            'velocity': [0.3, 0.3, 0.3]
        }
        
        print("Generating Qianfan satellite comprehensive analysis report...")
        
        # Simulate standard mission
        sol, thrusts = self.simulate_rendezvous_mission(
            initial_state, target_state, gains, duration=3600)
        
        # Generate various charts
        self.plot_orbital_trajectory_3d(sol, thrusts, 
                                       os.path.join(output_dir, '3d_trajectory.png'))
        
        self.plot_control_performance_analysis(sol, thrusts,
                                             os.path.join(output_dir, 'control_performance.png'))
        
        self.plot_phase_portrait_analysis(sol,
                                         os.path.join(output_dir, 'phase_analysis.png'))
        
        mc_results = self.plot_monte_carlo_analysis(num_simulations=50,
                                                   save_path=os.path.join(output_dir, 'monte_carlo.png'))
        
        # Generate technical report
        report = self.generate_technical_report(sol, thrusts, mc_results)
        
        with open(os.path.join(output_dir, 'technical_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Analysis report generated at: {output_dir}")
        return report
    
    def generate_technical_report(self, solution, thrusts, mc_results):
        """生成技术报告数据"""
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        vx, vy, vz = solution.y[3], solution.y[4], solution.y[5]
        
        final_distance = np.sqrt(x[-1]**2 + y[-1]**2 + z[-1]**2)
        final_velocity = np.sqrt(vx[-1]**2 + vy[-1]**2 + vz[-1]**2)
        
        # 燃料消耗计算
        thrust_magnitude = np.linalg.norm(thrusts, axis=1)
        dt = np.diff(solution.t)
        total_fuel = np.sum(thrust_magnitude[:-1] * dt) / (self.specific_impulse * self.g0)
        
        report = {
            'mission_summary': {
                'spacecraft': '千帆卫星',
                'mass': f"{self.spacecraft_mass} kg",
                'orbit_altitude': f"{self.orbit_altitude/1000} km",
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_metrics': {
                'final_position_error': f"{final_distance:.3f} m",
                'final_velocity_error': f"{final_velocity*1000:.3f} mm/s",
                'mission_duration': f"{solution.t[-1]/60:.1f} min",
                'fuel_consumption': f"{total_fuel*1000:.2f} g",
                'fuel_efficiency': f"{total_fuel/self.fuel_budget*100:.1f}% of budget"
            },
            'monte_carlo_results': {
                'success_rate': f"{mc_results['success_rate']*100:.1f}%",
                'average_fuel_consumption': f"{mc_results['avg_fuel_consumption']:.2f} g",
                'average_mission_time': f"{mc_results['avg_mission_time']:.1f} min"
            },
            'technical_specifications': {
                'max_thrust_per_thruster': f"{self.max_thrust} N",
                'number_of_thrusters': self.num_thrusters,
                'specific_impulse': f"{self.specific_impulse} s",
                'safety_distance': f"{self.safety_distance} m",
                'approach_speed_limit': f"{self.approach_speed_limit*1000} mm/s"
            }
        }
        
        return report

def main():
    """Main function: generate complete visualization analysis"""
    viz = QianfanVisualizationSuite()
    
    print("Qianfan Satellite Rendezvous and Docking Mission Visualization Analysis")
    print("=" * 70)
    
    # Generate comprehensive report
    report = viz.generate_comprehensive_report()
    
    print("\nKey Performance Metrics:")
    for key, value in report['performance_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\nMonte Carlo Analysis Results:")
    for key, value in report['monte_carlo_results'].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()