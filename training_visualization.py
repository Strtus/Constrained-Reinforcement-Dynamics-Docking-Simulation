#!/usr/bin/env python3
"""
千帆卫星训练结果可视化
使用项目中的高质量可视化脚本展示训练成果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

# 设置科学期刊风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class TrainingVisualization:
    """千帆卫星训练结果可视化器"""
    
    def __init__(self):
        self.training_data = None
        self.load_training_data()
    
    def load_training_data(self):
        """加载训练数据"""
        try:
            with open('training_outputs/training_progress.json', 'r') as f:
                self.training_data = json.load(f)
            print(f"✅ 成功加载 {len(self.training_data)} 轮训练数据")
        except FileNotFoundError:
            print("❌ 未找到训练数据文件")
            return False
        return True
    
    def plot_training_progress(self):
        """绘制训练进度分析"""
        if not self.training_data:
            return
        
        # 提取数据
        episodes = []
        success_rates = []
        avg_steps = []
        final_distances = []
        final_velocities = []
        
        for episode_data in self.training_data:
            episodes.append(episode_data['episode'])
            success_rates.append(episode_data['success_rate'])
            avg_steps.append(episode_data['avg_steps'])
            final_distances.append(episode_data.get('final_distance', 0))
            final_velocities.append(episode_data.get('final_velocity', 0))
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # 成功率趋势
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, success_rates, 'b-', linewidth=2, alpha=0.8)
        ax1.axhline(y=95, color='g', linestyle='--', alpha=0.6, label='95% 目标')
        ax1.set_xlabel('训练轮次', fontweight='bold')
        ax1.set_ylabel('成功率 (%)', fontweight='bold')
        ax1.set_title('千帆卫星对接成功率演化', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 平均步数趋势
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(episodes, avg_steps, 'r-', linewidth=2, alpha=0.8)
        ax2.set_xlabel('训练轮次', fontweight='bold')
        ax2.set_ylabel('平均对接时间 (步数)', fontweight='bold')
        ax2.set_title('对接效率优化', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 最终距离分布
        ax3 = fig.add_subplot(gs[1, 0])
        final_distances_valid = [d for d in final_distances if d > 0]
        if final_distances_valid:
            ax3.hist(final_distances_valid, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(x=2.0, color='r', linestyle='--', linewidth=2, label='成功阈值 (2m)')
            ax3.set_xlabel('最终对接距离 (m)', fontweight='bold')
            ax3.set_ylabel('频次', fontweight='bold')
            ax3.set_title('对接精度分布', fontweight='bold', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 最终速度分布
        ax4 = fig.add_subplot(gs[1, 1])
        final_velocities_valid = [v for v in final_velocities if v > 0]
        if final_velocities_valid:
            ax4.hist(final_velocities_valid, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(x=0.2, color='r', linestyle='--', linewidth=2, label='安全阈值 (0.2m/s)')
            ax4.set_xlabel('最终对接速度 (m/s)', fontweight='bold')
            ax4.set_ylabel('频次', fontweight='bold')
            ax4.set_title('对接安全性分布', fontweight='bold', fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 学习曲线分析
        ax5 = fig.add_subplot(gs[2, :])
        # 计算滑动平均成功率
        window_size = 100
        if len(success_rates) >= window_size:
            moving_avg = []
            for i in range(window_size-1, len(success_rates)):
                avg = np.mean(success_rates[i-window_size+1:i+1])
                moving_avg.append(avg)
            
            ax5.plot(episodes[window_size-1:], moving_avg, 'purple', linewidth=3, 
                    label=f'{window_size}轮滑动平均', alpha=0.9)
            ax5.plot(episodes, success_rates, 'lightblue', linewidth=1, alpha=0.6, 
                    label='原始数据')
            
            # 标注学习阶段
            stage1_end = 4000
            stage2_end = 8000
            ax5.axvline(x=stage1_end, color='red', linestyle=':', alpha=0.7, label='阶段1结束')
            ax5.axvline(x=stage2_end, color='orange', linestyle=':', alpha=0.7, label='阶段2结束')
            
            ax5.set_xlabel('训练轮次', fontweight='bold')
            ax5.set_ylabel('成功率 (%)', fontweight='bold')
            ax5.set_title('千帆卫星学习曲线 - 课程学习效果分析', fontweight='bold', fontsize=14)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle('千帆卫星交会对接训练成果总结', fontsize=16, fontweight='bold')
        
        # 保存图表
        os.makedirs('analysis_results', exist_ok=True)
        plt.savefig('analysis_results/training_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("✅ 训练分析图表已保存: analysis_results/training_comprehensive_analysis.png")
        
        plt.show()
    
    def plot_performance_metrics(self):
        """绘制性能指标分析"""
        if not self.training_data:
            return
        
        # 按阶段分析
        stage1_data = [d for d in self.training_data if d['episode'] <= 4000]
        stage2_data = [d for d in self.training_data if 4000 < d['episode'] <= 8000]
        stage3_data = [d for d in self.training_data if d['episode'] > 8000]
        
        stages = ['阶段1\n(简单)', '阶段2\n(中等)', '阶段3\n(困难)']
        success_rates = []
        avg_steps = []
        
        for stage_data in [stage1_data, stage2_data, stage3_data]:
            if stage_data:
                final_success_rate = stage_data[-1]['success_rate']
                final_avg_steps = stage_data[-1]['avg_steps']
                success_rates.append(final_success_rate)
                avg_steps.append(final_avg_steps)
            else:
                success_rates.append(0)
                avg_steps.append(0)
        
        # 创建对比图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 各阶段成功率
        bars1 = ax1.bar(stages, success_rates, color=['lightblue', 'lightgreen', 'lightcoral'], 
                        alpha=0.8, edgecolor='black')
        ax1.set_ylabel('成功率 (%)', fontweight='bold')
        ax1.set_title('各训练阶段成功率', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 各阶段平均步数
        bars2 = ax2.bar(stages, avg_steps, color=['lightyellow', 'lightpink', 'lightgray'], 
                        alpha=0.8, edgecolor='black')
        ax2.set_ylabel('平均对接时间 (步数)', fontweight='bold')
        ax2.set_title('各训练阶段效率', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, steps in zip(bars2, avg_steps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                    f'{steps:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('千帆卫星课程学习效果对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('analysis_results/stage_performance_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("✅ 阶段对比图表已保存: analysis_results/stage_performance_comparison.png")
        
        plt.show()
    
    def generate_summary_report(self):
        """生成训练总结报告"""
        if not self.training_data:
            return
        
        print("\n" + "="*60)
        print("🛰️  千帆卫星交会对接训练总结报告")
        print("="*60)
        
        # 总体统计
        total_episodes = len(self.training_data)
        final_success_rate = self.training_data[-1]['success_rate']
        final_avg_steps = self.training_data[-1]['avg_steps']
        
        print(f"📊 训练规模:")
        print(f"   总训练轮次: {total_episodes:,}")
        print(f"   最终成功率: {final_success_rate:.1f}%")
        print(f"   平均对接时间: {final_avg_steps:.1f} 步")
        
        # 各阶段分析
        stage1_episodes = [d for d in self.training_data if d['episode'] <= 4000]
        stage2_episodes = [d for d in self.training_data if 4000 < d['episode'] <= 8000]
        stage3_episodes = [d for d in self.training_data if d['episode'] > 8000]
        
        print(f"\n🎯 课程学习效果:")
        if stage1_episodes:
            stage1_final = stage1_episodes[-1]
            print(f"   阶段1 (简单): 成功率 {stage1_final['success_rate']:.1f}%")
        
        if stage2_episodes:
            stage2_final = stage2_episodes[-1]
            print(f"   阶段2 (中等): 成功率 {stage2_final['success_rate']:.1f}%")
        
        if stage3_episodes:
            stage3_final = stage3_episodes[-1]
            print(f"   阶段3 (困难): 成功率 {stage3_final['success_rate']:.1f}%")
        
        # 性能指标
        final_distances = [d.get('final_distance', 0) for d in self.training_data[-1000:] if d.get('final_distance', 0) > 0]
        final_velocities = [d.get('final_velocity', 0) for d in self.training_data[-1000:] if d.get('final_velocity', 0) > 0]
        
        if final_distances and final_velocities:
            avg_distance = np.mean(final_distances)
            avg_velocity = np.mean(final_velocities)
            
            print(f"\n🎯 对接精度 (最后1000轮):")
            print(f"   平均对接距离: {avg_distance:.3f} m")
            print(f"   平均对接速度: {avg_velocity:.3f} m/s")
            
            safe_landings = sum(1 for v in final_velocities if v <= 0.2)
            print(f"   安全对接率: {safe_landings/len(final_velocities)*100:.1f}%")
        
        print(f"\n✅ 训练状态: 完成")
        print(f"📈 算法收敛: {'是' if final_success_rate >= 95 else '否'}")
        print("="*60)

def main():
    """主函数"""
    print("🛰️ 千帆卫星训练结果可视化分析")
    print("使用项目中的高质量可视化技术生成顶刊水准图表")
    
    viz = TrainingVisualization()
    
    if viz.training_data:
        print("\n正在生成训练进度分析...")
        viz.plot_training_progress()
        
        print("\n正在生成性能指标对比...")
        viz.plot_performance_metrics()
        
        print("\n正在生成总结报告...")
        viz.generate_summary_report()
        
        print("\n🎉 所有可视化分析完成！")
        print("📁 图表保存位置: analysis_results/")
        print("   - training_comprehensive_analysis.png")
        print("   - stage_performance_comparison.png")
    else:
        print("❌ 无法加载训练数据，请先运行训练程序")

if __name__ == "__main__":
    main()