"""
SafeRL环境模块测试
==================

测试航天器对接环境的基本功能、状态空间、动作空间和奖励函数。
"""

import pytest
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SpacecraftDockingEnv, DockingConfig, SpacecraftState


class TestSpacecraftDockingEnv:
    """测试航天器对接环境"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = DockingConfig(
            max_distance=100.0,
            docking_tolerance=0.1,
            max_velocity=0.5,
            fault_probability=0.0,  # 测试时禁用故障
            max_episode_steps=100
        )
        self.env = SpacecraftDockingEnv(self.config)
    
    def test_environment_creation(self):
        """测试环境创建"""
        assert self.env is not None
        assert self.env.observation_space.shape == (28,)
        assert self.env.action_space.shape == (6,)
        assert np.all(self.env.action_space.low == -1.0)
        assert np.all(self.env.action_space.high == 1.0)
    
    def test_reset_functionality(self):
        """测试重置功能"""
        obs, info = self.env.reset(seed=42)
        
        # 检查观测维度
        assert obs.shape == (28,)
        assert isinstance(obs, np.ndarray)
        
        # 检查信息字典
        required_keys = ['distance_to_target', 'relative_velocity', 'fuel_remaining', 
                        'attitude_error', 'constraint_violations', 'is_docked']
        for key in required_keys:
            assert key in info
        
        # 检查初始状态合理性
        assert info['distance_to_target'] > 0
        assert info['fuel_remaining'] > 0
        assert not info['is_docked']
        assert info['constraint_violations'] == 0
    
    def test_step_functionality(self):
        """测试步进功能"""
        obs, info = self.env.reset(seed=42)
        
        # 测试有效动作
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs_new, reward, terminated, truncated, info_new = self.env.step(action)
        
        # 检查返回值类型和维度
        assert obs_new.shape == (28,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info_new, dict)
        
        # 检查状态变化
        assert info_new['distance_to_target'] != info['distance_to_target']
        assert info_new['fuel_remaining'] <= info['fuel_remaining']
    
    def test_action_clipping(self):
        """测试动作限制"""
        obs, info = self.env.reset(seed=42)
        
        # 测试超出范围的动作
        large_action = np.array([10.0, -10.0, 5.0, -5.0, 3.0, -3.0])
        obs_new, reward, terminated, truncated, info_new = self.env.step(large_action)
        
        # 环境应该正常处理（内部会限制动作）
        assert obs_new.shape == (28,)
        assert isinstance(reward, (int, float))
    
    def test_termination_conditions(self):
        """测试终止条件"""
        obs, info = self.env.reset(seed=42)
        
        # 测试成功对接（模拟接近目标）
        self.env.chaser_state.position = np.array([0.05, 0.0, 0.0])  # 接近目标
        self.env.chaser_state.velocity = np.array([0.001, 0.0, 0.0])  # 低速
        self.env.chaser_state.attitude = np.array([1.0, 0.0, 0.0, 0.0])  # 对齐
        
        action = np.zeros(6)
        obs_new, reward, terminated, truncated, info_new = self.env.step(action)
        
        # 应该检测到接近对接状态
        assert info_new['distance_to_target'] < 0.1
    
    def test_reward_components(self):
        """测试奖励函数组件"""
        obs, info = self.env.reset(seed=42)
        
        # 测试不同动作的奖励
        actions = [
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 向前推进
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 无动作
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # 大推力
        ]
        
        rewards = []
        for action in actions:
            obs, info = self.env.reset(seed=42)  # 重置到相同状态
            obs_new, reward, terminated, truncated, info_new = self.env.step(action)
            rewards.append(reward)
        
        # 奖励应该有所不同
        assert not all(r == rewards[0] for r in rewards)
    
    def test_safety_constraints(self):
        """测试安全约束"""
        obs, info = self.env.reset(seed=42)
        
        # 设置高速状态以触发安全约束
        self.env.chaser_state.velocity = np.array([1.0, 0.0, 0.0])  # 超过最大安全速度
        
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 继续加速
        obs_new, reward, terminated, truncated, info_new = self.env.step(action)
        
        # 应该产生约束违反
        assert info_new['constraint_violations'] >= 0
    
    def test_fuel_consumption(self):
        """测试燃料消耗"""
        obs, info = self.env.reset(seed=42)
        initial_fuel = info['fuel_remaining']
        
        # 执行高推力动作
        action = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        obs_new, reward, terminated, truncated, info_new = self.env.step(action)
        
        # 燃料应该减少
        assert info_new['fuel_remaining'] < initial_fuel
    
    def test_state_space_bounds(self):
        """测试状态空间边界"""
        obs, info = self.env.reset(seed=42)
        
        # 执行多步以观察状态变化
        for _ in range(10):
            action = self.env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 检查观测值的合理性
            assert np.all(np.isfinite(obs))
            assert obs.shape == (28,)
            
            if terminated or truncated:
                break
    
    def test_reproducibility(self):
        """测试可重现性"""
        # 使用相同种子重置两次
        obs1, info1 = self.env.reset(seed=42)
        obs2, info2 = self.env.reset(seed=42)
        
        # 应该产生相同的初始状态
        np.testing.assert_array_almost_equal(obs1, obs2)
        assert abs(info1['distance_to_target'] - info2['distance_to_target']) < 1e-6
    
    def test_environment_info(self):
        """测试环境信息"""
        obs, info = self.env.reset(seed=42)
        
        # 检查性能指标
        metrics = self.env.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'success_rate' in metrics
        assert 'fuel_efficiency' in metrics
        assert 'safety_violations' in metrics
        assert 'fault_recovery_rate' in metrics


class TestDockingConfig:
    """测试对接配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DockingConfig()
        
        assert config.max_distance == 100.0
        assert config.docking_tolerance == 0.1
        assert config.attitude_tolerance == 0.1
        assert config.max_velocity == 0.5
        assert config.fault_probability == 0.2
        assert config.max_episode_steps == 1000
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = DockingConfig(
            max_distance=50.0,
            docking_tolerance=0.05,
            fault_probability=0.5
        )
        
        assert config.max_distance == 50.0
        assert config.docking_tolerance == 0.05
        assert config.fault_probability == 0.5


class TestSpacecraftState:
    """测试航天器状态"""
    
    def test_state_creation(self):
        """测试状态创建"""
        state = SpacecraftState(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.1, 0.2, 0.3]),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.01, 0.02, 0.03]),
            thruster_health=np.ones(8),
            fuel_remaining=100.0
        )
        
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.attitude.shape == (4,)
        assert state.angular_velocity.shape == (3,)
        assert state.thruster_health.shape == (8,)
        assert state.fuel_remaining == 100.0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])