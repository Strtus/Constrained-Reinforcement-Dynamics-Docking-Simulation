"""
SafeRL智能体测试
===============

测试GRPO智能体的训练、推理和安全约束功能。
"""

import pytest
import numpy as np
import torch
import sys
import os
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import GRPOAgent, GRPOConfig, SafetyTransformerModel


class TestGRPOConfig:
    """测试GRPO配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = GRPOConfig()
        
        assert config.hidden_dims == [256, 256, 128]
        assert config.learning_rate == 1e-4
        assert config.gamma == 0.995
        assert config.safety_threshold == 0.1
        assert config.num_workers == 4
        assert config.training_steps == 1_000_000
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = GRPOConfig(
            hidden_dims=[128, 64],
            learning_rate=1e-3,
            safety_threshold=0.05,
            num_workers=2
        )
        
        assert config.hidden_dims == [128, 64]
        assert config.learning_rate == 1e-3
        assert config.safety_threshold == 0.05
        assert config.num_workers == 2


class TestSafetyTransformerModel:
    """测试安全Transformer模型"""
    
    def setup_method(self):
        """设置测试"""
        self.obs_space = Mock()
        self.obs_space.shape = (28,)
        
        self.action_space = Mock()
        self.action_space.shape = (6,)
        
        self.config = GRPOConfig(
            hidden_dims=[64, 32],
            use_transformer=False  # 简化测试
        )
        
        self.model = SafetyTransformerModel(
            obs_space=self.obs_space,
            action_space=self.action_space,
            num_outputs=12,  # 6 actions * 2 (mean + log_std)
            model_config={},
            name="test_model",
            config=self.config
        )
    
    def test_model_creation(self):
        """测试模型创建"""
        assert self.model is not None
        assert self.model.obs_dim == 28
        assert self.model.action_dim == 6
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 4
        obs = torch.randn(batch_size, 28)
        
        input_dict = {"obs": obs}
        state = []
        seq_lens = torch.ones(batch_size)
        
        output, new_state = self.model.forward(input_dict, state, seq_lens)
        
        assert output.shape == (batch_size, 12)  # 6 actions * 2
        assert len(new_state) == len(state)
    
    def test_value_function(self):
        """测试价值函数"""
        batch_size = 4
        obs = torch.randn(batch_size, 28)
        
        input_dict = {"obs": obs}
        state = []
        seq_lens = torch.ones(batch_size)
        
        # 先调用forward
        output, new_state = self.model.forward(input_dict, state, seq_lens)
        
        # 然后调用value_function
        values = self.model.value_function()
        
        assert values.shape == (batch_size,)
    
    def test_safety_function(self):
        """测试安全函数"""
        batch_size = 4
        obs = torch.randn(batch_size, 28)
        
        input_dict = {"obs": obs}
        state = []
        seq_lens = torch.ones(batch_size)
        
        # 先调用forward
        output, new_state = self.model.forward(input_dict, state, seq_lens)
        
        # 然后调用safety_function
        safety_scores = self.model.safety_function()
        
        assert safety_scores.shape == (batch_size,)
        assert torch.all(safety_scores >= 0.0)
        assert torch.all(safety_scores <= 1.0)


class TestGRPOAgent:
    """测试GRPO智能体"""
    
    def setup_method(self):
        """设置测试"""
        self.config = GRPOConfig(
            hidden_dims=[64, 32],
            num_workers=1,  # 简化测试
            training_steps=1000
        )
        self.agent = GRPOAgent(self.config)
    
    def test_agent_creation(self):
        """测试智能体创建"""
        assert self.agent is not None
        assert self.agent.config == self.config
        assert self.agent.algorithm is None  # 未设置算法
    
    def test_safety_metrics_initialization(self):
        """测试安全指标初始化"""
        metrics = self.agent.get_safety_metrics()
        
        assert 'constraint_violations' in metrics
        assert 'safety_rewards' in metrics
        assert 'lagrange_multipliers' in metrics
        assert 'fault_recoveries' in metrics
        
        assert metrics['constraint_violations'] == 0
        assert metrics['safety_rewards'] == []
        assert metrics['lagrange_multipliers'] == []
        assert metrics['fault_recoveries'] == 0
    
    def test_guided_reward_shaping(self):
        """测试引导式奖励重塑"""
        original_reward = 10.0
        state = np.random.randn(28)
        action = np.random.randn(6)
        info = {
            'distance_to_target': 5.0,
            'relative_velocity': 0.1,
            'attitude_error': 0.05,
            'thruster_health': [1.0] * 8
        }
        
        guided_reward, components = self.agent.guided_reward_shaping(
            original_reward, state, action, info
        )
        
        assert isinstance(guided_reward, float)
        assert isinstance(components, dict)
        assert 'original' in components
        assert 'safety' in components
        assert 'exploration' in components
        assert components['original'] == original_reward
    
    def test_safety_score_computation(self):
        """测试安全得分计算"""
        state = np.random.randn(28)
        info = {
            'distance_to_target': 10.0,
            'relative_velocity': 0.05,
            'attitude_error': 0.01
        }
        
        safety_score = self.agent._compute_safety_score(state, info)
        
        assert isinstance(safety_score, float)
        assert 0.0 <= safety_score <= 1.0
    
    def test_constraint_penalty_computation(self):
        """测试约束惩罚计算"""
        state = np.random.randn(28)
        
        # 测试安全状态
        safe_info = {
            'relative_velocity': 0.1,
            'distance_to_target': 1.0,
            'attitude_error': 0.05
        }
        
        safe_penalty = self.agent._compute_constraint_penalty(state, safe_info)
        assert safe_penalty >= 0.0
        
        # 测试违反约束状态
        unsafe_info = {
            'relative_velocity': 1.0,  # 超过安全速度
            'distance_to_target': 0.01,  # 过于接近
            'attitude_error': 0.5  # 姿态偏差过大
        }
        
        unsafe_penalty = self.agent._compute_constraint_penalty(state, unsafe_info)
        assert unsafe_penalty > safe_penalty
    
    def test_fault_recovery_reward(self):
        """测试故障恢复奖励"""
        state = np.random.randn(28)
        action = np.random.randn(6)
        
        # 无故障情况
        normal_info = {
            'thruster_health': [1.0] * 8
        }
        
        normal_reward = self.agent._compute_fault_recovery_reward(state, action, normal_info)
        assert normal_reward == 0.0
        
        # 有故障情况
        fault_info = {
            'thruster_health': [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
        
        fault_reward = self.agent._compute_fault_recovery_reward(state, action, fault_info)
        assert fault_reward >= 0.0
    
    def test_planning_reward(self):
        """测试规划奖励"""
        state = np.random.randn(28)
        action = np.random.randn(6)
        
        info = {
            'distance_to_target': 10.0,
            'relative_velocity': 0.1
        }
        
        planning_reward = self.agent._compute_planning_reward(state, action, info)
        assert isinstance(planning_reward, float)
        assert planning_reward >= 0.0
    
    def test_lagrange_multiplier_update(self):
        """测试拉格朗日乘数更新"""
        # 模拟约束违反
        self.agent.safety_metrics['constraint_violations'] = 10
        self.agent.safety_metrics['safety_rewards'] = [1.0] * 5
        
        initial_lambda = float(self.agent.lambda_safety.data)
        self.agent._update_lagrange_multipliers()
        updated_lambda = float(self.agent.lambda_safety.data)
        
        # 约束违反率高时应该增加乘数
        assert updated_lambda >= initial_lambda
    
    def test_safety_metrics_reset(self):
        """测试安全指标重置"""
        # 设置一些非零值
        self.agent.safety_metrics['constraint_violations'] = 5
        self.agent.safety_metrics['safety_rewards'] = [1.0, 2.0, 3.0]
        self.agent.safety_metrics['fault_recoveries'] = 2
        
        # 重置
        self.agent.reset_safety_metrics()
        
        # 检查是否重置
        metrics = self.agent.get_safety_metrics()
        assert metrics['constraint_violations'] == 0
        assert metrics['safety_rewards'] == []
        assert metrics['fault_recoveries'] == 0
    
    @patch('ray.init')
    @patch('ray.shutdown')
    def test_algorithm_setup_mock(self, mock_shutdown, mock_init):
        """测试算法设置（模拟版本）"""
        # 这个测试需要模拟Ray环境
        with patch('src.agent.PPO') as mock_ppo:
            mock_algorithm = Mock()
            mock_ppo.return_value = mock_algorithm
            
            env_config = {'test': True}
            
            # 这里需要模拟ModelCatalog注册
            with patch('ray.rllib.models.ModelCatalog.register_custom_model'):
                self.agent.setup_algorithm(env_config)
                
                # 检查算法是否被设置
                assert self.agent.algorithm == mock_algorithm


class TestGRPOIntegration:
    """测试GRPO集成功能"""
    
    def test_reward_component_integration(self):
        """测试奖励组件集成"""
        config = GRPOConfig()
        agent = GRPOAgent(config)
        
        # 创建测试数据
        original_reward = 5.0
        state = np.random.randn(28)
        action = np.random.randn(6)
        info = {
            'distance_to_target': 2.0,
            'relative_velocity': 0.2,
            'attitude_error': 0.1,
            'thruster_health': [0.8] * 8  # 轻微故障
        }
        
        # 计算引导奖励
        guided_reward, components = agent.guided_reward_shaping(
            original_reward, state, action, info
        )
        
        # 验证所有组件都存在
        expected_components = ['original', 'safety', 'exploration', 
                             'constraint_penalty', 'fault_recovery', 'planning']
        for component in expected_components:
            assert component in components
        
        # 验证引导奖励是所有组件的合理组合
        manual_sum = (components['original'] + components['safety'] + 
                     components['exploration'] - components['constraint_penalty'] +
                     components['fault_recovery'] + components['planning'])
        
        assert abs(guided_reward - manual_sum) < 1e-6


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])