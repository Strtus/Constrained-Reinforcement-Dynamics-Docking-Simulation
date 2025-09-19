#!/usr/bin/env python3
"""
千帆卫星训练脚本
==================

Author: Strtus
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "configs"))

# 导入配置
try:
    from qianfan_config import (
        QIANFAN_SATELLITE_CONFIG, 
        QIANFAN_MISSION_CONFIG, 
        QIANFAN_TRAINING_CONFIG
    )
    from qianfan_simulator import create_qianfan_simulator, QIANFAN_SIMULATION_CONFIG
except ImportError as e:
    print(f"配置导入错误: {e}")

# 导入框架组件
try:
    from environment import SpacecraftRvDEnvironment
    from agent import GRPOAgent
    from training import TrainingPipeline
    from visualization import VisualizationEngine
except ImportError as e:
    print(f"框架组件导入错误: {e}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'qianfan_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QianfanTrainingManager:
    """千帆卫星训练管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化千帆训练管理器"""
        self.config = self._load_config(config_path)
        self.training_id = f"qianfan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建输出目录
        self.output_dir = Path(f"training_outputs/{self.training_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.environment = None
        self.agent = None
        self.trainer = None
        self.visualizer = None
        
        logger.info(f"千帆训练管理器初始化完成，训练ID: {self.training_id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载训练配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    custom_config = yaml.safe_load(f)
                else:
                    custom_config = json.load(f)
            logger.info(f"加载自定义配置: {config_path}")
        else:
            custom_config = {}
        
        # 合并配置
        config = {
            'satellite': QIANFAN_SATELLITE_CONFIG,
            'mission': QIANFAN_MISSION_CONFIG,
            'training': QIANFAN_TRAINING_CONFIG,
            'simulation': QIANFAN_SIMULATION_CONFIG
        }
        
        # 应用自定义配置
        self._deep_update(config, custom_config)
        return config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_environment(self):
        """设置千帆训练环境"""
        logger.info("设置千帆卫星训练环境...")
        
        # 环境配置
        env_config = {
            'max_steps': self.config['training']['environment_config']['max_episode_steps'],
            'dt': self.config['training']['environment_config']['dt'],
            'initial_separation_range': self.config['training']['environment_config']['initial_separation_range'],
            'reward_weights': self.config['training']['reward_config'],
            'safety_constraints': self.config['mission']['safety_constraints'],
            'satellite_config': self.config['satellite'],
            'mission_config': self.config['mission']
        }
        
        try:
            self.environment = SpacecraftRvDEnvironment(config=env_config)
            logger.info("千帆环境创建成功")
        except Exception as e:
            logger.error(f"千帆环境创建失败: {e}")
            raise
    
    def setup_agent(self):
        """设置千帆智能体"""
        logger.info("设置千帆卫星智能体...")
        
        if self.environment is None:
            raise ValueError("必须先设置环境")
        
        # 智能体配置
        agent_config = self.config['training']['agent_config'].copy()
        agent_config.update({
            'observation_space': self.environment.observation_space,
            'action_space': self.environment.action_space,
            'device': 'cuda' if self._check_cuda() else 'cpu'
        })
        
        try:
            self.agent = GRPOAgent(**agent_config)
            logger.info("千帆智能体创建成功")
        except Exception as e:
            logger.error(f"千帆智能体创建失败: {e}")
            raise
    
    def setup_trainer(self):
        """设置训练管道"""
        logger.info("设置千帆训练管道...")
        
        if self.environment is None or self.agent is None:
            raise ValueError("必须先设置环境和智能体")
        
        # 训练配置
        training_config = {
            'total_timesteps': 1000000,  # 千帆任务需要更多训练
            'eval_freq': 10000,
            'save_freq': 50000,
            'log_interval': 1000,
            'curriculum_learning': True,
            'curriculum_stages': [
                {'name': 'basic_approach', 'max_distance': 100, 'steps': 200000},
                {'name': 'precise_docking', 'max_distance': 50, 'steps': 300000},
                {'name': 'fault_tolerance', 'max_distance': 1000, 'steps': 500000}
            ],
            'output_dir': str(self.output_dir)
        }
        
        try:
            self.trainer = TrainingPipeline(
                environment=self.environment,
                agent=self.agent,
                config=training_config
            )
            logger.info("千帆训练管道创建成功")
        except Exception as e:
            logger.error(f"千帆训练管道创建失败: {e}")
            raise
    
    def setup_visualizer(self):
        """设置可视化工具"""
        logger.info("设置千帆可视化工具...")
        
        viz_config = {
            'output_dir': str(self.output_dir / 'visualizations'),
            'real_time_plotting': False,
            'save_animations': True,
            'plot_trajectories': True,
            'plot_rewards': True,
            'plot_safety_metrics': True
        }
        
        try:
            self.visualizer = VisualizationEngine(config=viz_config)
            logger.info("千帆可视化工具创建成功")
        except Exception as e:
            logger.error(f"千帆可视化工具创建失败: {e}")
            raise
    
    def run_training(self):
        """执行千帆训练"""
        logger.info("开始千帆卫星训练...")
        
        # 保存配置
        config_file = self.output_dir / 'training_config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        try:
            # 执行训练
            training_results = self.trainer.train_with_curriculum()
            
            # 保存结果
            results_file = self.output_dir / 'training_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"千帆训练完成，结果保存至: {self.output_dir}")
            return training_results
            
        except Exception as e:
            logger.error(f"千帆训练失败: {e}")
            raise
    
    def evaluate_model(self, model_path: Optional[str] = None):
        """评估千帆模型"""
        logger.info("评估千帆模型...")
        
        if model_path:
            self.agent.load_model(model_path)
        
        try:
            evaluation_results = self.trainer.evaluate_agent(num_episodes=100)
            
            # 保存评估结果
            eval_file = self.output_dir / 'evaluation_results.json'
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"千帆模型评估完成，结果保存至: {eval_file}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"千帆模型评估失败: {e}")
            raise
    
    def _check_cuda(self) -> bool:
        """检查CUDA可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='千帆卫星训练脚本')
    parser.add_argument('--config', type=str, help='自定义配置文件路径')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='train', 
                       help='运行模式')
    parser.add_argument('--model', type=str, help='评估模型路径')
    parser.add_argument('--resume', type=str, help='继续训练的检查点路径')
    
    args = parser.parse_args()
    
    try:
        # 创建训练管理器
        manager = QianfanTrainingManager(config_path=args.config)
        
        # 设置组件
        manager.setup_environment()
        manager.setup_agent()
        manager.setup_trainer()
        manager.setup_visualizer()
        
        # 执行任务
        if args.mode in ['train', 'both']:
            if args.resume:
                logger.info(f"从检查点继续训练: {args.resume}")
                manager.agent.load_model(args.resume)
            
            training_results = manager.run_training()
            print(f"训练完成！成功率: {training_results.get('final_success_rate', 'N/A')}")
        
        if args.mode in ['eval', 'both']:
            eval_results = manager.evaluate_model(model_path=args.model)
            print(f"评估完成！平均奖励: {eval_results.get('mean_reward', 'N/A')}")
        
    except Exception as e:
        logger.error(f"千帆训练失败: {e}")
        sys.exit(1)
    
    logger.info("千帆训练程序完成")


if __name__ == "__main__":
    main()