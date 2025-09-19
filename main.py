#!/usr/bin/env python3
"""
Main execution entry point for the Spacecraft Safe RL Framework.
Integrates all advanced components for comprehensive mission simulation.

Aerospace Engineering Implementation
Author: Strtus
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Suppress scientific computing warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import framework components
try:
    from environment import SpacecraftDockingEnvironment
    from agent import GRPOAgent
    from simulator import SpacecraftSimulator
    from training import TrainingPipeline
    from visualization import VisualizationEngine, ComprehensiveAnalysisReport
except ImportError as e:
    print(f"Critical import error: {e}")
    print("Please ensure all components are in the src directory.")
    sys.exit(1)

import numpy as np
import torch


class SpacecraftMissionController:
    """
    Master controller for spacecraft rendezvous and docking missions.
    Integrates environment, agent, simulator, and training components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mission controller with configuration."""
        self.config = config
        self.setup_logging()
        
        # Initialize random seeds for reproducibility
        self.set_random_seeds(config.get('random_seed', 42))
        
        # Initialize components
        self.environment = None
        self.agent = None
        self.simulator = None
        self.training_pipeline = None
        self.visualization_engine = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Mission controller initialized")
    
    def setup_logging(self):
        """Configure professional logging system."""
        log_level = self.config.get('log_level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('spacecraft_mission.log')
            ]
        )
    
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducible results."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def initialize_mission_components(self):
        """Initialize all mission-critical components."""
        self.logger.info("Initializing mission components...")
        
        # Environment initialization
        env_config = self.config.get('environment', {})
        self.environment = SpacecraftDockingEnvironment(**env_config)
        self.logger.info("Environment initialized")
        
        # Agent initialization
        agent_config = self.config.get('agent', {})
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        
        self.agent = GRPOAgent(
            observation_space=observation_space,
            action_space=action_space,
            **agent_config
        )
        self.logger.info("Agent initialized")
        
        # Simulator initialization
        sim_config = self.config.get('simulator', {})
        self.simulator = SpacecraftSimulator(**sim_config)
        self.logger.info("Simulator initialized")
        
        # Training pipeline initialization
        training_config = self.config.get('training', {})
        self.training_pipeline = TrainingPipeline(
            environment=self.environment,
            agent=self.agent,
            **training_config
        )
        self.logger.info("Training pipeline initialized")
        
        # Visualization engine
        viz_config = self.config.get('visualization', {})
        self.visualization_engine = VisualizationEngine(**viz_config)
        self.logger.info("Visualization engine initialized")
    
    def execute_training_mission(self):
        """Execute comprehensive training mission."""
        self.logger.info("Starting training mission...")
        
        try:
            # Run training with curriculum learning
            training_results = self.training_pipeline.train_with_curriculum()
            
            self.logger.info("Training mission completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training mission failed: {e}")
            raise
    
    def execute_evaluation_mission(self, model_path: Optional[str] = None):
        """Execute evaluation mission with trained agent."""
        self.logger.info("Starting evaluation mission...")
        
        try:
            # Load trained model if provided
            if model_path:
                self.agent.load_model(model_path)
                self.logger.info(f"Loaded model from {model_path}")
            
            # Run evaluation episodes
            evaluation_results = self.training_pipeline.evaluate_agent(
                num_episodes=self.config.get('evaluation_episodes', 100)
            )
            
            # Generate comprehensive analysis
            analysis_report = ComprehensiveAnalysisReport(
                evaluation_results=evaluation_results,
                agent=self.agent,
                environment=self.environment
            )
            
            report_data = analysis_report.generate_complete_report()
            
            self.logger.info("Evaluation mission completed successfully")
            return evaluation_results, report_data
            
        except Exception as e:
            self.logger.error(f"Evaluation mission failed: {e}")
            raise
    
    def execute_simulation_mission(self, scenario_config: Dict[str, Any]):
        """Execute specific simulation scenario."""
        self.logger.info("Starting simulation mission...")
        
        try:
            # Configure simulation scenario
            self.simulator.configure_scenario(scenario_config)
            
            # Run simulation
            simulation_results = self.simulator.run_simulation()
            
            # Visualize results
            self.visualization_engine.create_trajectory_visualization(
                simulation_results
            )
            
            self.logger.info("Simulation mission completed successfully")
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Simulation mission failed: {e}")
            raise


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for spacecraft missions."""
    return {
        'random_seed': 42,
        'log_level': 'INFO',
        'environment': {
            'max_episode_steps': 2000,
            'safety_threshold': 0.1,
            'reward_shaping': True
        },
        'agent': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'lambda_gae': 0.95,
            'safety_weight': 0.5
        },
        'simulator': {
            'integration_method': 'dopri5',
            'fault_probability': 0.05,
            'sensor_noise_std': 0.01
        },
        'training': {
            'total_timesteps': 1000000,
            'curriculum_enabled': True,
            'hyperparameter_optimization': True
        },
        'visualization': {
            'save_plots': True,
            'interactive_mode': False
        },
        'evaluation_episodes': 100
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Spacecraft Safe RL Framework - Professional Implementation'
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'simulate'],
        default='train',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model (for evaluation)'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        help='Simulation scenario configuration'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
    
    # Initialize mission controller
    mission_controller = SpacecraftMissionController(config)
    mission_controller.initialize_mission_components()
    
    try:
        if args.mode == 'train':
            print("Executing training mission...")
            results = mission_controller.execute_training_mission()
            print(f"Training completed. Final performance: {results.get('final_performance', 'N/A')}")
            
        elif args.mode == 'evaluate':
            print("Executing evaluation mission...")
            eval_results, report = mission_controller.execute_evaluation_mission(args.model)
            print(f"Evaluation completed. Success rate: {eval_results.get('success_rate', 'N/A')}")
            
        elif args.mode == 'simulate':
            print("Executing simulation mission...")
            if args.scenario:
                import json
                with open(args.scenario, 'r') as f:
                    scenario_config = json.load(f)
            else:
                scenario_config = {'type': 'nominal_approach'}
            
            sim_results = mission_controller.execute_simulation_mission(scenario_config)
            print(f"Simulation completed. Mission status: {sim_results.get('status', 'N/A')}")
    
    except KeyboardInterrupt:
        print("\nMission interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Mission failed: {e}")
        sys.exit(1)
    
    print("Mission completed successfully")


if __name__ == "__main__":
    main()