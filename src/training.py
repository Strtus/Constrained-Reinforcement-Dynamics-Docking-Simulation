"""
Advanced Training and Evaluation System for Safe RL Spacecraft Docking
=====================================================================

Implementation of training pipeline with hyperparameter optimization,
performance evaluation, and safety metrics for autonomous spacecraft docking.

Author: Strtus
License: MIT
"""

import numpy as np
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections import defaultdict, deque
try:
    import matplotlib.pyplot as plt  # optional
    import seaborn as sns  # optional
except Exception:
    plt = None
    sns = None
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TrainingConfiguration:
    """Training configuration with aerospace-specific parameters"""
    
    # Training schedule
    total_episodes: int = 100000
    evaluation_frequency: int = 1000
    checkpoint_frequency: int = 5000
    early_stopping_patience: int = 10000
    
    # Learning parameters
    learning_rate_schedule: str = "cosine_annealing"  # linear, exponential, cosine_annealing
    initial_learning_rate: float = 3e-4
    final_learning_rate: float = 1e-6
    batch_size: int = 256
    replay_buffer_size: int = 1000000
    
    # Environment parameters
    max_episode_steps: int = 1200
    fault_injection_probability: float = 0.02
    progressive_difficulty: bool = True
    curriculum_learning: bool = True
    
    # Safety parameters
    constraint_violation_threshold: float = 0.1
    safety_budget: float = 0.05  # Allowed constraint violation rate
    lagrange_multiplier_lr: float = 0.01
    
    # Evaluation parameters
    evaluation_episodes: int = 100
    deterministic_evaluation: bool = True
    monte_carlo_simulations: int = 1000
    
    # Hardware acceleration
    use_gpu: bool = True
    num_workers: int = 4
    vectorized_envs: int = 8
    
    # Logging and monitoring
    log_interval: int = 100
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    save_trajectories: bool = True


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for spacecraft docking"""
    
    # Mission success metrics
    success_rate: float = 0.0
    docking_accuracy_mean: float = 0.0
    docking_accuracy_std: float = 0.0
    time_to_dock_mean: float = 0.0
    time_to_dock_std: float = 0.0
    
    # Safety metrics
    collision_rate: float = 0.0
    constraint_violation_rate: float = 0.0
    max_approach_velocity: float = 0.0
    safety_margin_mean: float = 0.0
    
    # Efficiency metrics  
    fuel_consumption_mean: float = 0.0
    fuel_consumption_std: float = 0.0
    control_effort_mean: float = 0.0
    trajectory_smoothness: float = 0.0
    
    # Robustness metrics
    fault_recovery_rate: float = 0.0
    performance_degradation_under_faults: float = 0.0
    sensor_noise_robustness: float = 0.0
    
    # Learning metrics
    sample_efficiency: float = 0.0
    convergence_episode: int = 0
    final_reward: float = 0.0
    
    # Statistical significance
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    statistical_power: float = 0.0


class PerformanceAnalyzer:
    """Advanced performance analysis for spacecraft docking missions"""
    
    def __init__(self):
        self.trajectory_data = []
        self.episode_metrics = []
        self.safety_violations = []
        
    def analyze_episode(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze single episode performance"""
        trajectory = episode_data['trajectory']
        info = episode_data['info']
        
        metrics = {}
        
        # Extract trajectory components
        positions = np.array([step['position'] for step in trajectory])
        velocities = np.array([step['velocity'] for step in trajectory])
        actions = np.array([step['action'] for step in trajectory])
        
        # Mission success analysis
        # Support both vector and scalar trajectories (when only magnitudes are logged)
        if positions.ndim == 2:
            final_distance = np.linalg.norm(positions[-1])
            distances = np.linalg.norm(positions, axis=1)
        else:
            final_distance = float(np.abs(positions[-1]))
            distances = np.abs(positions)

        if velocities.ndim == 2:
            final_velocity = np.linalg.norm(velocities[-1])
            speeds = np.linalg.norm(velocities, axis=1)
        else:
            final_velocity = float(np.abs(velocities[-1]))
            speeds = np.abs(velocities)
        
        metrics['success'] = info.get('termination_reason') == 'docking_success'
        metrics['final_distance'] = final_distance
        metrics['final_velocity'] = final_velocity
        metrics['episode_length'] = len(trajectory)
        
        # Safety analysis
        # distances and speeds computed above
        
        metrics['min_distance'] = np.min(distances)
        metrics['max_velocity'] = np.max(speeds)
        metrics['collision'] = np.any(distances < 0.1)
        metrics['unsafe_approach'] = np.any((distances < 1.0) & (speeds > 0.5))
        
        # Control effort analysis
        thrust_magnitudes = np.linalg.norm(actions[:, :3], axis=1) if actions.ndim == 2 else np.array([0.0])
        torque_magnitudes = np.linalg.norm(actions[:, 3:], axis=1) if actions.ndim == 2 else np.array([0.0])
        
        metrics['total_thrust_effort'] = np.sum(thrust_magnitudes)
        metrics['total_torque_effort'] = np.sum(torque_magnitudes)
        metrics['control_smoothness'] = self._calculate_smoothness(actions)
        
        # Fuel consumption (simplified model)
        metrics['fuel_consumption'] = info.get('fuel_remaining', 1.0)
        
        # Trajectory analysis
        metrics['approach_efficiency'] = self._calculate_approach_efficiency(positions)
        metrics['trajectory_length'] = self._calculate_trajectory_length(positions)
        
        return metrics
    
    def _calculate_smoothness(self, actions: np.ndarray) -> float:
        """Calculate control smoothness metric"""
        if len(actions) < 2:
            return 0.0
        
        action_derivatives = np.diff(actions, axis=0)
        smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(action_derivatives, axis=1)))
        
        return float(smoothness)
    
    def _calculate_approach_efficiency(self, positions: np.ndarray) -> float:
        """Calculate approach efficiency (straight-line vs actual path)"""
        if len(positions) < 2:
            return 0.0
        
        straight_line_distance = np.linalg.norm(positions[0] - positions[-1])
        actual_path_length = self._calculate_trajectory_length(positions)
        
        if actual_path_length > 0:
            efficiency = straight_line_distance / actual_path_length
        else:
            efficiency = 0.0
        
        return float(efficiency)
    
    def _calculate_trajectory_length(self, positions: np.ndarray) -> float:
        """Calculate total trajectory length"""
        if len(positions) < 2:
            return 0.0
        
        path_segments = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(path_segments, axis=1)
        
        return float(np.sum(segment_lengths))
    
    def analyze_batch_performance(self, episode_batch: List[Dict]) -> EvaluationMetrics:
        """Analyze performance across multiple episodes"""
        
        batch_metrics = [self.analyze_episode(episode) for episode in episode_batch]
        
        # Aggregate metrics
        success_rate = float(np.mean([bool(m['success']) for m in batch_metrics]))
        
        # Filter successful episodes for accuracy analysis
        successful_episodes = [m for m in batch_metrics if m['success']]
        
        if successful_episodes:
            docking_accuracies = [m['final_distance'] for m in successful_episodes]
            docking_accuracy_mean = float(np.mean(docking_accuracies))
            docking_accuracy_std = float(np.std(docking_accuracies))
            
            episode_lengths = [m['episode_length'] for m in successful_episodes]
            time_to_dock_mean = float(np.mean(episode_lengths))
            time_to_dock_std = float(np.std(episode_lengths))
        else:
            docking_accuracy_mean = docking_accuracy_std = 0.0
            time_to_dock_mean = time_to_dock_std = 0.0
        
        # Safety metrics
        collision_rate = float(np.mean([bool(m['collision']) for m in batch_metrics]))
        unsafe_approach_rate = float(np.mean([bool(m['unsafe_approach']) for m in batch_metrics]))
        max_velocity = float(np.max([m['max_velocity'] for m in batch_metrics]))
        
        # Efficiency metrics
        fuel_consumptions = [1.0 - m['fuel_consumption'] for m in batch_metrics]
        fuel_consumption_mean = float(np.mean(fuel_consumptions))
        fuel_consumption_std = float(np.std(fuel_consumptions))
        
        control_efforts = [m['total_thrust_effort'] + m['total_torque_effort'] 
                          for m in batch_metrics]
        control_effort_mean = float(np.mean(control_efforts))
        
        trajectory_smoothness = float(np.mean([m['control_smoothness'] for m in batch_metrics]))
        
        # Statistical analysis
        n = len(batch_metrics)
        if n > 1:
            ci_low, ci_high = stats.t.interval(
                0.95, n - 1,
                loc=success_rate,
                scale=stats.sem([float(m['success']) for m in batch_metrics])
            )
            confidence_interval = (float(ci_low), float(ci_high)) if ci_low is not None and ci_high is not None else (0.0, 0.0)
        else:
            confidence_interval = (0.0, 0.0)
        
        return EvaluationMetrics(
            success_rate=float(success_rate),
            docking_accuracy_mean=float(docking_accuracy_mean),
            docking_accuracy_std=float(docking_accuracy_std),
            time_to_dock_mean=float(time_to_dock_mean),
            time_to_dock_std=float(time_to_dock_std),
            collision_rate=float(collision_rate),
            constraint_violation_rate=float(unsafe_approach_rate),
            max_approach_velocity=float(max_velocity),
            fuel_consumption_mean=float(fuel_consumption_mean),
            fuel_consumption_std=float(fuel_consumption_std),
            control_effort_mean=float(control_effort_mean),
            trajectory_smoothness=float(trajectory_smoothness),
            confidence_interval_95=(float(confidence_interval[0]), float(confidence_interval[1]))
        )


class HyperparameterOptimizer:
    """Bayesian optimization for hyperparameter tuning"""
    
    def __init__(self, parameter_space: Dict[str, Tuple], objective_metric: str = 'success_rate'):
        self.parameter_space = parameter_space
        self.objective_metric = objective_metric
        self.evaluation_history = []
        
    def suggest_parameters(self, n_iterations: int = 5) -> Dict[str, Any]:
        """Suggest next hyperparameters using Bayesian optimization"""
        
        if len(self.evaluation_history) < n_iterations:
            # Random sampling for initial exploration
            return self._random_sample()
        else:
            # Bayesian optimization (simplified implementation)
            return self._bayesian_suggest()
    
    def _random_sample(self) -> Dict[str, Any]:
        """Random sampling from parameter space"""
        params = {}
        
        for param_name, (min_val, max_val) in self.parameter_space.items():
            if isinstance(min_val, int):
                params[param_name] = np.random.randint(min_val, max_val + 1)
            else:
                params[param_name] = np.random.uniform(min_val, max_val)
        
        return params
    
    def _bayesian_suggest(self) -> Dict[str, Any]:
        """Simplified Bayesian optimization suggestion"""
        # For this implementation, use best parameters with noise
        best_eval = max(self.evaluation_history, key=lambda x: x['performance'])
        best_params = best_eval['parameters']
        
        # Add noise to best parameters
        noisy_params = {}
        for param_name, value in best_params.items():
            min_val, max_val = self.parameter_space[param_name]
            noise_scale = 0.1 * (max_val - min_val)
            
            if isinstance(value, int):
                noisy_value = int(np.clip(value + np.random.normal(0, noise_scale), 
                                        min_val, max_val))
            else:
                noisy_value = np.clip(value + np.random.normal(0, noise_scale), 
                                    min_val, max_val)
            
            noisy_params[param_name] = noisy_value
        
        return noisy_params
    
    def update_evaluation(self, parameters: Dict[str, Any], performance: float):
        """Update evaluation history"""
        self.evaluation_history.append({
            'parameters': parameters.copy(),
            'performance': performance
        })
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best parameters found so far"""
        if not self.evaluation_history:
            return {}
        
        best_eval = max(self.evaluation_history, key=lambda x: x['performance'])
        return best_eval['parameters']


class CurriculumLearning:
    """Curriculum learning for progressive difficulty increase"""
    
    def __init__(self):
        self.current_difficulty = 0.0
        self.success_window = deque(maxlen=100)
        self.difficulty_progression = [
            {'initial_distance': (10, 30), 'fault_rate': 0.0, 'noise_scale': 0.5},
            {'initial_distance': (30, 60), 'fault_rate': 0.01, 'noise_scale': 0.75},
            {'initial_distance': (60, 100), 'fault_rate': 0.02, 'noise_scale': 1.0},
            {'initial_distance': (100, 150), 'fault_rate': 0.03, 'noise_scale': 1.25},
            {'initial_distance': (50, 150), 'fault_rate': 0.05, 'noise_scale': 1.5}
        ]
        
    def update_progress(self, success: bool):
        """Update learning progress"""
        self.success_window.append(success)
        
        if len(self.success_window) >= 50:
            recent_success_rate = np.mean(self.success_window)
            
            # Advance curriculum if success rate is high enough
            if recent_success_rate > 0.8 and self.current_difficulty < len(self.difficulty_progression) - 1:
                self.current_difficulty += 0.1
                logger.info(f"Curriculum advanced to difficulty {self.current_difficulty:.1f}")
            # Reduce difficulty if success rate is too low
            elif recent_success_rate < 0.3 and self.current_difficulty > 0:
                self.current_difficulty = max(0, self.current_difficulty - 0.1)
                logger.info(f"Curriculum reduced to difficulty {self.current_difficulty:.1f}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current curriculum configuration"""
        difficulty_level = min(int(self.current_difficulty), len(self.difficulty_progression) - 1)
        
        # Interpolate between difficulty levels if needed
        if self.current_difficulty % 1.0 > 0:
            next_level = min(difficulty_level + 1, len(self.difficulty_progression) - 1)
            alpha = self.current_difficulty % 1.0
            
            current_config = self.difficulty_progression[difficulty_level]
            next_config = self.difficulty_progression[next_level]
            
            # Interpolate configuration
            config = {}
            for key in current_config:
                if isinstance(current_config[key], tuple):
                    # Interpolate tuple values
                    val1 = current_config[key]
                    val2 = next_config[key]
                    config[key] = (
                        val1[0] + alpha * (val2[0] - val1[0]),
                        val1[1] + alpha * (val2[1] - val1[1])
                    )
                else:
                    # Interpolate scalar values
                    val1 = current_config[key]
                    val2 = next_config[key]
                    config[key] = val1 + alpha * (val2 - val1)
        else:
            config = self.difficulty_progression[difficulty_level]
        
        return config


class TrainingPipeline:
    """Complete training pipeline for safe RL spacecraft docking"""
    
    def __init__(self, 
                 agent_class: type,
                 environment_class: type,
                 config: TrainingConfiguration):
        
        self.agent_class = agent_class
        self.environment_class = environment_class
        self.config = config
        
        # Initialize components
        self.performance_analyzer = PerformanceAnalyzer()
        self.curriculum = CurriculumLearning() if config.curriculum_learning else None
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_performance = 0.0
        self.early_stopping_counter = 0
        
        # Data collection
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'safety_violations': [],
            'fuel_consumption': [],
            'training_time': []
        }
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        self.log_dir = Path(f"logs/training_{int(time.time())}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        logger.info("Training pipeline initialized")
        logger.info(f"Configuration: {asdict(self.config)}")
    
    def train_agent(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop"""
        
        logger.info("Starting agent training")
        start_time = time.time()
        
        # Initialize environment and agent
        env_config = self.get_current_env_config()
        env = self.environment_class(env_config)
        agent = self.agent_class(env.observation_space.shape[0], env.action_space.shape[0])
        
        try:
            # Training loop
            for episode in range(self.config.total_episodes):
                self.episode_count = episode
                
                # Run training episode
                episode_data = self.run_training_episode(env, agent)
                self.process_episode_data(episode_data)
                
                # Periodic evaluation
                if episode % self.config.evaluation_frequency == 0:
                    eval_metrics = self.evaluate_agent(agent, env)
                    self.process_evaluation_metrics(eval_metrics)
                    
                    # Check for early stopping
                    if self.check_early_stopping(eval_metrics):
                        logger.info(f"Early stopping triggered at episode {episode}")
                        break
                
                # Save checkpoints
                if episode % self.config.checkpoint_frequency == 0 and save_path:
                    checkpoint_path = f"{save_path}_episode_{episode}.pt"
                    agent.save_model(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Update curriculum
                if self.curriculum:
                    success = episode_data.get('success', False)
                    self.curriculum.update_progress(success)
                    env_config = self.get_current_env_config()
                    env = self.environment_class(env_config)  # Recreate with new config
                
                # Logging
                if episode % self.config.log_interval == 0:
                    self.log_training_progress(episode, episode_data)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final evaluation and save
            final_metrics = self.evaluate_agent(agent, env)
            training_time = time.time() - start_time
            
            # Save final model
            if save_path:
                agent.save_model(f"{save_path}_final.pt")
                self.save_training_results(f"{save_path}_results.json", final_metrics, training_time)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return {
                'final_metrics': final_metrics,
                'training_time': training_time,
                'total_episodes': self.episode_count,
                'training_history': self.training_history
            }
    
    def run_training_episode(self, env, agent) -> Dict[str, Any]:
        """Run single training episode"""
        
        rs = env.reset()
        state = rs[0] if isinstance(rs, tuple) else rs
        episode_reward = 0.0
        episode_steps = 0
        trajectory = []
        info: Dict[str, Any] = {}
        
        done = False
        while not done and episode_steps < self.config.max_episode_steps:
            # Select action
            action, action_info = agent.select_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            constraint_costs = np.array(list(info.get('constraint_violations', {}).values()))
            lagrange_multipliers = info.get('lagrange_multipliers', np.zeros(4))
            
            agent.store_experience(
                state, action, reward, next_state, done, 
                constraint_costs, lagrange_multipliers
            )
            
            # Record trajectory
            trajectory.append({
                'state': state.copy(),
                'action': action.copy(),
                'reward': reward,
                'position': info.get('distance_to_target', 0),
                'velocity': info.get('relative_velocity_magnitude', 0)
            })
            
            # Update
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Agent update
            if self.total_steps % agent.config.update_frequency == 0:
                update_info = agent.update_policy()
                if update_info and self.total_steps % (agent.config.update_frequency * 10) == 0:
                    logger.debug(f"Agent update: {update_info}")
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'trajectory': trajectory,
            'info': info,
            'success': info.get('termination_reason') == 'docking_success'
        }
    
    def evaluate_agent(self, agent, env) -> EvaluationMetrics:
        """Comprehensive agent evaluation"""
        
        logger.info("Starting agent evaluation")
        evaluation_episodes = []
        
        for eval_ep in range(self.config.evaluation_episodes):
            rs = env.reset()
            state = rs[0] if isinstance(rs, tuple) else rs
            episode_data = {
                'trajectory': [],
                'info': {}
            }
            
            done = False
            steps = 0
            
            while not done and steps < self.config.max_episode_steps:
                action, _ = agent.select_action(state, deterministic=self.config.deterministic_evaluation)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_data['trajectory'].append({
                    'position': info.get('distance_to_target', 0),
                    'velocity': info.get('relative_velocity_magnitude', 0),
                    'action': action.copy()
                })
                
                state = next_state
                steps += 1
            
            episode_data['info'] = info
            evaluation_episodes.append(episode_data)
        
        # Analyze performance
        metrics = self.performance_analyzer.analyze_batch_performance(evaluation_episodes)
        
        logger.info(f"Evaluation completed: Success rate = {metrics.success_rate:.3f}")
        
        return metrics
    
    def get_current_env_config(self) -> Dict[str, Any]:
        """Get current environment configuration based on curriculum"""
        
        base_config = {
            'max_steps': self.config.max_episode_steps,
            'fault_injection_probability': self.config.fault_injection_probability
        }
        
        if self.curriculum:
            curriculum_config = self.curriculum.get_current_config()
            base_config.update(curriculum_config)
        
        return base_config
    
    def process_episode_data(self, episode_data: Dict[str, Any]):
        """Process and store episode data"""
        
        self.training_history['episodes'].append(self.episode_count)
        self.training_history['rewards'].append(episode_data['episode_reward'])
        
        # Compute rolling averages
        window_size = min(100, len(self.training_history['rewards']))
        if len(self.training_history['rewards']) >= window_size:
            recent_rewards = self.training_history['rewards'][-window_size:]
            recent_success = [ep['success'] for ep in [episode_data]]  # Would need episode history
            
            # For now, just store current episode success
            success_rate = 1.0 if episode_data['success'] else 0.0
            self.training_history['success_rates'].append(success_rate)
    
    def process_evaluation_metrics(self, metrics: EvaluationMetrics):
        """Process evaluation metrics"""
        
        # Update best performance tracking
        if metrics.success_rate > self.best_performance:
            self.best_performance = metrics.success_rate
            self.early_stopping_counter = 0
            logger.info(f"New best performance: {self.best_performance:.3f}")
        else:
            self.early_stopping_counter += 1
    
    def check_early_stopping(self, metrics: EvaluationMetrics) -> bool:
        """Check early stopping criteria"""
        
        # Stop if no improvement for patience episodes
        if self.early_stopping_counter >= self.config.early_stopping_patience:
            return True
        
        # Stop if performance target achieved
        if metrics.success_rate >= 0.95 and metrics.collision_rate <= 0.01:
            logger.info("Performance target achieved!")
            return True
        
        return False
    
    def log_training_progress(self, episode: int, episode_data: Dict[str, Any]):
        """Log training progress"""
        
        logger.info(
            f"Episode {episode}: "
            f"Reward = {episode_data['episode_reward']:.2f}, "
            f"Steps = {episode_data['episode_steps']}, "
            f"Success = {episode_data['success']}"
        )
    
    def save_training_results(self, filepath: str, final_metrics: EvaluationMetrics, 
                            training_time: float):
        """Save comprehensive training results"""
        
        results = {
            'configuration': asdict(self.config),
            'final_metrics': asdict(final_metrics),
            'training_time': training_time,
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'best_performance': self.best_performance,
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {filepath}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Generate training curve plots"""
        
        # Lazy import in case matplotlib/seaborn are not installed in minimal setups
        global plt, sns
        if plt is None or sns is None:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                import seaborn as sns  # type: ignore
            except Exception:
                logger.warning("Matplotlib/Seaborn not available; skipping plot generation")
                return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward curve
        if self.training_history['rewards']:
            axes[0, 0].plot(self.training_history['episodes'], self.training_history['rewards'])
            axes[0, 0].set_title('Training Reward')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Success rate curve
        if self.training_history['success_rates']:
            axes[0, 1].plot(self.training_history['episodes'][-len(self.training_history['success_rates']):], 
                           self.training_history['success_rates'])
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        # Add more plots as needed
        axes[1, 0].text(0.5, 0.5, 'Fuel Consumption\n(To be implemented)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'Safety Violations\n(To be implemented)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def optimize_hyperparameters(agent_class: type, environment_class: type, 
                           parameter_space: Dict[str, Tuple],
                           n_trials: int = 20) -> Dict[str, Any]:
    """Run hyperparameter optimization"""
    
    logger.info("Starting hyperparameter optimization")
    
    optimizer = HyperparameterOptimizer(parameter_space)
    best_params = {}
    best_performance = 0.0
    
    for trial in range(n_trials):
        logger.info(f"Hyperparameter trial {trial + 1}/{n_trials}")
        
        # Get suggested parameters
        params = optimizer.suggest_parameters()
        
        # Create config with suggested parameters
        config = TrainingConfiguration()
        for param_name, value in params.items():
            setattr(config, param_name, value)
        
        # Reduce episodes for hyperparameter search
        config.total_episodes = 5000
        config.evaluation_frequency = 1000
        
        try:
            # Train with suggested parameters
            pipeline = TrainingPipeline(agent_class, environment_class, config)
            results = pipeline.train_agent()
            
            performance = results['final_metrics'].success_rate
            optimizer.update_evaluation(params, performance)
            
            if performance > best_performance:
                best_performance = performance
                best_params = params.copy()
                logger.info(f"New best hyperparameters found: {best_params}")
            
        except Exception as e:
            logger.error(f"Trial {trial} failed: {e}")
            optimizer.update_evaluation(params, 0.0)  # Assign worst performance
    
    logger.info(f"Hyperparameter optimization completed")
    logger.info(f"Best performance: {best_performance:.3f}")
    logger.info(f"Best parameters: {best_params}")
    
    return {
        'best_parameters': best_params,
        'best_performance': best_performance,
        'optimization_history': optimizer.evaluation_history
    }