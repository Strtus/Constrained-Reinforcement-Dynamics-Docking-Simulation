"""
Advanced Visualization and Analysis Tools for Safe RL Spacecraft Docking
=======================================================================

Implementation of 3D visualization, trajectory analysis, and 
AI decision interpretation tools for autonomous spacecraft rendezvous and docking.

Author: Strtus
License: MIT
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class VisualizationConfig:
    """Configuration for visualization and analysis"""
    
    # 3D visualization
    figure_size: Tuple[int, int] = (12, 9)
    animation_fps: int = 30
    trajectory_line_width: float = 2.0
    spacecraft_scale: float = 1.0
    
    # Colors and styling
    chaser_color: str = 'blue'
    target_color: str = 'red'
    trajectory_color: str = 'green'
    safety_zone_color: str = 'yellow'
    danger_zone_color: str = 'red'
    
    # Analysis parameters
    decision_analysis_window: int = 50
    attention_threshold: float = 0.1
    importance_threshold: float = 0.05
    
    # Export settings
    export_format: str = 'png'
    export_dpi: int = 300
    save_animations: bool = True


class TrajectoryAnalyzer:
    """Advanced trajectory analysis for spacecraft docking missions"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
    def analyze_trajectory_characteristics(self, trajectory: List[Dict]) -> Dict[str, Any]:
        """Analyze trajectory characteristics and performance metrics"""
        
        if not trajectory:
            return {}
        
        # Extract trajectory data
        positions = np.array([step['position'] for step in trajectory])
        velocities = np.array([step['velocity'] for step in trajectory])
        actions = np.array([step['action'] for step in trajectory])
        
        analysis = {}
        
        # Trajectory geometry analysis
        analysis['total_distance'] = self._calculate_path_length(positions)
        analysis['displacement'] = np.linalg.norm(positions[-1] - positions[0])
        analysis['efficiency'] = analysis['displacement'] / analysis['total_distance'] if analysis['total_distance'] > 0 else 0
        
        # Velocity profile analysis
        speeds = np.linalg.norm(velocities, axis=1)
        analysis['max_speed'] = np.max(speeds)
        analysis['avg_speed'] = np.mean(speeds)
        analysis['speed_variance'] = np.var(speeds)
        
        # Control effort analysis
        thrust_magnitudes = np.linalg.norm(actions[:, :3], axis=1)
        analysis['total_thrust_effort'] = np.sum(thrust_magnitudes)
        analysis['avg_thrust'] = np.mean(thrust_magnitudes)
        analysis['control_smoothness'] = self._calculate_control_smoothness(actions)
        
        # Safety analysis
        distances_to_target = np.linalg.norm(positions, axis=1)
        analysis['min_distance'] = np.min(distances_to_target)
        analysis['safety_violations'] = self._count_safety_violations(positions, velocities)
        
        # Approach phase analysis
        analysis['approach_phases'] = self._identify_approach_phases(positions, velocities)
        
        # Fuel efficiency estimation
        analysis['estimated_fuel_consumption'] = self._estimate_fuel_consumption(actions)
        
        return analysis
    
    def _calculate_path_length(self, positions: np.ndarray) -> float:
        """Calculate total path length"""
        if len(positions) < 2:
            return 0.0
        
        path_segments = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(path_segments, axis=1)
        
        return np.sum(segment_lengths)
    
    def _calculate_control_smoothness(self, actions: np.ndarray) -> float:
        """Calculate control smoothness metric"""
        if len(actions) < 2:
            return 1.0
        
        action_differences = np.diff(actions, axis=0)
        smoothness_metric = 1.0 / (1.0 + np.mean(np.linalg.norm(action_differences, axis=1)))
        
        return smoothness_metric
    
    def _count_safety_violations(self, positions: np.ndarray, velocities: np.ndarray) -> int:
        """Count safety constraint violations"""
        violations = 0
        
        distances = np.linalg.norm(positions, axis=1)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Collision avoidance violations
        violations += np.sum(distances < 0.5)
        
        # High-speed approach violations
        close_range_mask = distances < 2.0
        high_speed_mask = speeds > 0.5
        violations += np.sum(close_range_mask & high_speed_mask)
        
        return violations
    
    def _identify_approach_phases(self, positions: np.ndarray, velocities: np.ndarray) -> Dict[str, Any]:
        """Identify different phases of the approach"""
        
        distances = np.linalg.norm(positions, axis=1)
        speeds = np.linalg.norm(velocities, axis=1)
        
        phases = {}
        
        # Far-field approach (>50m)
        far_field_mask = distances > 50.0
        if np.any(far_field_mask):
            phases['far_field'] = {
                'duration': np.sum(far_field_mask),
                'avg_speed': np.mean(speeds[far_field_mask]),
                'start_distance': np.max(distances[far_field_mask]),
                'end_distance': 50.0
            }
        
        # Mid-field approach (10-50m)
        mid_field_mask = (distances <= 50.0) & (distances > 10.0)
        if np.any(mid_field_mask):
            phases['mid_field'] = {
                'duration': np.sum(mid_field_mask),
                'avg_speed': np.mean(speeds[mid_field_mask]),
                'start_distance': 50.0,
                'end_distance': 10.0
            }
        
        # Near-field approach (1-10m)
        near_field_mask = (distances <= 10.0) & (distances > 1.0)
        if np.any(near_field_mask):
            phases['near_field'] = {
                'duration': np.sum(near_field_mask),
                'avg_speed': np.mean(speeds[near_field_mask]),
                'start_distance': 10.0,
                'end_distance': 1.0
            }
        
        # Terminal approach (<1m)
        terminal_mask = distances <= 1.0
        if np.any(terminal_mask):
            phases['terminal'] = {
                'duration': np.sum(terminal_mask),
                'avg_speed': np.mean(speeds[terminal_mask]),
                'start_distance': 1.0,
                'final_distance': np.min(distances)
            }
        
        return phases
    
    def _estimate_fuel_consumption(self, actions: np.ndarray) -> float:
        """Estimate fuel consumption based on control actions"""
        
        # Simplified fuel consumption model
        thrust_magnitudes = np.linalg.norm(actions[:, :3], axis=1)
        
        # Assume specific impulse of 230s and thruster efficiency
        specific_impulse = 230.0  # seconds
        g0 = 9.81  # m/s^2
        time_step = 0.5  # seconds
        
        # Fuel mass flow rate = thrust / (Isp * g0)
        fuel_flow_rates = thrust_magnitudes / (specific_impulse * g0)
        total_fuel_consumed = np.sum(fuel_flow_rates) * time_step
        
        return total_fuel_consumed


class DecisionAnalyzer:
    """AI decision analysis and interpretability tools"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
    def analyze_decision_patterns(self, trajectory: List[Dict], 
                                agent_states: List[Dict]) -> Dict[str, Any]:
        """Analyze AI decision patterns and extract interpretable insights"""
        
        if not trajectory or not agent_states:
            return {}
        
        analysis = {}
        
        # Extract decision data
        actions = np.array([step['action'] for step in trajectory])
        states = np.array([step['state'] for step in trajectory])
        
        # Action pattern analysis
        analysis['action_patterns'] = self._analyze_action_patterns(actions)
        
        # State-action correlation analysis
        analysis['state_action_correlations'] = self._analyze_state_action_correlations(states, actions)
        
        # Decision consistency analysis
        analysis['decision_consistency'] = self._analyze_decision_consistency(states, actions)
        
        # Critical decision identification
        analysis['critical_decisions'] = self._identify_critical_decisions(trajectory, actions)
        
        # Safety-related decisions
        analysis['safety_decisions'] = self._analyze_safety_decisions(trajectory, actions)
        
        return analysis
    
    def _analyze_action_patterns(self, actions: np.ndarray) -> Dict[str, Any]:
        """Analyze patterns in agent actions"""
        
        patterns = {}
        
        # Action magnitude analysis
        thrust_actions = actions[:, :3]
        torque_actions = actions[:, 3:]
        
        thrust_magnitudes = np.linalg.norm(thrust_actions, axis=1)
        torque_magnitudes = np.linalg.norm(torque_actions, axis=1)
        
        patterns['thrust_statistics'] = {
            'mean': np.mean(thrust_magnitudes),
            'std': np.std(thrust_magnitudes),
            'max': np.max(thrust_magnitudes),
            'usage_rate': np.mean(thrust_magnitudes > 0.1)
        }
        
        patterns['torque_statistics'] = {
            'mean': np.mean(torque_magnitudes),
            'std': np.std(torque_magnitudes),
            'max': np.max(torque_magnitudes),
            'usage_rate': np.mean(torque_magnitudes > 0.1)
        }
        
        # Action direction preferences
        patterns['thrust_direction_bias'] = self._calculate_direction_bias(thrust_actions)
        patterns['torque_direction_bias'] = self._calculate_direction_bias(torque_actions)
        
        # Control mode identification
        patterns['control_modes'] = self._identify_control_modes(actions)
        
        return patterns
    
    def _calculate_direction_bias(self, actions: np.ndarray) -> Dict[str, float]:
        """Calculate bias in action directions"""
        
        # Remove zero actions
        non_zero_mask = np.linalg.norm(actions, axis=1) > 1e-6
        if not np.any(non_zero_mask):
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        active_actions = actions[non_zero_mask]
        
        # Calculate average direction
        mean_direction = np.mean(active_actions, axis=0)
        direction_magnitude = np.linalg.norm(mean_direction)
        
        if direction_magnitude > 1e-6:
            normalized_direction = mean_direction / direction_magnitude
            bias = {
                'x': normalized_direction[0],
                'y': normalized_direction[1], 
                'z': normalized_direction[2]
            }
        else:
            bias = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        return bias
    
    def _identify_control_modes(self, actions: np.ndarray) -> Dict[str, Any]:
        """Identify different control modes used by the agent"""
        
        thrust_actions = actions[:, :3]
        torque_actions = actions[:, 3:]
        
        thrust_magnitudes = np.linalg.norm(thrust_actions, axis=1)
        torque_magnitudes = np.linalg.norm(torque_actions, axis=1)
        
        # Define thresholds
        thrust_threshold = 0.1
        torque_threshold = 0.1
        
        # Classify control modes
        thrust_only = (thrust_magnitudes > thrust_threshold) & (torque_magnitudes <= torque_threshold)
        torque_only = (thrust_magnitudes <= thrust_threshold) & (torque_magnitudes > torque_threshold)
        combined = (thrust_magnitudes > thrust_threshold) & (torque_magnitudes > torque_threshold)
        inactive = (thrust_magnitudes <= thrust_threshold) & (torque_magnitudes <= torque_threshold)
        
        modes = {
            'thrust_only': np.mean(thrust_only),
            'torque_only': np.mean(torque_only),
            'combined': np.mean(combined),
            'inactive': np.mean(inactive)
        }
        
        return modes
    
    def _analyze_state_action_correlations(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """Analyze correlations between states and actions"""
        
        correlations = {}
        
        # Position-action correlations
        positions = states[:, :3]  # Assuming first 3 elements are position
        velocities = states[:, 3:6]  # Assuming next 3 elements are velocity
        
        thrust_actions = actions[:, :3]
        
        # Calculate correlations
        pos_thrust_corr = self._calculate_correlation_matrix(positions, thrust_actions)
        vel_thrust_corr = self._calculate_correlation_matrix(velocities, thrust_actions)
        
        correlations['position_thrust'] = pos_thrust_corr
        correlations['velocity_thrust'] = vel_thrust_corr
        
        # Distance-based action analysis
        distances = np.linalg.norm(positions, axis=1)
        thrust_magnitudes = np.linalg.norm(thrust_actions, axis=1)
        
        if len(distances) > 1 and len(thrust_magnitudes) > 1:
            distance_thrust_corr = np.corrcoef(distances, thrust_magnitudes)[0, 1]
            correlations['distance_thrust_correlation'] = distance_thrust_corr
        
        return correlations
    
    def _calculate_correlation_matrix(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix between two arrays"""
        
        try:
            # Ensure arrays have same length
            min_length = min(len(array1), len(array2))
            array1_trimmed = array1[:min_length]
            array2_trimmed = array2[:min_length]
            
            # Calculate correlation for each component pair
            correlations = np.zeros((array1.shape[1], array2.shape[1]))
            
            for i in range(array1.shape[1]):
                for j in range(array2.shape[1]):
                    if np.std(array1_trimmed[:, i]) > 1e-6 and np.std(array2_trimmed[:, j]) > 1e-6:
                        correlations[i, j] = np.corrcoef(array1_trimmed[:, i], array2_trimmed[:, j])[0, 1]
            
            return correlations
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return np.zeros((array1.shape[1], array2.shape[1]))
    
    def _analyze_decision_consistency(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """Analyze consistency of decisions in similar states"""
        
        consistency = {}
        
        # Group similar states and analyze action variance
        # For simplicity, use position distance as similarity metric
        positions = states[:, :3]
        
        similarity_threshold = 1.0  # meters
        consistent_groups = 0
        total_variance = 0.0
        
        for i in range(len(positions)):
            # Find similar states
            distances = np.linalg.norm(positions - positions[i], axis=1)
            similar_indices = np.where(distances < similarity_threshold)[0]
            
            if len(similar_indices) > 1:
                similar_actions = actions[similar_indices]
                action_variance = np.mean(np.var(similar_actions, axis=0))
                total_variance += action_variance
                consistent_groups += 1
        
        if consistent_groups > 0:
            consistency['average_action_variance'] = total_variance / consistent_groups
            consistency['consistent_decision_groups'] = consistent_groups
        else:
            consistency['average_action_variance'] = 0.0
            consistency['consistent_decision_groups'] = 0
        
        return consistency
    
    def _identify_critical_decisions(self, trajectory: List[Dict], actions: np.ndarray) -> List[Dict]:
        """Identify critical decision points in the trajectory"""
        
        critical_decisions = []
        
        positions = np.array([step['position'] for step in trajectory])
        distances = np.linalg.norm(positions, axis=1)
        
        # Critical decision criteria
        for i, step in enumerate(trajectory):
            is_critical = False
            criticality_reasons = []
            
            # Close proximity decisions
            if distances[i] < 2.0:
                is_critical = True
                criticality_reasons.append("close_proximity")
            
            # High action magnitude decisions
            action_magnitude = np.linalg.norm(actions[i])
            if action_magnitude > 0.8:  # High action threshold
                is_critical = True
                criticality_reasons.append("high_action_magnitude")
            
            # Rapid state changes
            if i > 0:
                position_change = np.linalg.norm(positions[i] - positions[i-1])
                if position_change > 0.5:  # Rapid movement threshold
                    is_critical = True
                    criticality_reasons.append("rapid_state_change")
            
            if is_critical:
                critical_decisions.append({
                    'timestep': i,
                    'position': positions[i],
                    'action': actions[i],
                    'criticality_reasons': criticality_reasons,
                    'distance_to_target': distances[i]
                })
        
        return critical_decisions
    
    def _analyze_safety_decisions(self, trajectory: List[Dict], actions: np.ndarray) -> Dict[str, Any]:
        """Analyze safety-related decision making"""
        
        safety_analysis = {}
        
        positions = np.array([step['position'] for step in trajectory])
        velocities = np.array([step['velocity'] for step in trajectory])
        
        distances = np.linalg.norm(positions, axis=1)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Safety-critical situations
        danger_zone_mask = distances < 1.0
        high_speed_mask = speeds > 0.5
        
        if np.any(danger_zone_mask):
            danger_actions = actions[danger_zone_mask]
            safety_analysis['danger_zone_actions'] = {
                'count': np.sum(danger_zone_mask),
                'avg_action_magnitude': np.mean(np.linalg.norm(danger_actions, axis=1)),
                'emergency_braking_rate': np.mean(danger_actions[:, 0] < -0.3)  # Negative thrust
            }
        
        if np.any(high_speed_mask):
            high_speed_actions = actions[high_speed_mask]
            safety_analysis['high_speed_actions'] = {
                'count': np.sum(high_speed_mask),
                'avg_action_magnitude': np.mean(np.linalg.norm(high_speed_actions, axis=1)),
                'deceleration_rate': np.mean(high_speed_actions[:, 0] < 0)  # Braking actions
            }
        
        # Safety constraint adherence
        safety_violations = 0
        for i in range(len(distances)):
            if distances[i] < 0.5:  # Collision zone
                safety_violations += 1
            elif distances[i] < 2.0 and speeds[i] > 0.5:  # High-speed approach
                safety_violations += 1
        
        safety_analysis['safety_violation_rate'] = safety_violations / len(trajectory)
        
        return safety_analysis


class VisualizationEngine:
    """3D visualization engine for spacecraft docking scenarios"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
    def create_3d_trajectory_plot(self, trajectory: List[Dict], 
                                save_path: Optional[str] = None) -> str:
        """Create 3D trajectory visualization"""
        
        # For now, create a text-based representation
        # In a full implementation, this would use matplotlib or plotly
        
        positions = np.array([step['position'] for step in trajectory])
        
        plot_data = {
            'trajectory_points': len(positions),
            'start_position': positions[0].tolist() if len(positions) > 0 else [0, 0, 0],
            'end_position': positions[-1].tolist() if len(positions) > 0 else [0, 0, 0],
            'trajectory_length': self._calculate_trajectory_length(positions),
            'visualization_type': '3D_trajectory'
        }
        
        # Generate visualization description
        description = f"""
3D Trajectory Visualization
==========================
Trajectory Points: {plot_data['trajectory_points']}
Start Position: {plot_data['start_position']}
End Position: {plot_data['end_position']}
Total Length: {plot_data['trajectory_length']:.2f}m

Visualization would show:
- 3D spacecraft trajectory in Hill frame coordinates
- Target spacecraft at origin
- Safety zones and constraints
- Velocity vectors at key points
- Thruster firing indicators
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(plot_data, f, indent=2)
                f.write(f"\n\n{description}")
            
            return f"3D trajectory plot saved to {save_path}"
        
        return description
    
    def create_decision_heatmap(self, decision_analysis: Dict[str, Any],
                              save_path: Optional[str] = None) -> str:
        """Create decision analysis heatmap"""
        
        # Create text-based heatmap representation
        
        action_patterns = decision_analysis.get('action_patterns', {})
        thrust_stats = action_patterns.get('thrust_statistics', {})
        
        heatmap_data = {
            'thrust_mean': thrust_stats.get('mean', 0.0),
            'thrust_std': thrust_stats.get('std', 0.0),
            'thrust_usage_rate': thrust_stats.get('usage_rate', 0.0),
            'visualization_type': 'decision_heatmap'
        }
        
        description = f"""
Decision Analysis Heatmap
========================
Thrust Usage Rate: {heatmap_data['thrust_usage_rate']:.1%}
Average Thrust: {heatmap_data['thrust_mean']:.3f}
Thrust Variability: {heatmap_data['thrust_std']:.3f}

Heatmap would show:
- Action frequency across state space
- Decision confidence regions
- Critical decision boundaries
- Safety constraint influences
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(heatmap_data, f, indent=2)
                f.write(f"\n\n{description}")
            
            return f"Decision heatmap saved to {save_path}"
        
        return description
    
    def _calculate_trajectory_length(self, positions: np.ndarray) -> float:
        """Calculate total trajectory length"""
        if len(positions) < 2:
            return 0.0
        
        path_segments = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(path_segments, axis=1)
        
        return np.sum(segment_lengths)


class ComprehensiveAnalysisReport:
    """Generate comprehensive analysis reports"""
    
    def __init__(self):
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.decision_analyzer = DecisionAnalyzer()
        self.visualization_engine = VisualizationEngine()
        
    def generate_mission_report(self, trajectory: List[Dict], 
                              agent_states: List[Dict],
                              save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive mission analysis report"""
        
        logger.info("Generating comprehensive mission analysis report")
        
        # Perform all analyses
        trajectory_analysis = self.trajectory_analyzer.analyze_trajectory_characteristics(trajectory)
        decision_analysis = self.decision_analyzer.analyze_decision_patterns(trajectory, agent_states)
        
        # Create visualizations
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            trajectory_plot = self.visualization_engine.create_3d_trajectory_plot(
                trajectory, str(save_dir / "trajectory_3d.json")
            )
            
            decision_heatmap = self.visualization_engine.create_decision_heatmap(
                decision_analysis, str(save_dir / "decision_heatmap.json")
            )
        else:
            trajectory_plot = "Visualization not saved"
            decision_heatmap = "Visualization not saved"
        
        # Compile comprehensive report
        report = {
            'mission_summary': self._generate_mission_summary(trajectory_analysis),
            'trajectory_analysis': trajectory_analysis,
            'decision_analysis': decision_analysis,
            'safety_assessment': self._generate_safety_assessment(trajectory_analysis, decision_analysis),
            'performance_metrics': self._compile_performance_metrics(trajectory_analysis),
            'recommendations': self._generate_recommendations(trajectory_analysis, decision_analysis),
            'visualizations': {
                'trajectory_3d': trajectory_plot,
                'decision_heatmap': decision_heatmap
            },
            'report_metadata': {
                'generation_time': datetime.now().isoformat(),
                'trajectory_length': len(trajectory),
                'analysis_version': '1.0'
            }
        }
        
        # Save complete report
        if save_dir:
            report_path = save_dir / "comprehensive_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Comprehensive report saved to {report_path}")
        
        return report
    
    def _generate_mission_summary(self, trajectory_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level mission summary"""
        
        efficiency = trajectory_analysis.get('efficiency', 0.0)
        safety_violations = trajectory_analysis.get('safety_violations', 0)
        fuel_consumption = trajectory_analysis.get('estimated_fuel_consumption', 0.0)
        
        mission_success = efficiency > 0.7 and safety_violations == 0
        
        summary = {
            'mission_success': mission_success,
            'overall_efficiency': efficiency,
            'safety_status': 'SAFE' if safety_violations == 0 else 'VIOLATIONS_DETECTED',
            'fuel_efficiency': 'EXCELLENT' if fuel_consumption < 0.1 else 'MODERATE',
            'key_achievements': [],
            'areas_for_improvement': []
        }
        
        # Add key achievements
        if efficiency > 0.8:
            summary['key_achievements'].append('High trajectory efficiency')
        if safety_violations == 0:
            summary['key_achievements'].append('No safety violations')
        if fuel_consumption < 0.05:
            summary['key_achievements'].append('Excellent fuel efficiency')
        
        # Add improvement areas
        if efficiency < 0.5:
            summary['areas_for_improvement'].append('Improve trajectory planning')
        if safety_violations > 0:
            summary['areas_for_improvement'].append('Enhance safety constraints')
        if fuel_consumption > 0.2:
            summary['areas_for_improvement'].append('Optimize control effort')
        
        return summary
    
    def _generate_safety_assessment(self, trajectory_analysis: Dict[str, Any], 
                                  decision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate safety assessment"""
        
        safety_violations = trajectory_analysis.get('safety_violations', 0)
        min_distance = trajectory_analysis.get('min_distance', float('inf'))
        
        safety_decisions = decision_analysis.get('safety_decisions', {})
        violation_rate = safety_decisions.get('safety_violation_rate', 0.0)
        
        assessment = {
            'overall_safety_rating': 'SAFE' if violation_rate < 0.01 else 'CAUTION',
            'total_violations': safety_violations,
            'minimum_approach_distance': min_distance,
            'violation_rate': violation_rate,
            'safety_recommendations': []
        }
        
        # Generate safety recommendations
        if min_distance < 0.5:
            assessment['safety_recommendations'].append('Increase collision avoidance margin')
        if violation_rate > 0.05:
            assessment['safety_recommendations'].append('Strengthen safety constraints')
        
        return assessment
    
    def _compile_performance_metrics(self, trajectory_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile key performance metrics"""
        
        return {
            'trajectory_efficiency': trajectory_analysis.get('efficiency', 0.0),
            'control_smoothness': trajectory_analysis.get('control_smoothness', 0.0),
            'fuel_consumption': trajectory_analysis.get('estimated_fuel_consumption', 0.0),
            'path_length': trajectory_analysis.get('total_distance', 0.0),
            'approach_phases': trajectory_analysis.get('approach_phases', {}),
            'performance_score': self._calculate_overall_performance_score(trajectory_analysis)
        }
    
    def _calculate_overall_performance_score(self, trajectory_analysis: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        
        efficiency = trajectory_analysis.get('efficiency', 0.0)
        smoothness = trajectory_analysis.get('control_smoothness', 0.0)
        fuel_efficiency = max(0, 1.0 - trajectory_analysis.get('estimated_fuel_consumption', 0.0))
        safety_score = 1.0 if trajectory_analysis.get('safety_violations', 0) == 0 else 0.5
        
        # Weighted combination
        score = (0.3 * efficiency + 0.2 * smoothness + 0.2 * fuel_efficiency + 0.3 * safety_score) * 100
        
        return min(100.0, max(0.0, score))
    
    def _generate_recommendations(self, trajectory_analysis: Dict[str, Any],
                                decision_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        efficiency = trajectory_analysis.get('efficiency', 0.0)
        smoothness = trajectory_analysis.get('control_smoothness', 0.0)
        safety_violations = trajectory_analysis.get('safety_violations', 0)
        
        # Efficiency recommendations
        if efficiency < 0.6:
            recommendations.append("Improve trajectory planning to increase path efficiency")
        
        # Control recommendations
        if smoothness < 0.5:
            recommendations.append("Implement smoother control strategies to reduce fuel consumption")
        
        # Safety recommendations
        if safety_violations > 0:
            recommendations.append("Strengthen safety constraints and emergency procedures")
        
        # Decision quality recommendations
        action_patterns = decision_analysis.get('action_patterns', {})
        if action_patterns.get('thrust_statistics', {}).get('std', 0) > 0.5:
            recommendations.append("Reduce action variability for more consistent performance")
        
        return recommendations