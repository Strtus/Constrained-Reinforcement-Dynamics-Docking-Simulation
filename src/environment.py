"""
Advanced Spacecraft Rendezvous and Docking Environment for Safe RL
================================================================

Implementation based on orbital mechanics principles and 
control theory for autonomous spacecraft docking. Implements Hill-Clohessy-
Wiltshire equations, quaternion attitude dynamics, and constrained optimization.

Author: Strtus
License: MIT
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional, List
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class OrbitalMechanics:
    """Orbital mechanics utility class for spacecraft dynamics"""
    
    def __init__(self, altitude_km: float = 400.0):
        """Initialize orbital parameters
        
        Args:
            altitude_km: Orbital altitude in kilometers
        """
        self.earth_mu = 3.986004418e14  # [m^3/s^2] Earth gravitational parameter
        self.earth_radius = 6.371e6     # [m] Earth radius
        self.altitude = altitude_km * 1000.0  # Convert to meters
        self.orbital_radius = self.earth_radius + self.altitude
        self.mean_motion = np.sqrt(self.earth_mu / self.orbital_radius**3)
        self.orbital_period = 2 * np.pi / self.mean_motion
        
    def hill_clohessy_wiltshire_dynamics(self, state: np.ndarray, 
                                       thrust_hill: np.ndarray, 
                                       mass: float) -> np.ndarray:
        """Compute relative orbital dynamics using Hill-Clohessy-Wiltshire equations
        
        Args:
            state: [x, y, z, vx, vy, vz] in Hill frame
            thrust_hill: Thrust vector in Hill frame [N]
            mass: Spacecraft mass [kg]
            
        Returns:
            State derivative [vx, vy, vz, ax, ay, az]
        """
        pos = state[:3]
        vel = state[3:6]
        
        n = self.mean_motion  # Mean motion [rad/s]
        
        # Acceleration components in Hill frame
        # x: radial (Earth-pointing), y: along-track, z: cross-track
        ax = 3*n**2*pos[0] + 2*n*vel[1] + thrust_hill[0]/mass
        ay = -2*n*vel[0] + thrust_hill[1]/mass
        az = -n**2*pos[2] + thrust_hill[2]/mass
        
        return np.array([vel[0], vel[1], vel[2], ax, ay, az])


class AttitudeDynamics:
    """Rigid body attitude dynamics with quaternion representation"""
    
    @staticmethod
    def quaternion_kinematics(quat: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Compute quaternion time derivative
        
        Args:
            quat: Quaternion [w, x, y, z]
            omega: Angular velocity [rad/s]
            
        Returns:
            Quaternion derivative
        """
        # Quaternion kinematic equation: q_dot = 0.5 * Omega * q
        # where Omega is the skew-symmetric matrix of omega
        w, x, y, z = quat
        wx, wy, wz = omega
        
        q_dot = 0.5 * np.array([
            -x*wx - y*wy - z*wz,
            w*wx + y*wz - z*wy,
            w*wy - x*wz + z*wx,
            w*wz + x*wy - y*wx
        ])
        
        return q_dot
    
    @staticmethod
    def euler_equations(omega: np.ndarray, torque: np.ndarray, 
                       inertia: np.ndarray) -> np.ndarray:
        """Compute angular acceleration using Euler's equations
        
        Args:
            omega: Angular velocity [rad/s]
            torque: Applied torque [N*m]
            inertia: Inertia matrix [kg*m^2]
            
        Returns:
            Angular acceleration [rad/s^2]
        """
        # Euler's equation: I*omega_dot + omega x (I*omega) = torque
        H = inertia @ omega  # Angular momentum
        omega_cross_H = np.cross(omega, H)
        omega_dot = np.linalg.solve(inertia, torque - omega_cross_H)
        
        return omega_dot
    
    @staticmethod
    def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = quat
        
        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        
        return R


class ThrusterAllocation:
    """Thruster allocation and fault management system"""
    
    def __init__(self, num_thrusters: int = 12, max_thrust: float = 22.0):
        """Initialize thruster configuration
        
        Args:
            num_thrusters: Number of thrusters
            max_thrust: Maximum thrust per thruster [N]
        """
        self.num_thrusters = num_thrusters
        self.max_thrust = max_thrust
        
        # Generate thruster configuration matrix (3 forces + 3 torques)
        self.allocation_matrix = self._generate_allocation_matrix()
        self.health_status = np.ones(num_thrusters)
        
    def _generate_allocation_matrix(self) -> np.ndarray:
        """Generate 6x12 thruster allocation matrix for forces and torques"""
        # Simplified 4x3 thruster configuration
        # 4 thrusters per axis (X, Y, Z) with moment arms for torque generation
        allocation = np.zeros((6, self.num_thrusters))
        
        # Force allocation (first 3 rows)
        allocation[0, 0:4] = 1.0   # +X force thrusters
        allocation[1, 4:8] = 1.0   # +Y force thrusters  
        allocation[2, 8:12] = 1.0  # +Z force thrusters
        
        # Torque allocation (last 3 rows) - moment arms of 1m
        allocation[3, 4:8] = np.array([1.0, -1.0, 1.0, -1.0])  # Roll torque
        allocation[4, 8:12] = np.array([1.0, -1.0, 1.0, -1.0]) # Pitch torque
        allocation[5, 0:4] = np.array([1.0, -1.0, 1.0, -1.0])  # Yaw torque
        
        return allocation
    
    def allocate_commands(self, desired_wrench: np.ndarray) -> np.ndarray:
        """Allocate desired forces/torques to individual thrusters
        
        Args:
            desired_wrench: [fx, fy, fz, tx, ty, tz]
            
        Returns:
            Thruster commands [12x1]
        """
        # Apply health degradation
        effective_allocation = self.allocation_matrix * self.health_status
        
        # Pseudo-inverse allocation with constraints
        try:
            thruster_commands = np.linalg.pinv(effective_allocation) @ desired_wrench
            
            # Apply thrust limits
            thruster_commands = np.clip(thruster_commands, 0, self.max_thrust)
            
            return thruster_commands
            
        except np.linalg.LinAlgError:
            # Fallback to simplified allocation
            return self._fallback_allocation(desired_wrench)
    
    def _fallback_allocation(self, desired_wrench: np.ndarray) -> np.ndarray:
        """Simplified fallback allocation method"""
        commands = np.zeros(self.num_thrusters)
        
        # Direct force mapping
        for i in range(3):
            thrust_per_thruster = max(0, desired_wrench[i]) / 4.0
            start_idx = i * 4
            commands[start_idx:start_idx+4] = thrust_per_thruster
            
        return np.clip(commands, 0, self.max_thrust)
    
    def inject_fault(self, thruster_id: int, degradation_factor: float = 0.5):
        """Inject thruster fault"""
        if 0 <= thruster_id < self.num_thrusters:
            self.health_status[thruster_id] = degradation_factor


class SafetyConstraints:
    """Safety constraint evaluation and enforcement"""
    
    def __init__(self, config: Dict[str, float]):
        """Initialize safety parameters"""
        self.collision_radius = config.get('collision_radius', 0.5)
        self.max_approach_velocity = config.get('max_approach_velocity', 2.0)
        self.workspace_boundary = config.get('workspace_boundary', 200.0)
        self.max_angular_rate = config.get('max_angular_rate', np.deg2rad(5.0))
        
    def evaluate_constraints(self, state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate all safety constraints
        
        Args:
            state: Current spacecraft state
            
        Returns:
            Dictionary of constraint violations
        """
        position = state['position']
        velocity = state['velocity']
        angular_velocity = state['angular_velocity']
        
        distance = np.linalg.norm(position)
        speed = np.linalg.norm(velocity)
        angular_speed = np.linalg.norm(angular_velocity)
        
        violations = {}
        
        # Collision avoidance
        if distance < self.collision_radius:
            violations['collision'] = (self.collision_radius - distance)**2
        else:
            violations['collision'] = 0.0
        
        # Velocity constraint
        if speed > self.max_approach_velocity:
            violations['velocity'] = (speed - self.max_approach_velocity)**2
        else:
            violations['velocity'] = 0.0
        
        # Workspace boundary
        if distance > self.workspace_boundary:
            violations['boundary'] = (distance - self.workspace_boundary)**2
        else:
            violations['boundary'] = 0.0
        
        # Angular rate constraint
        if angular_speed > self.max_angular_rate:
            violations['angular_rate'] = (angular_speed - self.max_angular_rate)**2
        else:
            violations['angular_rate'] = 0.0
        
        return violations


class SpacecraftRvDEnvironment(gym.Env):
    """
    Professional Spacecraft Rendezvous and Docking Environment
    
    Implements high-fidelity 6DOF dynamics with:
    - Hill-Clohessy-Wiltshire relative orbital mechanics
    - Quaternion-based attitude dynamics  
    - Realistic thruster allocation and fault modeling
    - Safety-critical constraints for autonomous docking
    - Constrained optimization for safe RL
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the spacecraft docking environment
        
        Args:
            config: Environment configuration dictionary
        """
        super().__init__()
        
        # Load configuration
        self.config = config or {}
        self._initialize_parameters()
        self._initialize_components()
        self._define_spaces()
        
        # Episode state
        self.current_step = 0
        self.episode_metrics = {}
        
        # Initialize environment
        self.reset()
    
    def _initialize_parameters(self):
        """Initialize all environment parameters"""
        # Mission parameters
        self.max_episode_steps = self.config.get('max_steps', 1200)
        self.dt = self.config.get('dt', 0.5)  # [s]
        
        # Spacecraft parameters
        self.chaser_mass = 750.0  # [kg]
        self.chaser_inertia = np.diag([125.0, 150.0, 100.0])  # [kg*m^2]
        self.target_mass = 15000.0  # [kg]
        
        # Propulsion system
        self.max_thrust_per_thruster = 22.0  # [N]
        self.specific_impulse = 230.0  # [s]
        self.num_thrusters = 12
        
        # Docking criteria (based on ISS requirements)
        self.docking_tolerance = 0.05  # [m]
        self.docking_velocity_limit = 0.1  # [m/s]
        self.attitude_tolerance = np.deg2rad(2.0)  # [rad]
        
        # Safety parameters
        safety_config = {
            'collision_radius': 0.5,
            'max_approach_velocity': 2.0,
            'workspace_boundary': 200.0,
            'max_angular_rate': np.deg2rad(5.0)
        }
        
        # Fault modeling
        self.fault_rate = 2.5e-5  # [faults/s]
        self.sensor_noise_std = {
            'position': 0.01,      # [m]
            'velocity': 0.005,     # [m/s]
            'attitude': np.deg2rad(0.1),  # [rad]
            'angular_rate': np.deg2rad(0.05)  # [rad/s]
        }
        
        # Initialize components
        self.orbital_mechanics = OrbitalMechanics(altitude_km=400.0)
        self.attitude_dynamics = AttitudeDynamics()
        self.thruster_system = ThrusterAllocation(self.num_thrusters, self.max_thrust_per_thruster)
        self.safety_constraints = SafetyConstraints(safety_config)
        
        # Lagrangian multipliers for constrained optimization
        self.lagrange_multipliers = np.zeros(4)
        self.constraint_penalty_weight = 1000.0
        
    def _initialize_components(self):
        """Initialize system components"""
        pass  # Components initialized in _initialize_parameters
        
    def _define_spaces(self):
        """Define observation and action spaces"""
        # State: [pos(3), vel(3), quat(4), omega(3), thruster_health(12), fuel(1)]
        state_dim = 26
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Action: [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
        # Normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial conditions"""
        super().reset(seed=seed)
        
        # Generate initial conditions for terminal approach phase
        self._generate_initial_conditions()
        
        # Reset episode variables
        self.current_step = 0
        self.episode_metrics = {
            'fuel_consumption': 0.0,
            'constraint_violations': 0,
            'max_approach_velocity': 0.0,
            'min_distance': np.linalg.norm(self.position),
            'docking_accuracy': np.inf
        }
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Validate and process action
        action = np.clip(action, -1.0, 1.0)
        
        # Convert normalized action to physical commands
        thrust_body = action[:3] * self.max_thrust_per_thruster * 4  # 4 thrusters per axis
        torque_body = action[3:] * 5.0  # [N*m] maximum control torque
        
        # Update thruster health (fault injection)
        self._update_thruster_health()
        
        # Allocate thruster commands
        wrench_command = np.concatenate([thrust_body, torque_body])
        effective_thrust, effective_torque = self._apply_thruster_allocation(wrench_command)
        
        # Integrate 6DOF dynamics
        self._integrate_dynamics(effective_thrust, effective_torque)
        
        # Add sensor noise
        self._apply_sensor_noise()
        
        # Calculate reward with safety constraints
        reward, constraint_costs = self._calculate_constrained_reward(action)
        
        # Update Lagrangian multipliers
        self._update_lagrange_multipliers(constraint_costs)
        
        # Check termination
        terminated, truncated, termination_info = self._check_termination()
        
        # Update metrics
        self._update_episode_metrics()
        
        # Construct info dictionary
        info = self._construct_info_dict(termination_info, constraint_costs)
        
        self.current_step += 1
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _generate_initial_conditions(self):
        """Generate realistic initial conditions for terminal approach"""
        # Initial position (50-150m range in approach corridor)
        approach_distance = np.random.uniform(50.0, 150.0)
        approach_angle = np.random.uniform(-np.pi/6, np.pi/6)
        
        self.position = np.array([
            -approach_distance * np.cos(approach_angle),  # Anti-radial approach
            np.random.uniform(-10.0, 10.0),               # Along-track offset
            np.random.uniform(-5.0, 5.0)                  # Cross-track offset
        ], dtype=np.float32)
        
        # Initial velocity (controlled approach)
        v_approach = np.random.uniform(0.05, 0.2)
        self.velocity = np.array([
            v_approach * np.cos(approach_angle),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.02, 0.02)
        ], dtype=np.float32)
        
        # Initial attitude (small misalignment)
        misalignment_angle = np.random.uniform(0, np.deg2rad(10.0))
        misalignment_axis = self._random_unit_vector()
        self.attitude_quaternion = self._axis_angle_to_quaternion(
            misalignment_axis, misalignment_angle
        )
        
        # Initial angular velocity
        self.angular_velocity = np.random.uniform(
            -np.deg2rad(1.0), np.deg2rad(1.0), size=3
        ).astype(np.float32)
        
        # Initialize systems
        self.thruster_system.health_status = np.random.uniform(0.85, 1.0, size=12)
        self.fuel_level = 1.0
        self.active_faults = set()
    
    def _integrate_dynamics(self, thrust_body: np.ndarray, torque_body: np.ndarray):
        """Integrate 6DOF spacecraft dynamics"""
        # Current state
        state_trans = np.concatenate([self.position, self.velocity])
        state_rot = np.concatenate([self.attitude_quaternion, self.angular_velocity])
        
        # Transform thrust to Hill frame
        R_body_to_hill = self.attitude_dynamics.quaternion_to_rotation_matrix(
            self.attitude_quaternion
        )
        thrust_hill = R_body_to_hill @ thrust_body
        
        # Integrate translational dynamics
        def trans_dynamics(t, y):
            return self.orbital_mechanics.hill_clohessy_wiltshire_dynamics(
                y, thrust_hill, self.chaser_mass
            )
        
        # Integrate rotational dynamics  
        def rot_dynamics(t, y):
            quat = y[:4]
            omega = y[4:]
            
            q_dot = self.attitude_dynamics.quaternion_kinematics(quat, omega)
            omega_dot = self.attitude_dynamics.euler_equations(
                omega, torque_body, self.chaser_inertia
            )
            
            return np.concatenate([q_dot, omega_dot])
        
        # Solve ODEs
        try:
            # Translational motion
            sol_trans = solve_ivp(trans_dynamics, [0, self.dt], state_trans,
                                method='RK45', rtol=1e-8, atol=1e-10)
            
            # Rotational motion
            sol_rot = solve_ivp(rot_dynamics, [0, self.dt], state_rot,
                              method='RK45', rtol=1e-8, atol=1e-10)
            
            if sol_trans.success and sol_rot.success:
                # Update states
                final_trans = sol_trans.y[:, -1]
                final_rot = sol_rot.y[:, -1]
                
                self.position = final_trans[:3].astype(np.float32)
                self.velocity = final_trans[3:].astype(np.float32)
                self.attitude_quaternion = final_rot[:4].astype(np.float32)
                self.angular_velocity = final_rot[4:].astype(np.float32)
                
                # Normalize quaternion
                self.attitude_quaternion /= np.linalg.norm(self.attitude_quaternion)
                
                # Update fuel consumption
                thrust_magnitude = np.linalg.norm(thrust_body)
                mass_flow_rate = thrust_magnitude / (self.specific_impulse * 9.81)
                self.fuel_level -= (mass_flow_rate * self.dt) / self.chaser_mass
                self.fuel_level = max(0.0, self.fuel_level)
                
            else:
                # Fallback to Euler integration
                self._euler_integration_fallback(thrust_body, torque_body)
                
        except Exception as e:
            warnings.warn(f"Integration failed: {e}, using fallback")
            self._euler_integration_fallback(thrust_body, torque_body)
    
    def _euler_integration_fallback(self, thrust_body: np.ndarray, torque_body: np.ndarray):
        """Simple Euler integration fallback"""
        # Transform thrust to Hill frame
        R_body_to_hill = self.attitude_dynamics.quaternion_to_rotation_matrix(
            self.attitude_quaternion
        )
        thrust_hill = R_body_to_hill @ thrust_body
        
        # Translational dynamics
        state_trans = np.concatenate([self.position, self.velocity])
        trans_deriv = self.orbital_mechanics.hill_clohessy_wiltshire_dynamics(
            state_trans, thrust_hill, self.chaser_mass
        )
        
        self.position += trans_deriv[:3] * self.dt
        self.velocity += trans_deriv[3:] * self.dt
        
        # Rotational dynamics
        q_dot = self.attitude_dynamics.quaternion_kinematics(
            self.attitude_quaternion, self.angular_velocity
        )
        omega_dot = self.attitude_dynamics.euler_equations(
            self.angular_velocity, torque_body, self.chaser_inertia
        )
        
        self.attitude_quaternion += q_dot * self.dt
        self.angular_velocity += omega_dot * self.dt
        
        # Normalize quaternion
        self.attitude_quaternion /= np.linalg.norm(self.attitude_quaternion)
    
    def _apply_thruster_allocation(self, wrench_command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply thruster allocation with fault handling"""
        thruster_commands = self.thruster_system.allocate_commands(wrench_command)
        
        # Calculate actual force/torque from thruster commands
        actual_wrench = self.thruster_system.allocation_matrix @ thruster_commands
        
        return actual_wrench[:3], actual_wrench[3:]
    
    def _update_thruster_health(self):
        """Update thruster health status with stochastic fault injection"""
        fault_probability = self.fault_rate * self.dt
        
        for i in range(self.num_thrusters):
            if np.random.random() < fault_probability:
                if i not in self.active_faults:
                    degradation = np.random.uniform(0.3, 0.8)
                    self.thruster_system.inject_fault(i, degradation)
                    self.active_faults.add(i)
    
    def _apply_sensor_noise(self):
        """Add realistic sensor noise to state measurements"""
        # Position noise (GPS/relative navigation)
        pos_noise = np.random.normal(0, self.sensor_noise_std['position'], 3)
        self.position += pos_noise.astype(np.float32)
        
        # Velocity noise (Doppler measurements)
        vel_noise = np.random.normal(0, self.sensor_noise_std['velocity'], 3)
        self.velocity += vel_noise.astype(np.float32)
        
        # Attitude noise (star tracker)
        att_noise_angle = np.random.normal(0, self.sensor_noise_std['attitude'])
        if abs(att_noise_angle) > 1e-6:
            noise_axis = self._random_unit_vector()
            noise_quat = self._axis_angle_to_quaternion(noise_axis, att_noise_angle)
            self.attitude_quaternion = self._quaternion_multiply(
                self.attitude_quaternion, noise_quat
            )
            self.attitude_quaternion /= np.linalg.norm(self.attitude_quaternion)
        
        # Angular rate noise (gyroscope drift)
        omega_noise = np.random.normal(0, self.sensor_noise_std['angular_rate'], 3)
        self.angular_velocity += omega_noise.astype(np.float32)
    
    def _calculate_constrained_reward(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate reward with safety constraints using augmented Lagrangian"""
        distance = np.linalg.norm(self.position)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        # Primary mission reward
        if self._check_docking_success():
            mission_reward = 1000.0
        else:
            # Progress-based reward shaping
            distance_reward = -0.5 * distance
            velocity_reward = -2.0 * velocity_magnitude if distance < 10.0 else -0.1 * velocity_magnitude
            
            # Attitude alignment reward
            attitude_error = self._calculate_attitude_alignment_error()
            attitude_reward = -10.0 * attitude_error
            
            # Fuel efficiency
            thrust_penalty = -0.1 * np.linalg.norm(action[:3])
            
            mission_reward = distance_reward + velocity_reward + attitude_reward + thrust_penalty
        
        # Evaluate safety constraints
        state_dict = {
            'position': self.position,
            'velocity': self.velocity,
            'angular_velocity': self.angular_velocity
        }
        constraint_violations = self.safety_constraints.evaluate_constraints(state_dict)
        constraint_costs = np.array(list(constraint_violations.values()))
        
        # Augmented Lagrangian penalty
        constraint_penalty = np.sum(self.lagrange_multipliers * constraint_costs)
        constraint_penalty += 0.5 * self.constraint_penalty_weight * np.sum(constraint_costs**2)
        
        total_reward = mission_reward - constraint_penalty
        
        return total_reward, constraint_costs
    
    def _update_lagrange_multipliers(self, constraint_costs: np.ndarray):
        """Update Lagrangian multipliers for constraint satisfaction"""
        learning_rate = 0.01
        self.lagrange_multipliers += learning_rate * constraint_costs
        self.lagrange_multipliers = np.maximum(0, self.lagrange_multipliers)  # Keep non-negative
    
    def _check_docking_success(self) -> bool:
        """Check if docking success criteria are met"""
        distance = np.linalg.norm(self.position)
        velocity_magnitude = np.linalg.norm(self.velocity)
        attitude_error = self._calculate_attitude_alignment_error()
        angular_rate = np.linalg.norm(self.angular_velocity)
        
        return (distance <= self.docking_tolerance and
                velocity_magnitude <= self.docking_velocity_limit and
                attitude_error <= self.attitude_tolerance and
                angular_rate <= np.deg2rad(0.5))
    
    def _calculate_attitude_alignment_error(self) -> float:
        """Calculate attitude alignment error from target-pointing"""
        # Target-pointing quaternion (identity)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Calculate relative quaternion
        error_quat = self._quaternion_multiply(
            self._quaternion_conjugate(target_quat), self.attitude_quaternion
        )
        
        # Convert to angle (smallest angle representation)
        angle_error = 2 * np.arccos(np.abs(error_quat[0]))
        return min(angle_error, 2*np.pi - angle_error)
    
    def _check_termination(self) -> Tuple[bool, bool, str]:
        """Check termination conditions"""
        distance = np.linalg.norm(self.position)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        # Success termination
        if self._check_docking_success():
            return True, False, "docking_success"
        
        # Failure terminations
        if distance < 0.1:  # Collision
            return True, False, "collision"
        
        if distance > 200.0:  # Out of bounds
            return True, False, "out_of_bounds"
        
        if velocity_magnitude > 5.0:  # Unsafe velocity
            return True, False, "unsafe_velocity"
        
        if self.fuel_level <= 0.0:  # Fuel depletion
            return True, False, "fuel_depletion"
        
        # Timeout truncation
        if self.current_step >= self.max_episode_steps:
            return False, True, "timeout"
        
        return False, False, "continuing"
    
    def _update_episode_metrics(self):
        """Update episode performance metrics"""
        distance = np.linalg.norm(self.position)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        self.episode_metrics['fuel_consumption'] = 1.0 - self.fuel_level
        self.episode_metrics['max_approach_velocity'] = max(
            self.episode_metrics['max_approach_velocity'], velocity_magnitude
        )
        self.episode_metrics['min_distance'] = min(
            self.episode_metrics['min_distance'], distance
        )
        
        if distance <= self.docking_tolerance:
            self.episode_metrics['docking_accuracy'] = distance
    
    def _construct_info_dict(self, termination_info: str, constraint_costs: np.ndarray) -> Dict:
        """Construct comprehensive info dictionary"""
        distance = np.linalg.norm(self.position)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        return {
            'distance_to_target': distance,
            'relative_velocity_magnitude': velocity_magnitude,
            'attitude_error_deg': np.rad2deg(self._calculate_attitude_alignment_error()),
            'angular_rate_deg_s': np.rad2deg(np.linalg.norm(self.angular_velocity)),
            'fuel_remaining': self.fuel_level,
            'active_faults': len(self.active_faults),
            'thruster_health_min': np.min(self.thruster_system.health_status),
            'constraint_violations': {
                'collision': constraint_costs[0],
                'velocity': constraint_costs[1], 
                'boundary': constraint_costs[2],
                'angular_rate': constraint_costs[3]
            },
            'lagrange_multipliers': self.lagrange_multipliers.copy(),
            'termination_reason': termination_info,
            'episode_metrics': self.episode_metrics.copy()
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        obs = np.concatenate([
            self.position,                              # 3
            self.velocity,                              # 3  
            self.attitude_quaternion,                   # 4
            self.angular_velocity,                      # 3
            self.thruster_system.health_status,        # 12
            [self.fuel_level]                          # 1
        ])  # Total: 26 dimensions
        
        return obs.astype(np.float32)
    
    # Utility methods
    def _random_unit_vector(self) -> np.ndarray:
        """Generate random unit vector"""
        vec = np.random.randn(3)
        return vec / np.linalg.norm(vec)
    
    def _axis_angle_to_quaternion(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion"""
        axis_normalized = axis / np.linalg.norm(axis)
        w = np.cos(angle / 2)
        xyz = axis_normalized * np.sin(angle / 2)
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Calculate quaternion conjugate"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def render(self, mode: str = 'human'):
        """Render environment state"""
        if mode == 'human':
            distance = np.linalg.norm(self.position)
            velocity = np.linalg.norm(self.velocity)
            attitude_error = np.rad2deg(self._calculate_attitude_alignment_error())
            
            print(f"Step: {self.current_step}")
            print(f"Distance: {distance:.3f}m")
            print(f"Velocity: {velocity:.3f}m/s") 
            print(f"Attitude Error: {attitude_error:.1f}°")
            print(f"Fuel: {self.fuel_level:.2%}")
            print(f"Active Faults: {len(self.active_faults)}")
            print("-" * 30)