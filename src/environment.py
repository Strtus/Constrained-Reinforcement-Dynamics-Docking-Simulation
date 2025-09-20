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
    """Orbital mechanics utility class for spacecraft dynamics (HCW + perturbations)"""

    def __init__(self,
                 a_km: float = 7450.0,
                 inclination_deg: float = 89.0,
                 enable_j2: bool = True,
                 enable_drag: bool = True):
        """Initialize orbital parameters per mission spec

        Args:
            a_km: Semi-major axis in kilometers (default 7450 km)
            inclination_deg: Inclination [deg]
            enable_j2: Include J2 perturbation
            enable_drag: Include atmospheric drag (high-altitude small)
        """
        self.earth_mu = 3.986004418e14  # [m^3/s^2]
        self.earth_radius = 6.371e6     # [m]
        self.J2 = 1.08262668e-3
        self.a = a_km * 1000.0
        self.inclination = np.deg2rad(inclination_deg)
        # Near-circular
        self.orbital_radius = self.a
        self.mean_motion = np.sqrt(self.earth_mu / self.orbital_radius**3)
        self.orbital_period = 2 * np.pi / self.mean_motion
        # Perturbation flags
        self.enable_j2 = enable_j2
        self.enable_drag = enable_drag
        # Drag model params (very low density at ~500-1100 km)
        self.Cd = 2.2
        self.A_ref = 1.0  # m^2 reference area (placeholder)
        self.rho_500km = 1e-12  # kg/m^3 order of magnitude
        self.rho_1000km = 1e-13

    def _density_exponential(self, altitude_m: float) -> float:
        # Simple piecewise exponential model between 500-1100km
        if altitude_m <= 500e3:
            return self.rho_500km
        if altitude_m >= 1100e3:
            return self.rho_1000km
        # Linear log interpolation in log-space
        f = (altitude_m - 500e3) / (600e3)
        return np.exp(np.log(self.rho_500km) * (1 - f) + np.log(self.rho_1000km) * f)

    def _j2_accel_radial_bias_hill(self, r_hill: np.ndarray) -> np.ndarray:
        """Approximate J2 differential acceleration projected in Hill axes.
        This is a simplified correction (bias scaling with position) sufficient for RL shaping.
        """
        if not self.enable_j2:
            return np.zeros(3)
        r0 = self.orbital_radius
        # Magnitude scale of J2 acceleration at orbit radius
        aJ2_mag = 1.5 * self.J2 * self.earth_mu * (self.earth_radius ** 2) / (r0 ** 4)
        # Project along radial and cross-track per inclination
        x, y, z = r_hill
        # Bias mostly affects radial (x) and cross-track (z)
        return np.array([
            -aJ2_mag * (1 - 5 * np.sin(self.inclination) ** 2) * (x / r0),
            0.0,
            +aJ2_mag * (2 - 5 * np.sin(self.inclination) ** 2) * (z / r0)
        ])

    def _drag_accel_hill(self, vrel_hill: np.ndarray, altitude_m: float, mass: float) -> np.ndarray:
        if not self.enable_drag:
            return np.zeros(3)
        rho = self._density_exponential(altitude_m)
        vmag = np.linalg.norm(vrel_hill)
        if vmag < 1e-6:
            return np.zeros(3)
        a_drag_mag = 0.5 * rho * self.Cd * self.A_ref * vmag / max(mass, 1.0)
        return -a_drag_mag * vrel_hill

    def hill_clohessy_wiltshire_dynamics(self, state: np.ndarray,
                                       thrust_hill: np.ndarray,
                                       mass: float,
                                       altitude_m: float) -> np.ndarray:
        """HCW dynamics with small J2 and drag correction in Hill frame

        state: [x, y, z, vx, vy, vz] in Hill frame
        thrust_hill: [N] in Hill
        mass: kg
        altitude_m: target orbital altitude for drag density eval
        """
        pos = state[:3]
        vel = state[3:6]
        n = self.mean_motion

        # Nominal HCW
        ax = 3 * n ** 2 * pos[0] + 2 * n * vel[1] + thrust_hill[0] / mass
        ay = -2 * n * vel[0] + thrust_hill[1] / mass
        az = -n ** 2 * pos[2] + thrust_hill[2] / mass

        a = np.array([ax, ay, az])
        # Perturbations (approximate)
        a += self._j2_accel_radial_bias_hill(pos)
        a += self._drag_accel_hill(vel, altitude_m, mass)

        return np.array([vel[0], vel[1], vel[2], a[0], a[1], a[2]])


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
    
    def __init__(self, num_thrusters: int = 8, max_thrust: float = 0.02):
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

        # Force allocation (first 3 rows) — 8 thrusters, 2 per +/- axis
        # indices: 0:+X,1:-X,2:+Y,3:-Y,4:+Z,5:-Z,6,7 spares for torque small corrections
        if self.num_thrusters >= 6:
            allocation[0, 0] = +1.0
            allocation[0, 1] = -1.0
            allocation[1, 2] = +1.0
            allocation[1, 3] = -1.0
            allocation[2, 4] = +1.0
            allocation[2, 5] = -1.0

        # Torque allocation (last 3 rows) — small capability via moment arms (0.5 m)
        # Keep very limited torque as attitude is independently controlled
        if self.num_thrusters >= 8:
            allocation[3, 6] = +0.5
            allocation[4, 6] = -0.5
            allocation[5, 7] = +0.5
        
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
        self.los_angle_limit_deg = config.get('los_angle_limit_deg', 30.0)  # ±30° corridor
        
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

        # Line-of-sight corridor (about +Y in Hill frame). Penalize if off-axis > limit
        # Off-axis angle from +Y: angle between position vector and +Y axis
        if distance > 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0])
            cos_theta = np.dot(position / distance, y_axis)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            off_axis_deg = np.degrees(np.arccos(cos_theta))
        else:
            off_axis_deg = 0.0
        if off_axis_deg > self.los_angle_limit_deg:
            violations['los'] = ((off_axis_deg - self.los_angle_limit_deg) / self.los_angle_limit_deg) ** 2
        else:
            violations['los'] = 0.0
        
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        self.max_episode_steps = self.config.get('max_steps', 5400)  # up to 90 min with 1s steps
        self.dt = self.config.get('dt', 1.0)  # [s], adaptive later to 0.1s inside 1 km
        
        # Spacecraft parameters (per spec)
        self.chaser_mass = 200.0  # [kg]
        self.chaser_inertia = np.diag([60.0, 60.0, 40.0])  # [kg*m^2] placeholder
        self.target_mass = 2000.0  # [kg] placeholder for deputy/target
        
        # Propulsion system — Hall thrusters (Krypton)
        self.max_thrust_per_thruster = 0.02  # [N] = 20 mN
        self.specific_impulse = 1385.0  # [s]
        self.min_pulse = 0.1  # [s]
        self.delta_v_budget = self.config.get('delta_v_budget', 150.0)  # m/s (100-200)
        self.num_thrusters = 8
        # Pulse/thermal limits
        self.max_pulses = int(self.config.get('max_pulses', 1000))
        self.pulse_count = 0
        # Thermal power cap (approx), assume efficiency ~0.6
        self.thermal_power_w = float(self.config.get('thermal_power_w', 550.0))
        self.thruster_efficiency = float(self.config.get('thruster_efficiency', 0.6))
        
        # Docking criteria (per spec)
        self.docking_tolerance = 0.1  # [m]
        self.docking_velocity_limit = 0.01  # [m/s]
        self.attitude_tolerance = np.deg2rad(0.5)  # [rad]
        self.hold_time_required = 10.0  # [s]
        self._hold_time_accum = 0.0
        
        # Safety parameters
        safety_config = {
            'collision_radius': 0.1,        # keep-out if not docked
            'max_approach_velocity': 0.5,   # per near-field
            'workspace_boundary': 20000.0,  # 20 km bound
            'max_angular_rate': np.deg2rad(0.5),
            'los_angle_limit_deg': 30.0
        }
        
        # Fault modeling
        self.fault_rate = 1e-6  # lower fault rate
        self.sensor_noise_std = {
            'position': 1.0,                 # [m] GNSS
            'velocity': 0.01,                # [m/s]
            'attitude': np.deg2rad(0.1),     # [rad]
            'angular_rate': np.deg2rad(0.05) # [rad/s]
        }
        self.obs_delay_s = 0.3  # 0.1-0.5 s
        self.actuator_lag_s = 1.0
        self._obs_buffer = []
        self._cmd_buffer = []
        
        # Initialize components
        self.altitude_m = self.config.get('altitude_m', (7450e3 - 6.371e6))
        self.orbital_mechanics = OrbitalMechanics(
            a_km=7450.0,
            inclination_deg=89.0,
            enable_j2=True,
            enable_drag=True
        )
        self.attitude_dynamics = AttitudeDynamics()
        self.thruster_system = ThrusterAllocation(self.num_thrusters, self.max_thrust_per_thruster)
        self.safety_constraints = SafetyConstraints(safety_config)
        
        # Lagrangian multipliers for constrained optimization
        # Constraints: collision, velocity, boundary, angular_rate, los
        self.lagrange_multipliers = np.zeros(5)
        self.constraint_penalty_weight = 1000.0
        
    def _initialize_components(self):
        """Initialize system components"""
        pass  # Components initialized in _initialize_parameters
        
    def _define_spaces(self):
        """Define observation and action spaces"""
        # State: [pos(3), vel(3), quat(4), omega(3), thruster_health(num_thrusters), fuel(1)]
        state_dim = 3 + 3 + 4 + 3 + self.num_thrusters + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Action (normalized): [thrust_x, thrust_y, thrust_z, pulse_duration]
        # thrust in [-1,1] -> [-20,20] mN each; pulse_duration in [-1,1] -> [0.1, 1.0] s
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    
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
        self._hold_time_accum = 0.0
        self._obs_buffer.clear()
        self._cmd_buffer.clear()
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Adaptive time step: 1 s until <1 km, then 0.1 s
        distance_now = np.linalg.norm(getattr(self, 'position', np.zeros(3))) if hasattr(self, 'position') else 1e9
        self.dt = 1.0 if distance_now >= 1000.0 else 0.1

        # Validate and process action (normalized)
        a = np.clip(action, -1.0, 1.0)
        thrust_mN = a[:3] * 20.0  # [-20, 20] mN
        pulse_s = (a[3] + 1.0) * 0.45 + 0.1  # map [-1,1] -> [0.1, 1.0]
        pulse_s = max(self.min_pulse, float(pulse_s))

        # Convert to average thrust over dt (N)
        avg_factor = min(pulse_s / max(self.dt, 1e-6), 1.0)
        thrust_body = (thrust_mN * 1e-3) * avg_factor  # N
        # Thermal power cap -> clamp total thrust magnitude T <= 2*eta*P/ve
        g0 = 9.81
        ve = self.specific_impulse * g0
        t_max_power = max(1e-6, 2.0 * self.thruster_efficiency * self.thermal_power_w / max(ve, 1e-6))
        t_mag = np.linalg.norm(thrust_body)
        if t_mag > t_max_power:
            thrust_body = thrust_body * (t_max_power / t_mag)
        torque_body = np.zeros(3)  # attitude handled independently (assumed)
        
        # Update thruster health (fault injection)
        self._update_thruster_health()
        
        # Allocate thruster commands
        wrench_command = np.concatenate([thrust_body, torque_body])
        effective_thrust, effective_torque = self._apply_thruster_allocation(wrench_command)
        
        # Integrate 6DOF dynamics (with actuator lag as 1st-order buffer)
        self._cmd_buffer.append(thrust_body.copy())
        lag_steps = max(1, int(np.ceil(self.actuator_lag_s / self.dt)))
        if len(self._cmd_buffer) > lag_steps:
            effective_thrust_body = self._cmd_buffer.pop(0)
        else:
            effective_thrust_body = thrust_body

        # Integrate 6DOF dynamics
        self._integrate_dynamics(effective_thrust_body, effective_torque)
        
        # Add sensor noise and observation delay
        self._push_observation_buffer(self._noisy_observation_vector())
        
        # Count pulses if actual command used
        if np.linalg.norm(thrust_body) > 1e-8 and pulse_s >= self.min_pulse:
            self.pulse_count += 1
        
        # Track delta-v used (approximation: |a_cmd| * dt)
        dv_inc = np.linalg.norm(thrust_body) * self.dt / max(self.chaser_mass, 1.0)
        self._dv_used += dv_inc
        
        # Calculate reward with safety constraints
        reward, constraint_costs = self._calculate_constrained_reward(a)
        
        # Update Lagrangian multipliers
        self._update_lagrange_multipliers(constraint_costs)
        
        # Check termination
        terminated, truncated, termination_info = self._check_termination()
        
        # Update metrics
        self._update_episode_metrics()

        # CSV logging (optional)
        if self.enable_csv_logging:
            self._csv_log_step(thrust_body)
        
        # Construct info dictionary
        info = self._construct_info_dict(termination_info, constraint_costs)
        
        self.current_step += 1
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _generate_initial_conditions(self):
        """Generate initial conditions for far-range approach per spec"""
        # Start 10±2 km trailing along-track (y negative), small radial (x) or swap with some probability
        dist_km = np.random.uniform(8.0, 12.0)
        trailing = True if np.random.rand() < 0.7 else False  # 后方优先
        if trailing:
            self.position = np.array([
                np.random.uniform(-1.0e3, 1.0e3),      # radial within ±1 km
                -dist_km * 1000.0,                      # along-track trailing
                np.random.uniform(-800.0, 800.0)        # cross-track < 1 km
            ], dtype=np.float32)
        else:
            self.position = np.array([
                dist_km * 1000.0,                       # radial lead/lag
                np.random.uniform(-1.0e3, 1.0e3),
                np.random.uniform(-800.0, 800.0)
            ], dtype=np.float32)
        
        # Initial velocity bounds per spec
        self.velocity = np.array([
            np.random.uniform(-0.1, 0.1),                        # radial <0.1 m/s
            np.random.uniform(0.0, 0.5) * ( -1.0 if trailing else +1.0),  # along-track 0-0.5 m/s
            np.random.uniform(-0.1, 0.1)                          # cross-track <0.1 m/s
        ], dtype=np.float32)
        
        # Initial attitude ±2°
        misalignment_angle = np.random.uniform(0, np.deg2rad(2.0))
        misalignment_axis = self._random_unit_vector()
        self.attitude_quaternion = self._axis_angle_to_quaternion(
            misalignment_axis, misalignment_angle
        )
        
        # Initial angular velocity
        self.angular_velocity = np.random.uniform(
            -np.deg2rad(1.0), np.deg2rad(1.0), size=3
        ).astype(np.float32)
        
        # Initialize systems
        self.thruster_system.health_status = np.random.uniform(0.85, 1.0, size=self.num_thrusters)
        self.fuel_level = 1.0
        self.active_faults = set()
        self._dv_used = 0.0
        self._init_csv_logger_once()
        self.pulse_count = 0
    
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
                y, thrust_hill, self.chaser_mass, self.altitude_m
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
                
                # Update fuel consumption (rocket equation approximation)
                thrust_magnitude = np.linalg.norm(thrust_body)
                mass_flow_rate = thrust_magnitude / (self.specific_impulse * 9.81)
                self.fuel_level -= (mass_flow_rate * self.dt)
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
            state_trans, thrust_hill, self.chaser_mass, self.altitude_m
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
    
    def _noisy_observation_vector(self) -> np.ndarray:
        """Create a noisy observation vector without corrupting the true state"""
        pos = self.position + np.random.normal(0, self.sensor_noise_std['position'], 3)
        vel = self.velocity + np.random.normal(0, self.sensor_noise_std['velocity'], 3)
        # Attitude noise as small-angle quaternion multiplication (applied to a copy)
        quat = self.attitude_quaternion.copy()
        att_noise_angle = np.random.normal(0, self.sensor_noise_std['attitude'])
        if abs(att_noise_angle) > 1e-8:
            noise_axis = self._random_unit_vector()
            noise_quat = self._axis_angle_to_quaternion(noise_axis, att_noise_angle)
            quat = self._quaternion_multiply(quat, noise_quat)
            quat /= max(1e-8, np.linalg.norm(quat))
        omega = self.angular_velocity + np.random.normal(0, self.sensor_noise_std['angular_rate'], 3)
        obs = np.concatenate([
            pos.astype(np.float32),
            vel.astype(np.float32),
            quat.astype(np.float32),
            omega.astype(np.float32),
            self.thruster_system.health_status.astype(np.float32),
            [np.float32(self.fuel_level)]
        ])
        return obs.astype(np.float32)

    def _push_observation_buffer(self, obs_vec: Optional[np.ndarray] = None):
        # Buffer observations to simulate delay (0.1-0.5 s)
        self._obs_buffer.append(obs_vec if obs_vec is not None else self._raw_observation_vector())
        delay_steps = max(1, int(np.ceil(self.obs_delay_s / max(self.dt, 1e-6))))
        while len(self._obs_buffer) > delay_steps:
            self._obs_buffer.pop(0)
    
    def _calculate_constrained_reward(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """High-fidelity reward with soft constraints per spec"""
        distance = np.linalg.norm(self.position)
        speed = np.linalg.norm(self.velocity)
        att_err = self._calculate_attitude_alignment_error()
        # LOS off-axis angle from +Y axis
        if distance > 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0])
            cos_theta = np.dot(self.position / distance, y_axis)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            los_angle = np.degrees(np.arccos(cos_theta))
        else:
            los_angle = 0.0

        # Weights
        w_dist = 1.0
        w_vel = 0.5
        w_fuel = 0.3
        w_path = 0.2  # V-bar preference
        w_los = 0.2
        w_sense = 0.1

        # Base shaping
        dist_term = -w_dist * (distance / 1000.0)  # scale kilometers
        vel_term = -w_vel * speed
        fuel_term = w_fuel * max(0.0, 1.0 - (1.0 - self.fuel_level))  # remaining fuel proxy

        # V-bar path preference: penalize radial excursions when far
        path_term = -w_path * (abs(self.position[0]) / 1000.0)

        # LOS corridor: penalize being outside ±30° from +Y except very near final (<10 m)
        if distance < 10.0:
            los_term = 0.0
        else:
            los_limit = 30.0
            los_term = -w_los * max(0.0, los_angle - los_limit) / los_limit

        # Sensing term placeholder (e.g., visibility, range quality)
        sensing_term = w_sense * 0.0

        mission_reward = dist_term + vel_term + fuel_term + path_term + los_term + sensing_term

        # Success bonus or penalties
        if self._check_docking_success():
            mission_reward += 1000.0
        # Hard penalties
        terminated, _, reason = self._check_termination(soft_check=True)
        if terminated:
            if reason == 'collision':
                mission_reward -= 100.0
            elif reason == 'timeout':
                mission_reward -= 50.0
        
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

        within = (distance <= self.docking_tolerance and
                  velocity_magnitude <= self.docking_velocity_limit and
                  attitude_error <= self.attitude_tolerance and
                  angular_rate <= np.deg2rad(0.5))
        if within:
            self._hold_time_accum += self.dt
        else:
            self._hold_time_accum = 0.0
        return self._hold_time_accum >= self.hold_time_required
    
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
    
    def _check_termination(self, soft_check: bool = False) -> Tuple[bool, bool, str]:
        """Check termination conditions"""
        distance = np.linalg.norm(self.position)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        # Success termination
        if self._check_docking_success():
            return True, False, "docking_success"
        
        # Failure terminations
        if distance < 0.1:  # Collision
            return True, False, "collision"
        
        if distance > 20000.0:  # Out of bounds (20 km)
            return True, False, "out_of_bounds"
        
        if velocity_magnitude > 5.0:  # Unsafe velocity
            return True, False, "unsafe_velocity"
        
        if self.fuel_level <= 0.0:  # Fuel depletion
            return True, False, "fuel_depletion"

        if self._dv_used >= self.delta_v_budget:  # Delta-v budget exceeded
            return True, False, "dv_budget_exceeded"
        
        if self.pulse_count >= self.max_pulses:  # Pulse count exceeded
            return True, False, "pulse_limit_exceeded"
        
        # Timeout truncation
        if self.current_step >= self.max_episode_steps:
            return (True, False, "timeout") if soft_check else (False, True, "timeout")
        
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
        
        # Map constraint costs to dict (ensure keys match SafetyConstraints)
        constraint_keys = ['collision', 'velocity', 'boundary', 'angular_rate', 'los']
        constraint_dict = {}
        for i, k in enumerate(constraint_keys[:len(constraint_costs)]):
            constraint_dict[k] = constraint_costs[i]

        return {
            'distance_to_target': distance,
            'relative_velocity_magnitude': velocity_magnitude,
            'attitude_error_deg': np.rad2deg(self._calculate_attitude_alignment_error()),
            'angular_rate_deg_s': np.rad2deg(np.linalg.norm(self.angular_velocity)),
            'fuel_remaining': self.fuel_level,
            'active_faults': len(self.active_faults),
            'thruster_health_min': np.min(self.thruster_system.health_status),
            'constraint_violations': constraint_dict,
            'lagrange_multipliers': self.lagrange_multipliers.copy(),
            'termination_reason': termination_info,
            'episode_metrics': self.episode_metrics.copy()
        }
    
    def _raw_observation_vector(self) -> np.ndarray:
        return np.concatenate([
            self.position,                 # 3
            self.velocity,                 # 3
            self.attitude_quaternion,      # 4
            self.angular_velocity,         # 3
            self.thruster_system.health_status,  # 8
            [self.fuel_level]              # 1
        ]).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """Get delayed/noisy observation vector"""
        delay_steps = max(1, int(np.ceil(self.obs_delay_s / max(self.dt, 1e-6))))
        if len(self._obs_buffer) >= delay_steps:
            return self._obs_buffer[-delay_steps]
        return self._raw_observation_vector()
    
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

    # --- CSV logging helpers ---
    def _init_csv_logger_once(self):
        self.enable_csv_logging = bool(self.config.get('enable_csv_logging', False))
        self._csv_initialized = getattr(self, '_csv_initialized', False)
        if not self.enable_csv_logging or self._csv_initialized:
            return
        try:
            import os, csv
            self._csv_path = self.config.get('csv_log_path', 'training_outputs/rvd_trajectory.csv')
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
            # Initialize file with header
            if not os.path.exists(self._csv_path):
                with open(self._csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = [
                        'step','dt','dv_used',
                        'pos_x','pos_y','pos_z',
                        'vel_x','vel_y','vel_z',
                        'quat_w','quat_x','quat_y','quat_z',
                        'omega_x','omega_y','omega_z',
                        'thrust_x','thrust_y','thrust_z'
                    ]
                    writer.writerow(header)
            self._csv_initialized = True
        except Exception:
            self.enable_csv_logging = False  # disable on failure

    def _csv_log_step(self, thrust_body: np.ndarray):
        if not getattr(self, '_csv_initialized', False):
            return
        try:
            import csv
            with open(self._csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_step,
                    self.dt,
                    self._dv_used,
                    *self.position.tolist(),
                    *self.velocity.tolist(),
                    *self.attitude_quaternion.tolist(),
                    *self.angular_velocity.tolist(),
                    *thrust_body.tolist()
                ])
        except Exception:
            pass