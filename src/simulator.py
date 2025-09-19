"""
Advanced Spacecraft Dynamics Simulator for Safe RL
=================================================

Implementation of high-fidelity 6DOF spacecraft dynamics
including orbital mechanics, attitude control, propulsion systems, and 
fault modeling for autonomous rendezvous and docking missions.

Author: Strtus
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve
import warnings
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThrusterType(Enum):
    """Thruster types with different characteristics"""
    HYDRAZINE = "hydrazine"
    COLD_GAS = "cold_gas"
    ELECTRIC = "electric"
    CHEMICAL = "chemical"


class FaultType(Enum):
    """Spacecraft fault types"""
    THRUSTER_DEGRADATION = "thruster_degradation"
    THRUSTER_FAILURE = "thruster_failure"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_NOISE = "sensor_noise"
    ATTITUDE_SENSOR_FAILURE = "attitude_sensor_failure"
    NAVIGATION_DRIFT = "navigation_drift"


@dataclass
class SpacecraftProperties:
    """Physical properties of spacecraft"""
    mass: float  # [kg] 
    inertia_matrix: np.ndarray  # [kg*m^2] 3x3 inertia tensor
    geometry: Dict[str, float]  # Geometric properties
    drag_coefficient: float = 2.2  # Typical for spacecraft
    cross_sectional_area: float = 1.0  # [m^2]
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [m]
    center_of_pressure: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [m]


@dataclass
class PropulsionSystem:
    """Spacecraft propulsion system specifications"""
    thruster_type: ThrusterType
    num_thrusters: int
    max_thrust_per_thruster: float  # [N]
    specific_impulse: float  # [s]
    minimum_impulse_bit: float  # [N*s]
    thruster_positions: np.ndarray  # [m] positions relative to CM
    thruster_directions: np.ndarray  # unit vectors
    valve_response_time: float = 0.01  # [s]
    thrust_rise_time: float = 0.005  # [s]
    thrust_decay_time: float = 0.005  # [s]


@dataclass
class SensorSuite:
    """Spacecraft sensor system specifications"""
    gps_accuracy: float = 0.01  # [m] position accuracy
    gps_velocity_accuracy: float = 0.005  # [m/s] velocity accuracy
    star_tracker_accuracy: float = np.deg2rad(0.1)  # [rad] attitude accuracy
    gyroscope_bias_stability: float = np.deg2rad(0.01)  # [rad/s] angular rate bias
    gyroscope_noise: float = np.deg2rad(0.05)  # [rad/s] angular rate noise
    accelerometer_bias: float = 1e-6  # [m/s^2] acceleration bias
    range_sensor_accuracy: float = 0.001  # [m] relative range accuracy
    update_frequency: float = 10.0  # [Hz] sensor update rate


@dataclass 
class OrbitalEnvironment:
    """Orbital environment parameters"""
    altitude: float  # [m] orbital altitude
    inclination: float = 0.0  # [rad] orbital inclination
    eccentricity: float = 0.0  # orbital eccentricity
    earth_j2: float = 1.08262668e-3  # J2 perturbation coefficient
    atmospheric_density: float = 1e-12  # [kg/m^3] atmospheric density
    solar_pressure: float = 4.56e-6  # [N/m^2] solar radiation pressure


class OrbitalMechanicsEngine:
    """Advanced orbital mechanics computations"""
    
    def __init__(self, environment: OrbitalEnvironment):
        self.env = environment
        self.earth_mu = 3.986004418e14  # [m^3/s^2]
        self.earth_radius = 6.371e6  # [m]
        self.orbital_radius = self.earth_radius + environment.altitude
        self.mean_motion = np.sqrt(self.earth_mu / self.orbital_radius**3)
        
    def hill_clohessy_wiltshire_full(self, state: np.ndarray, 
                                   external_forces: np.ndarray,
                                   mass: float,
                                   include_j2: bool = True) -> np.ndarray:
        """
        Enhanced Hill-Clohessy-Wiltshire equations with J2 perturbations
        
        Args:
            state: [x, y, z, vx, vy, vz] in LVLH frame
            external_forces: External forces in LVLH frame [N]
            mass: Spacecraft mass [kg]
            include_j2: Include J2 gravitational perturbations
            
        Returns:
            State derivative [vx, vy, vz, ax, ay, az]
        """
        pos = state[:3]
        vel = state[3:6]
        
        x, y, z = pos
        vx, vy, vz = vel
        n = self.mean_motion
        
        # Standard Clohessy-Wiltshire accelerations
        ax_cw = 3*n**2*x + 2*n*vy + external_forces[0]/mass
        ay_cw = -2*n*vx + external_forces[1]/mass
        az_cw = -n**2*z + external_forces[2]/mass
        
        # J2 perturbation effects (if enabled)
        if include_j2:
            j2_accel = self._compute_j2_acceleration(pos)
            ax_cw += j2_accel[0]
            ay_cw += j2_accel[1] 
            az_cw += j2_accel[2]
        
        return np.array([vx, vy, vz, ax_cw, ay_cw, az_cw])
    
    def _compute_j2_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Compute J2 gravitational perturbation acceleration"""
        x, y, z = position
        
        # Convert to ECI frame (simplified)
        r_eci = np.array([self.orbital_radius + x, y, z])
        r_mag = np.linalg.norm(r_eci)
        
        if r_mag < 1e-6:
            return np.zeros(3)
        
        # J2 acceleration in ECI
        j2_factor = -1.5 * self.env.earth_j2 * self.earth_mu * (self.earth_radius**2) / (r_mag**5)
        
        ax_j2 = j2_factor * r_eci[0] * (5*(r_eci[2]**2)/(r_mag**2) - 1)
        ay_j2 = j2_factor * r_eci[1] * (5*(r_eci[2]**2)/(r_mag**2) - 1)
        az_j2 = j2_factor * r_eci[2] * (5*(r_eci[2]**2)/(r_mag**2) - 3)
        
        # Transform back to LVLH (simplified as identity for small relative motion)
        return np.array([ax_j2, ay_j2, az_j2]) * 1e-6  # Scale for relative motion
    
    def atmospheric_drag_force(self, velocity: np.ndarray, 
                              properties: SpacecraftProperties) -> np.ndarray:
        """Compute atmospheric drag force"""
        if self.env.atmospheric_density <= 0:
            return np.zeros(3)
        
        v_mag = np.linalg.norm(velocity)
        if v_mag < 1e-6:
            return np.zeros(3)
        
        # Drag force magnitude
        drag_magnitude = 0.5 * self.env.atmospheric_density * v_mag**2 * \
                        properties.drag_coefficient * properties.cross_sectional_area
        
        # Drag force direction (opposite to velocity)
        drag_direction = -velocity / v_mag
        
        return drag_magnitude * drag_direction
    
    def solar_radiation_pressure_force(self, sun_vector: np.ndarray,
                                     properties: SpacecraftProperties,
                                     eclipse_factor: float = 1.0) -> np.ndarray:
        """Compute solar radiation pressure force"""
        if np.linalg.norm(sun_vector) < 1e-6:
            return np.zeros(3)
        
        sun_unit = sun_vector / np.linalg.norm(sun_vector)
        
        # Solar pressure force (simplified model)
        pressure_magnitude = self.env.solar_pressure * properties.cross_sectional_area * eclipse_factor
        
        return pressure_magnitude * sun_unit


class AttitudeDynamicsEngine:
    """Advanced attitude dynamics with environmental torques"""
    
    def __init__(self, properties: SpacecraftProperties, environment: OrbitalEnvironment):
        self.properties = properties
        self.environment = environment
        self.earth_mu = 3.986004418e14
        self.orbital_radius = 6.371e6 + environment.altitude
        
    def rigid_body_dynamics(self, attitude_state: np.ndarray,
                           control_torque: np.ndarray,
                           include_disturbances: bool = True) -> np.ndarray:
        """
        Complete rigid body attitude dynamics with environmental disturbances
        
        Args:
            attitude_state: [q0, q1, q2, q3, wx, wy, wz] quaternion + angular velocity
            control_torque: Applied control torque [N*m]
            include_disturbances: Include environmental disturbance torques
            
        Returns:
            State derivative [q_dot, omega_dot]
        """
        quat = attitude_state[:4]
        omega = attitude_state[4:7]
        
        # Quaternion kinematics
        q_dot = 0.5 * self._quaternion_kinematics_matrix(quat) @ np.concatenate([[0], omega])
        
        # Total torque
        total_torque = control_torque.copy()
        
        if include_disturbances:
            # Gravity gradient torque
            total_torque += self._gravity_gradient_torque(quat)
            
            # Atmospheric drag torque
            total_torque += self._atmospheric_drag_torque(quat, omega)
            
            # Solar radiation pressure torque
            total_torque += self._solar_pressure_torque(quat)
        
        # Euler's equations for rigid body rotation
        inertia = self.properties.inertia_matrix
        omega_cross_H = np.cross(omega, inertia @ omega)
        omega_dot = np.linalg.solve(inertia, total_torque - omega_cross_H)
        
        return np.concatenate([q_dot, omega_dot])
    
    def _quaternion_kinematics_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Quaternion kinematics matrix"""
        w, x, y, z = quat
        return np.array([
            [-x, -y, -z],
            [w, -z, y],
            [z, w, -x],
            [-y, x, w]
        ])
    
    def _gravity_gradient_torque(self, quat: np.ndarray) -> np.ndarray:
        """Compute gravity gradient torque"""
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(quat)
        
        # Local vertical direction in body frame
        r_local = R.T @ np.array([1, 0, 0])  # Assuming x-axis points to Earth
        
        # Gravity gradient torque
        n_orbital = 3 * self.earth_mu / (self.orbital_radius**3)
        torque = n_orbital * np.cross(r_local, self.properties.inertia_matrix @ r_local)
        
        return torque
    
    def _atmospheric_drag_torque(self, quat: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Compute atmospheric drag torque"""
        if self.environment.atmospheric_density <= 0:
            return np.zeros(3)
        
        # Relative velocity in body frame (simplified)
        v_rel = np.array([0, 7800, 0])  # Approximate orbital velocity
        
        # Torque arm from center of mass to center of pressure
        torque_arm = self.properties.center_of_pressure - self.properties.center_of_mass
        
        # Drag force
        drag_magnitude = 0.5 * self.environment.atmospheric_density * np.linalg.norm(v_rel)**2 * \
                        self.properties.drag_coefficient * self.properties.cross_sectional_area
        
        drag_force = -drag_magnitude * v_rel / np.linalg.norm(v_rel)
        
        # Torque due to drag
        torque = np.cross(torque_arm, drag_force)
        
        return torque * 1e-6  # Scale for typical spacecraft
    
    def _solar_pressure_torque(self, quat: np.ndarray) -> np.ndarray:
        """Compute solar radiation pressure torque"""
        # Simplified solar vector (pointing from spacecraft to sun)
        sun_vector = np.array([1, 0, 0])  # Simplified constant direction
        
        # Torque arm
        torque_arm = self.properties.center_of_pressure - self.properties.center_of_mass
        
        # Solar pressure force
        pressure_magnitude = self.environment.solar_pressure * self.properties.cross_sectional_area
        pressure_force = pressure_magnitude * sun_vector
        
        # Torque due to solar pressure
        torque = np.cross(torque_arm, pressure_force)
        
        return torque * 1e-6  # Scale for typical spacecraft
    
    def _quaternion_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat
        
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])


class ThrusterSystemModel:
    """High-fidelity thruster system modeling"""
    
    def __init__(self, propulsion_config: PropulsionSystem):
        self.config = propulsion_config
        self.health_status = np.ones(propulsion_config.num_thrusters)
        self.fuel_remaining = 1.0  # Normalized fuel level
        
        # Thruster dynamics state
        self.current_thrust_levels = np.zeros(propulsion_config.num_thrusters)
        self.valve_states = np.zeros(propulsion_config.num_thrusters)  # 0 = closed, 1 = open
        
        # Fault tracking
        self.active_faults = {}
        
        # Performance tracking
        self.total_impulse_used = 0.0
        self.firing_count = np.zeros(propulsion_config.num_thrusters)
        
    def compute_forces_and_torques(self, thruster_commands: np.ndarray,
                                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forces and torques from thruster commands with realistic dynamics
        
        Args:
            thruster_commands: Commanded thrust levels [0, 1] for each thruster
            dt: Time step [s]
            
        Returns:
            force_body: Total force in body frame [N]
            torque_body: Total torque in body frame [N*m]
        """
        # Apply command limits and health degradation
        effective_commands = np.clip(thruster_commands, 0, 1) * self.health_status
        
        # Update valve states and thrust dynamics
        self._update_valve_dynamics(effective_commands, dt)
        self._update_thrust_dynamics(effective_commands, dt)
        
        # Compute actual thrust levels
        actual_thrusts = self.current_thrust_levels * self.config.max_thrust_per_thruster
        
        # Apply minimum impulse bit constraint
        actual_thrusts = self._apply_minimum_impulse_bit(actual_thrusts, dt)
        
        # Compute forces and torques
        force_body = np.zeros(3)
        torque_body = np.zeros(3)
        
        for i in range(self.config.num_thrusters):
            thrust_vector = actual_thrusts[i] * self.config.thruster_directions[i]
            thrust_position = self.config.thruster_positions[i]
            
            force_body += thrust_vector
            torque_body += np.cross(thrust_position, thrust_vector)
        
        # Update fuel consumption
        total_thrust = np.sum(actual_thrusts)
        mass_flow_rate = total_thrust / (self.config.specific_impulse * 9.81)
        fuel_consumed = mass_flow_rate * dt
        self.fuel_remaining -= fuel_consumed / 100.0  # Normalized
        self.fuel_remaining = max(0.0, self.fuel_remaining)
        
        # Update performance metrics
        self.total_impulse_used += total_thrust * dt
        self.firing_count += (actual_thrusts > 0).astype(float)
        
        return force_body, torque_body
    
    def _update_valve_dynamics(self, commands: np.ndarray, dt: float):
        """Update valve state dynamics with response time"""
        for i in range(self.config.num_thrusters):
            if commands[i] > 0 and self.valve_states[i] == 0:
                # Opening valve
                self.valve_states[i] = min(1.0, self.valve_states[i] + dt/self.config.valve_response_time)
            elif commands[i] == 0 and self.valve_states[i] > 0:
                # Closing valve
                self.valve_states[i] = max(0.0, self.valve_states[i] - dt/self.config.valve_response_time)
    
    def _update_thrust_dynamics(self, commands: np.ndarray, dt: float):
        """Update thrust level dynamics with rise/decay times"""
        for i in range(self.config.num_thrusters):
            target_thrust = commands[i] * self.valve_states[i]
            
            if target_thrust > self.current_thrust_levels[i]:
                # Thrust rise
                rise_rate = 1.0 / self.config.thrust_rise_time
                self.current_thrust_levels[i] = min(target_thrust, 
                    self.current_thrust_levels[i] + rise_rate * dt)
            elif target_thrust < self.current_thrust_levels[i]:
                # Thrust decay
                decay_rate = 1.0 / self.config.thrust_decay_time
                self.current_thrust_levels[i] = max(target_thrust,
                    self.current_thrust_levels[i] - decay_rate * dt)
    
    def _apply_minimum_impulse_bit(self, thrusts: np.ndarray, dt: float) -> np.ndarray:
        """Apply minimum impulse bit constraint"""
        impulse_bits = thrusts * dt
        
        for i in range(len(thrusts)):
            if 0 < impulse_bits[i] < self.config.minimum_impulse_bit:
                thrusts[i] = 0.0  # Thrust too small, turn off
        
        return thrusts
    
    def inject_fault(self, thruster_id: int, fault_type: FaultType, 
                    severity: float = 0.5, duration: Optional[float] = None):
        """Inject realistic thruster faults"""
        if 0 <= thruster_id < self.config.num_thrusters:
            if fault_type == FaultType.THRUSTER_DEGRADATION:
                self.health_status[thruster_id] *= (1.0 - severity)
            elif fault_type == FaultType.THRUSTER_FAILURE:
                self.health_status[thruster_id] = 0.0
            
            self.active_faults[thruster_id] = {
                'type': fault_type,
                'severity': severity,
                'start_time': 0.0,  # Would use actual time in full implementation
                'duration': duration
            }
            
            logger.info(f"Thruster {thruster_id} fault injected: {fault_type.value}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive thruster system status"""
        return {
            'health_status': self.health_status.copy(),
            'fuel_remaining': self.fuel_remaining,
            'current_thrust_levels': self.current_thrust_levels.copy(),
            'valve_states': self.valve_states.copy(),
            'active_faults': self.active_faults.copy(),
            'total_impulse_used': self.total_impulse_used,
            'firing_count': self.firing_count.copy(),
            'operational_thrusters': np.sum(self.health_status > 0.1)
        }


class SensorSystemModel:
    """Realistic sensor system with noise and faults"""
    
    def __init__(self, sensor_config: SensorSuite):
        self.config = sensor_config
        
        # Sensor states
        self.gps_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accelerometer_bias = np.zeros(3)
        
        # Fault states
        self.sensor_faults = {}
        
        # Measurement history for filtering
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=10)
        
    def measure_navigation_state(self, true_position: np.ndarray,
                               true_velocity: np.ndarray,
                               true_attitude: np.ndarray,
                               true_angular_velocity: np.ndarray,
                               dt: float) -> Dict[str, np.ndarray]:
        """
        Generate realistic sensor measurements with noise and biases
        
        Args:
            true_position: True relative position [m]
            true_velocity: True relative velocity [m/s]
            true_attitude: True attitude quaternion
            true_angular_velocity: True angular velocity [rad/s]
            dt: Time step [s]
            
        Returns:
            Dictionary of sensor measurements
        """
        measurements = {}
        
        # GPS position measurement
        if 'gps_position' not in self.sensor_faults:
            pos_noise = np.random.normal(0, self.config.gps_accuracy, 3)
            measurements['position'] = true_position + self.gps_bias + pos_noise
        else:
            measurements['position'] = np.full(3, np.nan)  # GPS failure
        
        # GPS velocity measurement  
        if 'gps_velocity' not in self.sensor_faults:
            vel_noise = np.random.normal(0, self.config.gps_velocity_accuracy, 3)
            measurements['velocity'] = true_velocity + vel_noise
        else:
            measurements['velocity'] = np.full(3, np.nan)  # GPS failure
        
        # Star tracker attitude measurement
        if 'star_tracker' not in self.sensor_faults:
            att_noise_angle = np.random.normal(0, self.config.star_tracker_accuracy)
            if abs(att_noise_angle) > 1e-8:
                noise_axis = self._random_unit_vector()
                noise_quat = self._axis_angle_to_quaternion(noise_axis, att_noise_angle)
                measurements['attitude'] = self._quaternion_multiply(true_attitude, noise_quat)
                measurements['attitude'] /= np.linalg.norm(measurements['attitude'])
            else:
                measurements['attitude'] = true_attitude.copy()
        else:
            measurements['attitude'] = np.array([1, 0, 0, 0])  # Default attitude
        
        # Gyroscope angular velocity measurement
        if 'gyroscope' not in self.sensor_faults:
            # Update gyro bias (random walk)
            self.gyro_bias += np.random.normal(0, self.config.gyroscope_bias_stability * np.sqrt(dt), 3)
            
            gyro_noise = np.random.normal(0, self.config.gyroscope_noise, 3)
            measurements['angular_velocity'] = true_angular_velocity + self.gyro_bias + gyro_noise
        else:
            measurements['angular_velocity'] = np.zeros(3)  # Gyro failure
        
        # Range sensor (if applicable)
        if 'range_sensor' not in self.sensor_faults:
            range_noise = np.random.normal(0, self.config.range_sensor_accuracy)
            measurements['range'] = np.linalg.norm(true_position) + range_noise
        else:
            measurements['range'] = np.nan
        
        # Store measurements for filtering
        self.position_history.append(measurements.get('position', np.full(3, np.nan)))
        self.velocity_history.append(measurements.get('velocity', np.full(3, np.nan)))
        
        return measurements
    
    def inject_sensor_fault(self, sensor_name: str, fault_type: FaultType,
                          severity: float = 1.0, duration: Optional[float] = None):
        """Inject sensor faults"""
        self.sensor_faults[sensor_name] = {
            'type': fault_type,
            'severity': severity,
            'start_time': 0.0,
            'duration': duration
        }
        
        if fault_type == FaultType.SENSOR_BIAS:
            if sensor_name == 'gps_position':
                self.gps_bias = np.random.normal(0, severity * self.config.gps_accuracy, 3)
            elif sensor_name == 'gyroscope':
                self.gyro_bias += np.random.normal(0, severity * np.deg2rad(1.0), 3)
        
        logger.info(f"Sensor fault injected: {sensor_name} - {fault_type.value}")
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get sensor system status"""
        return {
            'gps_bias': self.gps_bias.copy(),
            'gyro_bias': self.gyro_bias.copy(),
            'active_faults': self.sensor_faults.copy(),
            'update_frequency': self.config.update_frequency,
            'operational_sensors': len(set(['gps_position', 'gps_velocity', 'star_tracker', 'gyroscope']) 
                                      - set(self.sensor_faults.keys()))
        }
    
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


class SpacecraftSimulator:
    """
    High-fidelity spacecraft dynamics simulator for autonomous rendezvous and docking
    
    Integrates orbital mechanics, attitude dynamics, propulsion, and sensor systems
    with realistic fault modeling for safe RL training.
    """
    
    def __init__(self, 
                 spacecraft_props: SpacecraftProperties,
                 propulsion_config: PropulsionSystem,
                 sensor_config: SensorSuite,
                 orbital_env: OrbitalEnvironment):
        """Initialize the spacecraft simulator"""
        
        self.spacecraft_props = spacecraft_props
        self.orbital_env = orbital_env
        
        # Initialize subsystems
        self.orbital_mechanics = OrbitalMechanicsEngine(orbital_env)
        self.attitude_dynamics = AttitudeDynamicsEngine(spacecraft_props, orbital_env)
        self.thruster_system = ThrusterSystemModel(propulsion_config)
        self.sensor_system = SensorSystemModel(sensor_config)
        
        # Simulation state
        self.time = 0.0
        self.integration_method = 'RK45'
        self.integration_tolerance = 1e-8
        
        # Performance metrics
        self.simulation_stats = {
            'integration_failures': 0,
            'fault_injections': 0,
            'total_delta_v': 0.0,
            'total_angular_impulse': 0.0
        }
        
    def propagate_dynamics(self, 
                          initial_state: Dict[str, np.ndarray],
                          thruster_commands: np.ndarray,
                          dt: float,
                          include_disturbances: bool = True) -> Dict[str, np.ndarray]:
        """
        Propagate complete 6DOF spacecraft dynamics
        
        Args:
            initial_state: Dictionary containing position, velocity, attitude, angular_velocity
            thruster_commands: Thruster command vector [0, 1]
            dt: Integration time step [s]
            include_disturbances: Include environmental disturbances
            
        Returns:
            New state after propagation
        """
        
        # Extract state components
        position = initial_state['position']
        velocity = initial_state['velocity']
        attitude = initial_state['attitude']
        angular_velocity = initial_state['angular_velocity']
        
        # Compute thruster forces and torques
        thrust_force, thrust_torque = self.thruster_system.compute_forces_and_torques(
            thruster_commands, dt
        )
        
        # Compute environmental forces
        total_force = thrust_force.copy()
        total_torque = thrust_torque.copy()
        
        if include_disturbances:
            # Atmospheric drag
            drag_force = self.orbital_mechanics.atmospheric_drag_force(
                velocity, self.spacecraft_props
            )
            total_force += drag_force
            
            # Solar radiation pressure (simplified)
            sun_vector = np.array([1, 0, 0])  # Simplified sun direction
            srp_force = self.orbital_mechanics.solar_radiation_pressure_force(
                sun_vector, self.spacecraft_props
            )
            total_force += srp_force
        
        # Integrate translational dynamics
        trans_state = np.concatenate([position, velocity])
        
        def trans_dynamics(t, y):
            return self.orbital_mechanics.hill_clohessy_wiltshire_full(
                y, total_force, self.spacecraft_props.mass, include_j2=include_disturbances
            )
        
        # Integrate rotational dynamics
        rot_state = np.concatenate([attitude, angular_velocity])
        
        def rot_dynamics(t, y):
            return self.attitude_dynamics.rigid_body_dynamics(
                y, total_torque, include_disturbances
            )
        
        try:
            # Integrate translational motion
            sol_trans = solve_ivp(trans_dynamics, [0, dt], trans_state,
                                method=self.integration_method, 
                                rtol=self.integration_tolerance,
                                atol=self.integration_tolerance * 1e-2)
            
            # Integrate rotational motion
            sol_rot = solve_ivp(rot_dynamics, [0, dt], rot_state,
                              method=self.integration_method,
                              rtol=self.integration_tolerance,
                              atol=self.integration_tolerance * 1e-2)
            
            if sol_trans.success and sol_rot.success:
                # Extract final states
                final_trans = sol_trans.y[:, -1]
                final_rot = sol_rot.y[:, -1]
                
                new_position = final_trans[:3]
                new_velocity = final_trans[3:]
                new_attitude = final_rot[:4]
                new_angular_velocity = final_rot[4:]
                
                # Normalize quaternion
                new_attitude /= np.linalg.norm(new_attitude)
                
                # Update performance metrics
                delta_v = np.linalg.norm(thrust_force) * dt / self.spacecraft_props.mass
                angular_impulse = np.linalg.norm(thrust_torque) * dt
                self.simulation_stats['total_delta_v'] += delta_v
                self.simulation_stats['total_angular_impulse'] += angular_impulse
                
            else:
                # Integration failed, use fallback
                logger.warning("Integration failed, using Euler fallback")
                new_position, new_velocity, new_attitude, new_angular_velocity = \
                    self._euler_fallback(initial_state, total_force, total_torque, dt)
                self.simulation_stats['integration_failures'] += 1
        
        except Exception as e:
            logger.error(f"Integration error: {e}")
            new_position, new_velocity, new_attitude, new_angular_velocity = \
                self._euler_fallback(initial_state, total_force, total_torque, dt)
            self.simulation_stats['integration_failures'] += 1
        
        # Update simulation time
        self.time += dt
        
        # Return new state
        return {
            'position': new_position,
            'velocity': new_velocity,
            'attitude': new_attitude,
            'angular_velocity': new_angular_velocity
        }
    
    def _euler_fallback(self, state: Dict[str, np.ndarray],
                       force: np.ndarray, torque: np.ndarray, dt: float) -> Tuple:
        """Simple Euler integration fallback"""
        position = state['position']
        velocity = state['velocity']
        attitude = state['attitude']
        angular_velocity = state['angular_velocity']
        
        # Translational dynamics
        trans_state = np.concatenate([position, velocity])
        trans_deriv = self.orbital_mechanics.hill_clohessy_wiltshire_full(
            trans_state, force, self.spacecraft_props.mass, False
        )
        
        new_position = position + trans_deriv[:3] * dt
        new_velocity = velocity + trans_deriv[3:] * dt
        
        # Rotational dynamics
        rot_state = np.concatenate([attitude, angular_velocity])
        rot_deriv = self.attitude_dynamics.rigid_body_dynamics(
            rot_state, torque, False
        )
        
        new_attitude = attitude + rot_deriv[:4] * dt
        new_angular_velocity = angular_velocity + rot_deriv[4:] * dt
        
        # Normalize quaternion
        new_attitude /= np.linalg.norm(new_attitude)
        
        return new_position, new_velocity, new_attitude, new_angular_velocity
    
    def generate_sensor_measurements(self, true_state: Dict[str, np.ndarray],
                                   dt: float) -> Dict[str, np.ndarray]:
        """Generate realistic sensor measurements"""
        return self.sensor_system.measure_navigation_state(
            true_state['position'],
            true_state['velocity'], 
            true_state['attitude'],
            true_state['angular_velocity'],
            dt
        )
    
    def inject_system_fault(self, subsystem: str, fault_params: Dict[str, Any]):
        """Inject faults into various subsystems"""
        if subsystem == 'thruster':
            self.thruster_system.inject_fault(
                fault_params['thruster_id'],
                fault_params['fault_type'],
                fault_params.get('severity', 0.5),
                fault_params.get('duration', None)
            )
        elif subsystem == 'sensor':
            self.sensor_system.inject_sensor_fault(
                fault_params['sensor_name'],
                fault_params['fault_type'],
                fault_params.get('severity', 1.0),
                fault_params.get('duration', None)
            )
        
        self.simulation_stats['fault_injections'] += 1
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive simulator status"""
        return {
            'simulation_time': self.time,
            'thruster_status': self.thruster_system.get_system_status(),
            'sensor_status': self.sensor_system.get_sensor_status(),
            'simulation_stats': self.simulation_stats.copy(),
            'orbital_parameters': {
                'altitude': self.orbital_env.altitude,
                'mean_motion': self.orbital_mechanics.mean_motion,
                'orbital_period': self.orbital_mechanics.mean_motion * 2 * np.pi
            }
        }
    
    def reset_simulator(self):
        """Reset simulator to initial conditions"""
        self.time = 0.0
        self.thruster_system = ThrusterSystemModel(self.thruster_system.config)
        self.sensor_system = SensorSystemModel(self.sensor_system.config) 
        self.simulation_stats = {
            'integration_failures': 0,
            'fault_injections': 0,
            'total_delta_v': 0.0,
            'total_angular_impulse': 0.0
        }