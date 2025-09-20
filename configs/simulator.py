"""

=====================

Author: Strtus
"""

import numpy as np
import sys
from pathlib import Path

# 添加src路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from simulator import (
        SpacecraftProperties, PropulsionSystem, SensorSuite, 
        OrbitalEnvironment, ThrusterType, SpacecraftSimulator
    )
except ImportError:
    print("Warning: Simulator modules not available")


def create_qianfan_spacecraft_properties():
    """创建千帆卫星的物理属性配置"""
    return SpacecraftProperties(
        mass=150.0,  # kg - 千帆卫星质量
        inertia_matrix=np.array([
            [25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0], 
            [0.0, 0.0, 40.0]
        ]),  # kg*m^2 - 基于千帆卫星几何形状
        dimensions=np.array([1.2, 1.2, 0.8]),  # m - 长宽高
        center_of_mass=np.array([0.0, 0.0, 0.0]),  # m - 质心偏移
        solar_panel_area=1.5,  # m^2
        drag_coefficient=2.2,  # 典型卫星阻力系数
        reflectance_coefficient=0.1  # 太阳辐射压反射系数
    )


def create_qianfan_propulsion_system():
    """创建千帆卫星的推进系统配置"""
    return PropulsionSystem(
        thruster_type=ThrusterType.COLD_GAS,  # 冷气推进器
        num_thrusters=8,  # 8个推进器
        max_thrust_per_thruster=0.5,  # N
        specific_impulse=70.0,  # s - 冷气推进器比冲
        fuel_capacity=5.0,  # kg
        tank_pressure=200.0,  # bar
        thruster_positions=np.array([
            # 8个推进器的位置配置 (m)
            [0.6, 0.6, 0.4],    # +X+Y面
            [0.6, -0.6, 0.4],   # +X-Y面
            [-0.6, 0.6, 0.4],   # -X+Y面
            [-0.6, -0.6, 0.4],  # -X-Y面
            [0.6, 0.6, -0.4],   # +X+Y面下
            [0.6, -0.6, -0.4],  # +X-Y面下
            [-0.6, 0.6, -0.4],  # -X+Y面下
            [-0.6, -0.6, -0.4]  # -X-Y面下
        ]),
        thruster_directions=np.array([
            # 8个推进器的方向向量
            [-1, -1, 0],  # 指向中心
            [-1, 1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, -1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [1, 1, 0]
        ]) / np.sqrt(2),  # 归一化
        minimum_impulse_bit=0.001,  # N*s - 最小冲量位
        response_time=0.05,  # s - 推进器响应时间
        thrust_noise_std=0.02  # 推力噪声标准差
    )


def create_qianfan_sensor_suite():
    """创建千帆卫星的传感器配置"""
    return SensorSuite(
        # GPS接收机
        gps_position_noise=1.0,  # m - GPS位置精度
        gps_velocity_noise=0.01,  # m/s - GPS速度精度
        gps_update_rate=1.0,  # Hz - GPS更新频率
        
        # 星敏感器
        star_tracker_noise=np.deg2rad(0.01),  # rad - 星敏感器精度
        star_tracker_update_rate=10.0,  # Hz
        star_tracker_fov=120.0,  # deg - 视场角
        
        # 太阳敏感器
        sun_sensor_noise=np.deg2rad(0.5),  # rad
        sun_sensor_fov=120.0,  # deg
        
        # 磁强计
        magnetometer_noise=50e-9,  # T - 磁强计噪声
        magnetometer_bias=100e-9,  # T - 磁强计偏置
        
        # 激光雷达 (对接传感器)
        lidar_range=1000.0,  # m - 最大探测距离
        lidar_accuracy=0.1,  # m - 测距精度
        lidar_angular_resolution=0.1,  # deg - 角分辨率
        lidar_update_rate=20.0,  # Hz
        
        # 惯性测量单元
        gyro_noise=np.deg2rad(0.01),  # rad/s - 陀螺仪噪声
        gyro_bias_stability=np.deg2rad(0.1),  # rad/s - 陀螺仪偏置稳定性
        accelerometer_noise=1e-5,  # m/s^2 - 加速度计噪声
        
        # 通用传感器参数
        measurement_delay=0.1,  # s - 测量延迟
        dropout_probability=0.001  # 数据丢失概率
    )


def create_qianfan_orbital_environment():
    """创建千帆卫星的轨道环境"""
    return OrbitalEnvironment(
        altitude=500000.0,  # m - 500km轨道高度
        inclination=np.deg2rad(97.4),  # rad - 太阳同步轨道倾角
        eccentricity=0.001,  # 近圆轨道
        argument_of_perigee=0.0,  # rad
        raan=0.0,  # rad - 升交点赤经
        true_anomaly=0.0,  # rad
        
        # 环境扰动参数
        atmospheric_density=1e-12,  # kg/m^3 - 500km大气密度
        solar_flux=1361.0,  # W/m^2 - 太阳辐射通量
        earth_magnetic_field_strength=30e-6,  # T - 地磁场强度
        gravity_gradient_enabled=True,
        j2_perturbation_enabled=True,
        atmospheric_drag_enabled=True,
        solar_radiation_pressure_enabled=True,
        
        # 地球参数
        earth_radius=6.371e6,  # m
        earth_mu=3.986004418e14,  # m^3/s^2
        j2_coefficient=1.08263e-3,
        
        # 环境噪声
        atmospheric_density_variation=0.1,  # ±10%变化
        solar_flux_variation=0.05,  # ±5%变化
        magnetic_field_variation=0.1  # ±10%变化
    )


def create_qianfan_simulator():
    """创建完整的千帆卫星仿真器"""
    spacecraft_props = create_qianfan_spacecraft_properties()
    propulsion_config = create_qianfan_propulsion_system()
    sensor_config = create_qianfan_sensor_suite()
    orbital_env = create_qianfan_orbital_environment()
    
    try:
        simulator = SpacecraftSimulator(
            spacecraft_props=spacecraft_props,
            propulsion_config=propulsion_config,
            sensor_config=sensor_config,
            orbital_env=orbital_env
        )
        return simulator
    except Exception as e:
        print(f"无法创建仿真器: {e}")
        return None


# 千帆特定的仿真参数
QIANFAN_SIMULATION_CONFIG = {
    'integration_method': 'dopri5',  # 高精度积分方法
    'integration_tolerance': 1e-8,   # 积分容差
    'max_step_size': 1.0,           # s - 最大步长
    'min_step_size': 0.01,          # s - 最小步长
    
    # 环境扰动配置
    'disturbances': {
        'gravity_gradient': True,
        'atmospheric_drag': True,
        'solar_radiation_pressure': True,
        'earth_magnetic_field': True,
        'j2_perturbation': True
    },
    
    # 故障注入配置
    'fault_injection': {
        'enabled': True,
        'thruster_failure_rate': 1e-6,  # 故障/秒
        'sensor_bias_drift_rate': 1e-8,  # 偏置漂移率
        'communication_dropout_rate': 1e-4  # 通信中断率
    },
    
    # 性能监控
    'performance_monitoring': {
        'fuel_consumption_tracking': True,
        'thermal_analysis': False,  # 暂不包含热分析
        'power_consumption_tracking': True,
        'attitude_control_performance': True
    }
}


if __name__ == "__main__":
    # 测试千帆仿真器配置
    print("创建千帆卫星仿真器配置...")
    
    spacecraft = create_qianfan_spacecraft_properties()
    print(f"千帆卫星质量: {spacecraft.mass} kg")
    print(f"千帆卫星尺寸: {spacecraft.dimensions} m")
    
    propulsion = create_qianfan_propulsion_system()
    print(f"推进器数量: {propulsion.num_thrusters}")
    print(f"单个推进器推力: {propulsion.max_thrust_per_thruster} N")
    print(f"比冲: {propulsion.specific_impulse} s")
    
    orbital_env = create_qianfan_orbital_environment()
    print(f"轨道高度: {orbital_env.altitude/1000} km")
    print(f"轨道倾角: {np.rad2deg(orbital_env.inclination)} deg")
    
    simulator = create_qianfan_simulator()
    if simulator:
        print("千帆卫星仿真器创建成功！")
    else:
        print("千帆卫星仿真器创建失败")