"""
千帆卫星星座配置文件
=================

Author: Strtus
"""

# 千帆卫星基本参数
QIANFAN_SATELLITE_CONFIG = {
    # 卫星物理参数
    "satellite_mass": 150.0,  # kg - 千帆卫星质量
    "satellite_dimensions": {
        "length": 1.2,  # m
        "width": 1.2,   # m  
        "height": 0.8   # m
    },
    
    # 惯性矩阵 (kg⋅m²) - 基于卫星几何形状估算
    "inertia_matrix": [
        [25.0, 0.0, 0.0],
        [0.0, 25.0, 0.0], 
        [0.0, 0.0, 40.0]
    ],
    
    # 轨道参数
    "orbital_parameters": {
        "altitude": 500000,  # m - 500km轨道高度
        "inclination": 97.4,  # deg - 太阳同步轨道倾角
        "eccentricity": 0.001,  # 近圆轨道
        "argument_of_perigee": 0.0,  # deg
        "raan": 0.0,  # deg - 升交点赤经
        "true_anomaly": 0.0,  # deg
    },
    
    # 推进系统
    "propulsion_system": {
        "thruster_type": "cold_gas",  # 冷气推进器
        "num_thrusters": 8,  # 8个推进器
        "max_thrust_per_thruster": 0.5,  # N
        "specific_impulse": 70,  # s - 冷气推进器比冲
        "fuel_capacity": 5.0,  # kg
        "thruster_configuration": "cubic"  # 立方体配置
    },
    
    # 姿态控制系统
    "attitude_control": {
        "reaction_wheels": {
            "num_wheels": 4,  # 四轮配置
            "max_momentum": 0.1,  # N⋅m⋅s
            "max_torque": 0.01   # N⋅m
        },
        "magnetorquers": {
            "num_coils": 3,  # 三轴磁力矩器
            "max_magnetic_dipole": 1.0  # A⋅m²
        }
    },
    
    # 传感器系统
    "sensors": {
        "gps": {
            "position_accuracy": 1.0,  # m
            "velocity_accuracy": 0.01  # m/s
        },
        "star_tracker": {
            "attitude_accuracy": 0.01,  # deg
            "update_rate": 10  # Hz
        },
        "sun_sensor": {
            "accuracy": 0.5,  # deg
            "field_of_view": 120  # deg
        },
        "magnetometer": {
            "accuracy": 50e-9,  # T
            "range": 100e-6    # T
        },
        "lidar": {
            "range": 1000,  # m
            "accuracy": 0.1,  # m
            "update_rate": 20  # Hz
        }
    },
    
    # 通信系统
    "communication": {
        "data_rate": 100e6,  # bps - 100 Mbps
        "antenna_gain": 20,   # dBi
        "frequency_band": "Ka"  # Ka波段
    },
    
    # 电源系统
    "power_system": {
        "solar_panel_area": 1.5,  # m²
        "solar_efficiency": 0.28,  # 28%效率
        "battery_capacity": 50,    # Wh
        "power_consumption": 80    # W平均功耗
    },
    
    # 对接机构
    "docking_mechanism": {
        "type": "androgynous",  # 雌雄同体对接机构
        "capture_range": 0.5,   # m
        "approach_velocity_max": 0.1,  # m/s
        "alignment_tolerance": {
            "position": 0.05,  # m
            "attitude": 2.0    # deg
        }
    }
}

# 任务参数
QIANFAN_MISSION_CONFIG = {
    # 对接任务参数
    "docking_mission": {
        "target_satellite": "qianfan_target",
        "initial_separation": 1000,  # m
        "approach_phases": [
            {"name": "far_range", "distance": [1000, 100], "max_velocity": 1.0},
            {"name": "close_range", "distance": [100, 10], "max_velocity": 0.3},
            {"name": "final_approach", "distance": [10, 1], "max_velocity": 0.1}
        ]
    },
    
    # 安全约束
    "safety_constraints": {
        "max_relative_velocity": 0.5,  # m/s
        "keep_out_zone_radius": 2.0,   # m
        "approach_corridor_angle": 15.0,  # deg
        "fuel_reserve_ratio": 0.2,     # 20%燃料储备
        "max_contact_force": 50.0      # N
    },
    
    # 环境参数
    "environment": {
        "gravity_gradient": True,
        "atmospheric_drag": True,  # 500km高度仍有微弱大气阻力
        "solar_radiation_pressure": True,
        "earth_magnetic_field": True,
        "j2_perturbation": True
    }
}

# 训练参数
QIANFAN_TRAINING_CONFIG = {
    "environment_config": {
        "max_episode_steps": 3000,  # 更长的episode适应千帆任务
        "dt": 1.0,  # s - 控制周期
        "initial_separation_range": [50, 1000],  # m
        "target_approach_tolerance": 0.1,  # m
        "velocity_tolerance": 0.05,  # m/s
        "attitude_tolerance": 1.0,  # deg
    },
    
    "reward_config": {
        "distance_reward_weight": 1.0,
        "velocity_reward_weight": 0.5,
        "attitude_reward_weight": 0.3,
        "fuel_efficiency_weight": 0.2,
        "safety_penalty_weight": 10.0,
        "success_reward": 1000.0
    },
    
    "agent_config": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "lambda_gae": 0.95,
        "entropy_coeff": 0.01,
        "value_loss_coeff": 0.5,
        "max_grad_norm": 0.5,
        "safety_weight": 1.0
    }
}