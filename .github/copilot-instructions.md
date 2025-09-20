# AI Agent Instructions for CReDyS Framework

## Project Overview
CReDyS (Constrained Reinforcement Dynamics Docking Simulation) is a professional aerospace simulation framework for spacecraft rendezvous and docking using **Guided Reward Policy Optimization (GRPO)** with safety constraints. This is NOT a toy project - it implements production-grade orbital mechanics, quaternion attitude dynamics, and constrained reinforcement learning.

## Architecture Fundamentals

### Core Component Structure
- **`main.py`**: Mission controller orchestrating all components with professional logging and error handling
- **`src/environment.py`**: High-fidelity 6DOF spacecraft environment implementing Hill-Clohessy-Wiltshire equations and quaternion dynamics
- **`src/agent.py`**: Advanced GRPO agent with safety-aware neural networks, attention mechanisms, and Lagrangian multipliers
- **`src/simulator.py`**: Physics-accurate spacecraft dynamics with fault modeling and thruster allocation
- **`src/training.py`**: Training pipeline with curriculum learning and hyperparameter optimization
- **`src/visualization.py`**: Comprehensive analysis and plotting capabilities

### Key Domain-Specific Patterns

#### State Representation (26-dimensional)
```python
# State vector: [pos(3), vel(3), quat(4), omega(3), thruster_health(12), fuel(1)]
# Position/velocity in Hill frame (Earth-pointing coordinates)
# Quaternions for attitude (NOT Euler angles)
# Angular velocity in body frame
# Individual thruster health status (0.0-1.0)
```

#### Physical Dynamics Integration
- Use `solve_ivp` with RK45 for orbital mechanics (fallback to Euler if needed)
- Hill-Clohessy-Wiltshire equations for relative orbital motion
- Euler's equations for rigid body attitude dynamics
- Always normalize quaternions after integration

#### Safety Constraints Implementation
```python
# Four constraint types monitored:
constraints = {
    'collision': distance < 0.5,           # Collision avoidance
    'velocity': speed > max_velocity,      # Velocity limits  
    'boundary': distance > workspace,      # Workspace bounds
    'angular_rate': omega > max_omega      # Angular rate limits
}
# Use Lagrangian multipliers for constraint satisfaction
```

## Development Workflows

### Running the Framework
```bash
# Training mode (default)
python main.py --mode train --config config.yaml

# Evaluation mode
python main.py --mode evaluate --model checkpoints/best_model.pth

# Simulation mode
python main.py --mode simulate --scenario scenarios/nominal.json
```

### Testing Strategy
```bash
# Run comprehensive test suite
python test_framework.py

# Individual component testing
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_environment.py -v
```

### Configuration Management
- Use `configs/qianfan_config.py` for spacecraft-specific parameters
- Environment config in main script `create_default_config()`
- Agent hyperparameters in `GRPOConfig` dataclass
- Always validate configuration completeness

## Critical Implementation Guidelines

### Aerospace-Specific Conventions
1. **Units**: SI units throughout (meters, seconds, kg, radians)
2. **Coordinate Frames**: Hill frame for relative motion, body frame for rotations
3. **Quaternions**: Always [w,x,y,z] format, normalize after operations
4. **Thruster Allocation**: 12-thruster configuration with fault modeling
5. **Docking Criteria**: Position ≤5cm, velocity ≤10cm/s, attitude ≤2°

### GRPO Algorithm Implementation
- Multi-head attention for state encoding with physical feature extraction
- Safety-aware action scaling based on constraint violations
- Dual critics: value function + safety constraint estimation
- Lagrangian multiplier updates with non-negativity constraints
- Trust region optimization with adaptive constraint thresholds

### Error Handling Patterns
```python
# Physics integration fallback
try:
    sol = solve_ivp(dynamics_func, [0, dt], state, method='RK45')
    if sol.success:
        return sol.y[:, -1]
    else:
        return euler_integration_fallback(state, dt)
except Exception:
    return euler_integration_fallback(state, dt)
```

### Safety-Critical Best Practices
- Always validate control inputs before thruster allocation
- Monitor constraint violations in real-time
- Implement graceful degradation for sensor/actuator faults
- Log safety metrics for post-mission analysis
- Use deterministic policies for evaluation

## Key Dependencies & Integration Points

### External Libraries
- **PyTorch**: Neural networks with CUDA acceleration
- **SciPy**: ODE integration and optimization
- **Gymnasium**: RL environment interface
- **Ray RLlib**: Distributed training (if using examples/train_grpo.py)

### Mathematical Libraries
- Use `scipy.spatial.transform.Rotation` for quaternion operations
- `scipy.integrate.solve_ivp` for dynamics integration  
- `scipy.optimize` for constrained optimization problems
- `numpy.linalg` for matrix operations and norms

### Visualization Stack
- **Matplotlib/Seaborn**: Static plots and analysis
- **Plotly**: Interactive 3D trajectory visualization
- **TensorBoard**: Training metrics monitoring

## File Modification Guidelines

### When Modifying Environment
- Update both state dimension and observation space if changing state representation
- Ensure reward function maintains constraint penalty structure
- Test physics integration with various initial conditions
- Validate safety constraint evaluation logic

### When Modifying Agent
- Maintain attention mechanism architecture for state encoding
- Preserve Lagrangian multiplier update logic
- Test policy convergence with constraint satisfaction
- Verify gradient clipping and learning rate scheduling

### When Adding Features
- Follow the modular component structure in `main.py`
- Add comprehensive logging with appropriate log levels
- Include physics validation in `test_framework.py`
- Update configuration dictionaries with new parameters

## Common Issues & Solutions

### Physics Integration Failures
- Check for NaN values in state vectors
- Ensure quaternion normalization
- Validate control input magnitudes
- Use fallback Euler integration for stability

### Training Convergence Issues  
- Monitor constraint violation rates
- Adjust Lagrangian multiplier learning rates
- Check reward function scaling
- Verify experience buffer diversity

### Memory/Performance Issues
- Use appropriate batch sizes for available GPU memory
- Monitor replay buffer size limits
- Profile physics integration bottlenecks
- Consider parallel environment execution

Remember: This is aerospace-grade simulation software. Precision, safety, and physical accuracy are paramount. Always validate mathematical models against known orbital mechanics principles.