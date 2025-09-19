# Constrained-Reinforcement-Dynamics-Docking-Simulation

CReDyS: Constrained Reinforcement Dynamics Docking Simulation

***

**Description**

CReDyS (Constrained Reinforcement Dynamics Docking Simulation) is an open-source simulation framework designed for the study of autonomous control strategies in spacecraft rendezvous and docking (RvD) tasks. It is based on Constrained Reinforcement Learning (CRL) and utilizes the Guided Reward Policy Optimization (GRPO) algorithm to develop a 6 degrees of freedom (6DOF) dynamics control strategy. CReDyS ensures safe docking even in the presence of disturbances such as thruster faults.

CReDyS provides a simulation environment based on random fault injection, mathematical state spaces, reward functions, and safety constraints, focusing on robust control strategies in high-dimensional dynamics. The framework offers a highly scalable tool for aerospace dynamics, control theory, and reinforcement learning, supporting precise modeling and policy analysis.

***

**Research Background**

Spacecraft rendezvous and docking tasks require high-precision control in the presence of dynamic disturbances, such as thruster faults or sensor noise. Traditional control methods, such as Model Predictive Control (MPC), often struggle to meet safety and robustness requirements in complex environments. Constrained Reinforcement Learning provides a theoretical framework for autonomous control by introducing safety constraints, such as velocity and attitude limits, within the Markov Decision Process (MDP). This ensures efficient and safe operation in dynamic environments.

CReDyS uses the GRPO algorithm to efficiently optimize policies $\pi(a|s)$ in the 6DOF state space, ensuring safety constraints are met through Lagrangian multipliers or Constrained Policy Optimization (CPO). The framework focuses on the application of GRPO in spacecraft docking, particularly in fault-tolerant and safety-critical scenarios.

***

**Methodology**

CReDyS implements autonomous spacecraft docking control strategies using the following methods:

Safety constraint control defines constraints on relative velocity and attitude deviation. Specifically, the velocity constraint is $|\mathbf{v}| \leq v_{\text{max}}$, where $\mathbf{v} \in \mathbb{R}^3$ is the velocity vector, and the attitude constraint is $|\mathbf{q} - \mathbf{q}_{\text{target}}| \leq \epsilon$, where $\mathbf{q} \in \mathbb{S}^3$ is the quaternion representation of attitude.

The GRPO algorithm optimizes control policies by updating the policy gradient, where the optimization target is:

$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot (R(s,a) - \lambda C(s,a))]$

Here, $C(s,a)$ is the constraint cost function, and $\lambda$ is the Lagrangian multiplier. This process ensures that while improving control efficiency, the policy adheres to safety constraints.

Fault tolerance is achieved by training the model on simulated random faults, such as a 20% reduction in thruster efficiency, ensuring the strategy adapts to different failure scenarios and maximizes docking success.

SHAP values are used to decompose the action contributions of the GRPO policy, providing visualizations to support in-depth analysis of control performance.

***

**Technical Specifications**

6DOF dynamics modeling uses Spacecraft Robotics Toolkit (SRT) or Orekit to simulate the translational and rotational dynamics of spacecraft.

The translational dynamics are described by the Clohessy-Wiltshire equations, valid for low Earth orbit, as follows:

$$\ddot{\mathbf{r}} = -2 \boldsymbol{\omega}_0 \times \dot{\mathbf{r}} - \frac{\mu}{|\mathbf{R}|^3} \mathbf{r} + \frac{\mathbf{u}}{m} + \mathbf{d}$$

Where $\mathbf{r} \in \mathbb{R}^3$ is the relative position, $\boldsymbol{\omega}_0$ is the orbital angular velocity, $\mu$ is the gravitational constant, $\mathbf{R}$ is the orbital radius, $\mathbf{u} \in \mathbb{R}^3$ is the thrust, $m$ is the spacecraft mass, and $\mathbf{d}$ represents disturbances such as faults.

Rotational dynamics, based on quaternions, are described by:

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \circ \begin{bmatrix} 0 & \boldsymbol{\omega} \end{bmatrix}, \quad \mathbf{I} \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times \mathbf{I} \boldsymbol{\omega} = \boldsymbol{\tau} + \mathbf{d}\tau$$

Where $\mathbf{q} \in \mathbb{S}^3$ is the quaternion attitude representation, $\boldsymbol{\omega} \in \mathbb{R}^3$ is the angular velocity, $\mathbf{I}$ is the inertia matrix, $\boldsymbol{\tau} \in \mathbb{R}^3$ is the control torque, and $\mathbf{d}\tau$ represents disturbance torques. The symbol $\circ$ denotes quaternion multiplication.

The Constrained Reinforcement Learning environment is defined based on Ray RLlib and OpenAI Gym.

State space is defined as:

$$s = [\mathbf{r}, \mathbf{v}, \mathbf{q}, \boldsymbol{\omega}, \mathbf{f}]$$

Where $\mathbf{f}$ is the fault indicator (e.g., thruster efficiency $\eta \in [0,1]$).

Action space is continuous:

$$a = [\mathbf{u}, \boldsymbol{\tau}] \in \mathbb{R}^6$$

The reward function is:

$$ R(s,a) = w_1 \cdot \mathbb{1}_{|\mathbf{r}| < 0.1, |\mathbf{q} - \mathbf{q}_{\text{target}}| < \epsilon} - w_2 \cdot |\mathbf{r}| - w_3 \cdot \int |\mathbf{u}|^2 \, dt - w_4 \cdot \mathbb{1}_{|\mathbf{v}| > v_{\text{max}}}$$

Where $w_i$ are weights and $\mathbb{1}_{\cdot}$ is the indicator function.

The safety constraint is defined as:

$$C(s,a) = \max(0, |\mathbf{v}| - v_{\text{max}}) + \max(0, |\mathbf{q} - \mathbf{q}_{\text{target}}| - \epsilon)$$

The optimization ensures $\mathbb{E}[C(s,a)] \leq \delta$, where $\delta$ is the constraint threshold.

Termination conditions are met when docking succeeds ($|\mathbf{r}| < 0.1$ m, $|\mathbf{q} - \mathbf{q}_{\text{target}}| < \epsilon$) or fails (1000 steps timeout or unsafe state).
