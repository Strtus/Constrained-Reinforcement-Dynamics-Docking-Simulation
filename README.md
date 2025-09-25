# Constrained Reinforcement Dynamics Docking Simulation 
# 受约束强化动力学对接仿真





**Description**

CReDyS（受约束强化动力学对接仿真）是一个开源的仿真框架，旨在研究航天器自主控制策略在交会对接（RvD）任务中的应用。它基于受约束强化学习（CRL）并采用引导奖励策略优化（GRPO）算法，开发了一种六自由度（6DOF）动力学控制策略。CReDyS能够确保即使在推进器故障等扰动存在的情况下也能实现安全对接。

CReDyS提供了一个基于随机故障注入、数学状态空间、奖励函数和安全约束的仿真环境，重点研究高维动力学中的鲁棒控制策略。该框架为航空航天动力学、控制理论和强化学习提供了一个高度可扩展的工具，支持精确建模和策略分析。



**Background**

航天器交会对接任务需要在动态扰动（如推进器故障或传感器噪声）存在的情况下进行高精度控制。传统控制方法，如模型预测控制（MPC），往往难以在复杂环境中满足安全性和鲁棒性要求。受约束强化学习通过在马尔可夫决策过程（MDP）中引入速度和姿态限制等安全约束，为自主控制提供了理论框架。这确保了在动态环境中实现高效且安全的操作。

CReDyS使用GRPO算法，在6DOF状态空间中高效地优化策略$\pi(a|s)$，并通过拉格朗日乘子或受约束策略优化（CPO）确保满足安全约束。该框架专注于GRPO在航天器对接中的应用，特别是在容错和安全关键场景中的应用。



**Methodology**

CReDyS使用以下方法实现航天器自主对接控制策略：

安全约束控制定义了相对速度和姿态偏差的约束。具体来说，速度约束为$|\mathbf{v}| \leq v_{\text{max}}$，其中$\mathbf{v} \in \mathbb{R}^3$是速度向量，姿态约束为$|\mathbf{q} - \mathbf{q}_{\text{target}}| \leq \epsilon$，其中$\mathbf{q} \in \mathbb{S}^3$是姿态的四元数表示。

GRPO算法通过更新策略梯度来优化控制策略，优化目标为：

$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot (R(s,a) - \lambda C(s,a))]$

其中，$C(s,a)$是约束成本函数，$\lambda$是拉格朗日乘子。这个过程确保在提高控制效率的同时，策略仍然遵循安全约束。

容错性通过对模拟的随机故障进行训练来实现，如推进器效率减少20%，确保策略能够适应不同的故障场景并最大化对接成功率。

SHAP值用于分解GRPO策略的动作贡献，提供可视化支持，帮助深入分析控制性能。



**Technical Specifications**

6DOF动力学建模使用航天器机器人工具包（SRT）或Orekit来模拟航天器的平移和旋转动力学。

平移动力学由Clohessy-Wiltshire方程描述，适用于低地球轨道，形式如下：

$$\ddot{\mathbf{r}} = -2 \boldsymbol{\omega}_0 \times \dot{\mathbf{r}} - \frac{\mu}{|\mathbf{R}|^3} \mathbf{r} + \frac{\mathbf{u}}{m} + \mathbf{d}$$

其中$\mathbf{r} \in \mathbb{R}^3$是相对位置，$\boldsymbol{\omega}_0$是轨道角速度，$\mu$是引力常数，$\mathbf{R}$是轨道半径，$\mathbf{u} \in \mathbb{R}^3$是推力，$m$是航天器质量，$\mathbf{d}$表示扰动，如故障。

基于四元数的旋转动力学由以下方程描述：

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \circ \begin{bmatrix} 0 & \boldsymbol{\omega} \end{bmatrix}, \quad \mathbf{I} \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times \mathbf{I} \boldsymbol{\omega} = \boldsymbol{\tau} + \mathbf{d}\tau$$

其中$\mathbf{q} \in \mathbb{S}^3$是姿态的四元数表示，$\boldsymbol{\omega} \in \mathbb{R}^3$是角速度，$\mathbf{I}$是惯性矩阵，$\boldsymbol{\tau} \in \mathbb{R}^3$是控制力矩，$\mathbf{d}\tau$表示扰动力矩。符号$\circ$表示四元数乘法。

受约束强化学习环境基于Ray RLlib和OpenAI Gym定义。

状态空间定义为：

$$s = [\mathbf{r}, \mathbf{v}, \mathbf{q}, \boldsymbol{\omega}, \mathbf{f}]$$

其中$\mathbf{f}$是故障指示器（例如，推进器效率$\eta \in [0,1]$）。

动作空间是连续的：

$$a = [\mathbf{u}, \boldsymbol{\tau}] \in \mathbb{R}^6$$

奖励函数为：

$$ R(s,a) = w_1 \cdot \mathbb{1}_{|\mathbf{r}| < 0.1, |\mathbf{q} - \mathbf{q}_{\text{target}}| < \epsilon} - w_2 \cdot |\mathbf{r}| - w_3 \cdot \int |\mathbf{u}|^2 \, dt - w_4 \cdot \mathbb{1}_{|\mathbf{v}| > v_{\text{max}}}$$

其中$w_i$是权重，$\mathbb{1}_{\cdot}$是指示函数。

安全约束定义为：

$$C(s,a) = \max(0, |\mathbf{v}| - v_{\text{max}}) + \max(0, |\mathbf{q} - \mathbf{q}_{\text{target}}| - \epsilon)$$

优化确保$\mathbb{E}[C(s,a)] \leq \delta$，其中$\delta$是约束阈值。

当对接成功（$|\mathbf{r}| < 0.1$米，$|\mathbf{q} - \mathbf{q}_{\text{target}}| < \epsilon$）或失败（超时1000步或不安全状态）时，终止条件满足。


# Constrained-Reinforcement-Dynamics-Docking-Simulation

CReDyS: Constrained Reinforcement Dynamics Docking Simulation



**Description**

CReDyS (Constrained Reinforcement Dynamics Docking Simulation) is an open-source simulation framework designed for the study of autonomous control strategies in spacecraft rendezvous and docking (RvD) tasks. It is based on Constrained Reinforcement Learning (CRL) and utilizes the Guided Reward Policy Optimization (GRPO) algorithm to develop a 6 degrees of freedom (6DOF) dynamics control strategy. CReDyS ensures safe docking even in the presence of disturbances such as thruster faults.

CReDyS provides a simulation environment based on random fault injection, mathematical state spaces, reward functions, and safety constraints, focusing on robust control strategies in high-dimensional dynamics. The framework offers a highly scalable tool for aerospace dynamics, control theory, and reinforcement learning, supporting precise modeling and policy analysis.



**Background**

Spacecraft rendezvous and docking tasks require high-precision control in the presence of dynamic disturbances, such as thruster faults or sensor noise. Traditional control methods, such as Model Predictive Control (MPC), often struggle to meet safety and robustness requirements in complex environments. Constrained Reinforcement Learning provides a theoretical framework for autonomous control by introducing safety constraints, such as velocity and attitude limits, within the Markov Decision Process (MDP). This ensures efficient and safe operation in dynamic environments.

CReDyS uses the GRPO algorithm to efficiently optimize policies $\pi(a|s)$ in the 6DOF state space, ensuring safety constraints are met through Lagrangian multipliers or Constrained Policy Optimization (CPO). The framework focuses on the application of GRPO in spacecraft docking, particularly in fault-tolerant and safety-critical scenarios.



**Methodology**

CReDyS implements autonomous spacecraft docking control strategies using the following methods:

Safety constraint control defines constraints on relative velocity and attitude deviation. Specifically, the velocity constraint is $|\mathbf{v}| \leq v_{\text{max}}$, where $\mathbf{v} \in \mathbb{R}^3$ is the velocity vector, and the attitude constraint is $|\mathbf{q} - \mathbf{q}_{\text{target}}| \leq \epsilon$, where $\mathbf{q} \in \mathbb{S}^3$ is the quaternion representation of attitude.

The GRPO algorithm optimizes control policies by updating the policy gradient, where the optimization target is:

$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot (R(s,a) - \lambda C(s,a))]$

Here, $C(s,a)$ is the constraint cost function, and $\lambda$ is the Lagrangian multiplier. This process ensures that while improving control efficiency, the policy adheres to safety constraints.

Fault tolerance is achieved by training the model on simulated random faults, such as a 20% reduction in thruster efficiency, ensuring the strategy adapts to different failure scenarios and maximizes docking success.

SHAP values are used to decompose the action contributions of the GRPO policy, providing visualizations to support in-depth analysis of control performance.



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
