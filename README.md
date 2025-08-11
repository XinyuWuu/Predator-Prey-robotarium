# Predator-Prey Game: Structure and Objectives

This simulation outlines a classic **pursuit-evasion game** set in a 2D environment. It's designed to test control algorithms and artificial intelligence in a dynamic, competitive scenario.

https://github.com/user-attachments/assets/96e8852a-2e1b-46f0-afbb-5ff59dec3bba

https://github.com/user-attachments/assets/6052e066-b00e-4928-9078-5ffd7386b9ec

## Agents and Objectives 🤖

The game involves two agents with conflicting goals:

* **The Prey (Evader):** Its main objective is to navigate to a specific **target location** while avoiding obstacles and, most importantly, the predator.
* **The Predator (Pursuer):** Its main objective is to **chase and capture** the prey before the prey can reach its destination. A "capture" occurs when the distance between the two agents falls below a certain threshold (e.g., the sum of their collision radii).

---

## The Environment 🗺️

> [!NOTE]
> Original environment from https://github.com/robotarium/robotarium_python_simulator
> modified by HITSZ ML Lab.

> [!NOTE]
> You have to first install robotarium_python_simulator-master.zip by `pip install .` with `matplotlib==3.7.3`.

The agents operate within a well-defined space with several key features:

* **Boundaries:** The world is a **bounded 2D space**, which prevents the agents from moving infinitely.
* **Static Obstacles:** The environment contains **fixed, circular obstacles**. These force the agents to use more complex pathfinding and maneuvering rather than just moving in a straight line.
* **Target Zone:** A specific **goal area** is defined for the prey. Reaching this zone is the prey's win condition.

## Agent Dynamics & Asymmetry ⚙️

Both agents are modeled as **unicycle robots**, meaning their state is defined by their position and heading `(x, y, θ)` and controlled via linear and angular velocities `(v, w)`.

A critical aspect of the game's design is the **asymmetry in their abilities**, which creates a strategic trade-off:

* **Predator:** Possesses a **higher maximum linear speed**, making it faster in open areas.
* **Prey:** Has a **higher maximum angular velocity**, making it more agile and capable of executing sharper and faster turns.

This asymmetry means neither agent has a definitive advantage. The predator must anticipate the prey's agile dodges, while the prey must use its maneuverability to evade the faster pursuer.

## Game Termination: Winning & Losing 🏆

An episode of the game concludes when one of the following conditions is met:

1.  **Prey Wins:** The prey successfully enters the designated target zone.
2.  **Predator Wins:** The predator closes the distance and captures the prey.

# RL Solution Breakdown: Soft Actor-Critic

## RL Problem Setup 🤖

The problem is framed as a standard RL task with the following specifications:

* **Observation**: The agent's state includes the position and orientation of both the prey and the predator ($x_{prey}, y_{prey}, \cos\theta_{prey}, \sin\theta_{prey}, x_{pred}, y_{pred}, \cos\theta_{pred}, \sin\theta_{pred}$).
* **Action**: The agent outputs a continuous action vector `(v, w)`, which corresponds to the linear and angular velocity it should apply.
* **Reward Function**: The total reward is a composite signal designed to guide the agent's behavior. It's the sum of four distinct components: $r_{g} + r_{c} + r_{d} + r_{w}$.
    * **$r_{g}$ (Approaching Goal)**: A positive reward for moving towards the goal , calculated as $r_{g}=<V_{t}\cdot V_{d}> / {V_{max}}$.
    * **$r_{c}$ (Collision)**: A penalty for colliding with obstacles , calculated as $r_{c}=-(1-\frac{\Delta t}{\Delta})$.
    * **$r_{d}$ (Death)**: A large penalty of **-100** for being captured.
    * **$r_{w}$ (Win)**: A large reward of **+100** for successfully reaching the goal.

---
## Network Architecture 🧠

The solution utilizes several Multi-Layer Perceptrons (MLPs) as function approximators.

* **Actor (Policy) & Critic (Q-Value) Networks**: Both the actor and the twin critic networks use an architecture with three hidden layers, each containing **256 units**. The actor network outputs the parameters for a policy distribution from which actions are sampled.


## Training Process 📈

The model is trained using the SAC algorithm, which involves storing experiences in a replay buffer and performing gradient updates on the networks. The training is governed by the following update steps:

1.  **Critic (Q-Network) Update**: The two Q-networks are updated to minimize the Bellman error, using a target Q-value that includes the reward, the discounted future Q-value, and a policy entropy term.
2.  **Actor (Policy) Update**: The policy network is updated by maximizing the expected Q-value of its actions and the policy's entropy.
3.  **Temperature (α) Update**: The temperature parameter $\alpha$ is automatically tuned to balance the reward and entropy objectives by targeting a specific entropy level.
4.  **Target Network Update**: The weights of the target Q-networks are updated slowly to track the main networks using Polyak averaging with a factor of $\tau = 0.995$.
