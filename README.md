# Reinforcement Learning Tutorial for Robotics

<p align="center">
  This repository is an introduction to popular reinforcement learning algorithms, implemented using PyTorch or Stable Baselines3. These algorithms are tested across various simulation environments to help build an intuition in how RL methods are applied in robotics.
</p>

<p align="center">
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/titleimg.png" alt="Title Image" width="500"/>
</p>

## Table of Contents
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Algorithms](#algorithms)
- [Examples using Isaac Lab](#examples-using-isaac-lab)

## Introduction
Reinforcement Learning (RL) has emerged as a powerful framework in robot learning, enabling agents to discover optimal control policies through trial and error. From balancing a cartpole to performing complex path planning, RL offers a principled way to map sensory inputs to actions using reward-driven feedback.

This repository provides a hands-on, tutorial-style introduction to widely used RL algorithms, implemented using both PyTorch (from scratch) and Stable Baselines3 (SB3). It covers a range of discrete and continuous control tasks, exploring value-based methods (such as Q-learning and DQN) and hybrid actor-critic techniques (like A2C, PPO, DDPG, TD3, and SAC).

Robotics-specific examples using Isaac Sim and Isaac Lab are included for Cartpole balancing and Franka Panda reacher (trained using A2C and PPO).

## Environment Setup
1. Follow the [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to setup Isaac Sim 4.5.0 and Isaac Lab in a conda environment using pip on Ubuntu 22.04.
2. Run the following command to check for successfull setup:
   ```
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
   ```
3. Clone this repository:
   ```
   https://github.com/Chitra-devulapalli/Reinforcement-Learning.git
   ```
4. Run the following command to install additional python libraries required to run algorithms in this repository:
   ```
   pip install requirements.txt
   ```

## Algorithms

### 1. Q-learning
Q-learning is a foundational **value-based reinforcement learning algorithm**. It estimates the optimal action-value function `Q(s, a)` using the Bellman equation and updates the table through temporal difference learning. This method is typically used in discrete environments.

Algorithm: 
At each step:
- Choose action using ε-greedy policy from Q-table  
- Execute action, observe reward and next state  
- Update Q-value: <br>
  `Q(s, a) ← Q(s, a) + α [r + γ ⋅ max_a' Q(s', a') − Q(s, a)]` <br>
- Repeat until convergence or for a fixed number of episodes

#### Usage: 
```
cd Q_Learning 
python Q_learning.py
```

Results:
<p align="center">
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/Model-Free/Value-Based/Q-Learning/q_learning_heatmaps_and_policy.png" alt="Title Image" width="500"/>
</p>


### 2. DQN
Deep Q-Network (DQN) is an extension of Q-learning that uses a neural network to approximate the Q-value function. This allows it to scale to environments with large or continuous state spaces.

**Algorithm:**  
At each step:
- Encode the state as input to a neural network (the Q-network)
- Choose action using ε-greedy policy from predicted Q-values
- Execute action, observe reward and next state
- Store the transition `(s, a, r, s', done)` in a replay buffer
- Sample a batch of transitions from the buffer
- Compute target Q-value:
  `target = r + γ ⋅ max_a' Q_target(s', a')` <br>
- Update Q-network by minimizing: <br>
  `loss = (Q(s, a) - target)^2` <br>
- Periodically update the target network

#### Usage: 
```bash
cd Value/DQN
python train_dqn.py
```

Results:
<p align="center">
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/Model-Free/Value-Based/Q-Learning/dqn_heatmaps_and_policy-100.png" alt="Title Image" width="500"/>
</p>

### 3. Advantage Actor Critic (A2C)
Advantage Actor-Critic (A2C) is a synchronous, hybrid reinforcement learning algorithm that combines both value-based and policy-based approaches. The actor selects actions according to a policy, while the critic estimates the value function to guide learning. A2C improves training stability by reducing variance via the advantage estimate.

**Algorithm:**
- Use a shared network to produce both action logits (actor) and state value (critic)
- Sample actions from the softmax policy distribution
- Compute value estimates and advantages:  
  `advantage = (r + γ * V(s') - V(s))`
- Update actor to maximize expected return:  
  `loss_actor = -log_prob(action) * advantage`
- Update critic via mean squared error loss:  
  `loss_critic = (target - value)^2`
- Backpropagate total loss:  
  `loss = loss_actor + loss_critic`

#### Usage: 
```
cd Hybrid/A2C # for pytorch version from scratch
python train.py
python play.py

cd Hybrid/A2C_sb # for stable baselines version
python train.py
python play.py
```

#### Results:
Pytorch Implementation:
<p>
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/a2c_1cartpole.gif?raw=true" alt="A2C PyTorch" width="34%" />
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/a2c1_rewards.png?raw=true" alt="Rewards Plot PyTorch" width="45%" />
</p>

Stable Baselines Implementation:
<p>
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/a2c_2cartpole.gif?raw=true" alt="A2C SB3" width="34%" />
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/a2c2_rewards.png?raw=true" alt="Rewards Plot SB3" width="45%" />
</p>


### 4. Proximal Policy Optimization (PPO)
Proximal Policy Optimization (PPO) is a widely used policy gradient method that improves training stability by clipping the probability ratio between the new and old policies. This avoids large, destabilizing policy updates while still enabling efficient learning.

**Algorithm:**
- Collect trajectories using the current policy.
- Compute Generalized Advantage Estimation (GAE) for stable advantage calculation.
- Optimize the clipped surrogate objective:  
  `L_clip(θ) = E_t [ min(r_t(θ) * Â_t, clip(r_t(θ), 1 - ε, 1 + ε) * Â_t) ]` <br>
   where `r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)`  
- Use multiple epochs of minibatch updates on collected data.
- Add entropy bonus and critic value loss for improved exploration and stability.

#### Usage:

```bash
cd Hybrid/PPO
python train.py # for pytorch version from scratch
python play.py

cd Hybrid/PPO_sb # for stable baselines version
python train.py
python play.py
``` 
#### Results:
Pytorch Implementation:
<p>
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/ppo1_cartpole.gif" alt="PPO Pytorch" width="45%" />
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/ppo1_rewards.png" alt="Rewards Plot PyTorch" width="34%" />
</p>

Stable Baselines Implementation:
<p>
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/ppo2_cartpole.gif" alt="PPO SB3"width="45%" />
  <img src="https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/rewards_ppo2.png" alt="Rewards Plot SB3" width="34%" />
</p>

### 5. Deep Deterministic Policy Gradient (DDPG)
Deep Deterministic Policy Gradient (DDPG) is an off-policy, model-free reinforcement learning algorithm designed for environments with continuous action spaces. It combines the deterministic policy gradient algorithm with the actor-critic architecture and uses target networks along with a replay buffer to stabilize training.

**Algorithm:**
- Use an actor network to output deterministic actions and a critic network to estimate Q-values.
- Add exploration noise (e.g., Ornstein-Uhlenbeck) to the actor’s action during training.
- Store transitions `(s, a, r, s', done)` in a replay buffer.
- Sample mini-batches from the buffer and update:
  - **Critic**: using the Bellman target `Q_target = r + γ * Q'(s', μ'(s'))`
  - **Actor**: via policy gradient that maximizes the critic output.
- Soft update the target networks:  
  `θ_target ← τ * θ + (1 - τ) * θ_target`

#### Usage:

```bash
cd Hybrid/DDPG
python train.py # for pytorch version from scratch
python play.py

cd Hybrid/DDPG_sb
python train.py # for stable baselines version
python play.py
```

#### Results:

### 6. Twin Delayed Deep Deterministic Policy Gradient (TD3)
Twin Delayed DDPG (TD3) improves upon DDPG by addressing overestimation bias in Q-value estimates and improving policy stability. It uses two critic networks to take the minimum value estimate and delays actor updates, while adding clipped noise to target actions for smoother value approximation.

**Algorithm:**
- Maintain two Q-functions \( Q_1 \), \( Q_2 \) and a deterministic actor.
- Use **target policy smoothing**: add clipped noise to target actions.
- Use **twin critics**: take the minimum of \( Q_1 \), \( Q_2 \) during target computation.
- Delay actor updates (e.g., update actor every 2 critic updates).
- Perform soft updates for target networks.

#### Usage:

```bash
cd Hybrid/TD3 # for pytorch version from scratch
python train.py
python play.py

cd Hybrid/TD3 # for stable baselines version
python train.py
python play.py
```

#### Results:

### 7. Soft Actor-Critic (SAC)
Soft Actor-Critic (SAC) is an off-policy, actor-critic algorithm designed for continuous action spaces. It maximizes both the expected return and policy entropy, encouraging exploration and stability. SAC uses stochastic policies, twin Q-networks to reduce overestimation bias, and soft updates for target networks.

**Algorithm:**
- Learn a stochastic policy `π(a|s)` using reparameterization and entropy regularization.
- Train two Q-value critics and take the minimum to form the value target.
- Optimize the actor to maximize Q-values minus entropy penalty: <br>
  `J_π = E_{s_t ~ D, a_t ~ π}[α * log π(a_t|s_t) - Q(s_t, a_t)]`
- Use target networks for stability in critic updates.
- Perform soft updates:  
  `θ_target ← τ * θ + (1 - τ) * θ_target`

#### Usage:

```bash
cd Hybrid/SAC # for pytorch version from scratch 
python train.py
python play.py

cd Hybrid/SAC_sb # for stable baselines version
python train.py
python play.py
```

#### Results:

## Examples using Isaac Lab

We explored a range of robotics environments within Isaac Sim/Isaac Lab, using Stable Baselines to implement selected algorithms for tasks like balancing an inverted pendulum on a cartpole and guiding the end effector of a Franka robot to reach a cube.

| Environment and Algo        | Training                | Result                  |
|-----------------------------|-------------------------|-------------------------|
| Cartpole using A2C          | ![Training Gif](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/cartpole_train.gif) | ![Result Gif](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/cartpole_play.gif) |
| Franka Reacher using A2C    | ![Training Gif](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/a2ctrain_arm.gif) | ![Result Gif](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/a2c_res.gif) |
| Franka Reacher using PPO    | ![Training Gif](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/ppotrain_arm.gif) | ![Result Gif](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/ppo_res.gif) |

NOTE: The Isaac Lab Reacher environment that is compatible with Stable Baselines does not exist (Isaac-Reach-Franka-v0 is incompatible with SB3). Therefore, the Isaac-Lift-Cube-Franka-v0 environment has been modified by adjusting the ```RewardsCfg``` found in ```..../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift``` to simplify the task, focusing only on reaching the cube.

```
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=18.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.2}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=1.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=0.50,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

