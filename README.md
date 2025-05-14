# Reinforcement Learning Tutorial for Robotics

<p align="center">
  This repository is an introduction to popular reinforcement learning algorithms, implemented using PyTorch or Stable Baselines3. These algorithms are tested across various simulation environments to help build an intuition in how RL methods are applied in robotics.
</p>

![alt text](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/media/titleimg.png)

## Table of Contents
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Algorithms](#algorithms)
- [Examples using Isaac Lab](#examples-using-isaac-lab)

## Introduction

## Environment Setup

## Algorithms

### 1. Q-learning

Algorithm: aaaaaaaaaaaaaaaaaaaaaaaa

Usage: 
```
python something #fghj
pythin somegh #sdfgh
```

Results:
![alt text](https://github.com/Chitra-devulapalli/Reinforcement-Learning/blob/main/Model-Free/Value-Based/Q-Learning/q_learning_heatmaps_and_policy.png)

### 2. DQN

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

