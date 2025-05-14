#!/usr/bin/env python3
# Play a trained PPO (SB3) policy on Isaac‑Lab’s Lift‑Cube–Franka task
# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

HEADLESS   = False
EPISODES   = 1
CHECKPOINT = "best_model.zip"  # <‑ name of the saved PPO weights

ROOT = (
    "/home/chitra/Documents/Reinforcement-Learning/Model-Free/"
    "Hybrid/PPO_sb/arm_isaaclab"   # <‑ directory that contains the checkpoint
)
TASK_NAME = "Isaac-Lift-Cube-Franka-v0"

# ───────────────────────────────────────────────────────────────────────────────
# 0.  Launch Isaac‑Sim
# ───────────────────────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher
app_launcher   = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

# ───────────────────────────────────────────────────────────────────────────────
# 1.  Env helpers / config import
# ───────────────────────────────────────────────────────────────────────────────
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg_PLAY,
)

class PatchedSb3VecEnv(Sb3VecEnvWrapper):
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

# ───────────────────────────────────────────────────────────────────────────────
# 2.  Paths & env construction
# ───────────────────────────────────────────────────────────────────────────────
ROOT_PATH  = Path(ROOT)
MODEL_PATH = ROOT_PATH / CHECKPOINT

env_cfg               = FrankaCubeLiftEnvCfg_PLAY()
env_cfg.scene.num_envs = 1                # small grid for visual demo
# env_cfg.randomization = None
# env_cfg.observations.policy.enable_corruption = False
# env_cfg.scene.object.init_state.pos = (0.5, 0.00, 0.005)
# env_cfg.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)

if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

env = PatchedSb3VecEnv(env)

# ───────────────────────────────────────────────────────────────────────────────
# 3.  Load PPO policy
# ───────────────────────────────────────────────────────────────────────────────
model = PPO.load(str(MODEL_PATH))
print(f"[INFO] Loaded PPO policy from {MODEL_PATH}")

# ───────────────────────────────────────────────────────────────────────────────
# 4.  Roll‑out loop
# ───────────────────────────────────────────────────────────────────────────────
for ep in range(EPISODES):
    obs    = env.reset()
    done   = np.zeros(env.num_envs, dtype=bool)
    ret_ep = np.zeros(env.num_envs)

    while not done.any():                    # stop when *any* env finishes
        action, _          = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ret_ep            += reward
        if not HEADLESS:
            time.sleep(0.01)                 # slow down for clearer viewing
            simulation_app.update()

    print(f"Episode {ep+1}: mean return = {ret_ep.mean():.2f}")

# ───────────────────────────────────────────────────────────────────────────────
# 5.  Clean‑up
# ───────────────────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
print("✅  Playback finished.")
