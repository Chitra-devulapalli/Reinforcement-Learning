#!/usr/bin/env python3
# Play a trained A2C (SB3) policy on Isaac‑Lab’s Lift‑Cube–Franka task
# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────

import time

HEADLESS      = False
EPISODES      = 1
CHECKPOINT    = "best_model.zip"                     # <- point to your *Franka* model

ROOT = (
    "/home/chitra/Documents/Reinforcement-Learning/Model-Free/"
    "Hybrid/A2C_sb/arm_isaaclab"                 # <- put the lift‑cube run here
)
TASK_NAME = "Isaac-Lift-Cube-Franka-v0"              # <- **changed**

# ───────────────────────────────────────────────────────────────────────────────
# 0.  Launch Isaac‑Sim
# ───────────────────────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher
app_launcher   = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

# ───────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ───────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

# ► NEW: task‑specific config class
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg_PLAY,
)

# ───────────────────────────────────────────────────────────────────────────────
# 2.  SB3 helper patch (unchanged)
# ───────────────────────────────────────────────────────────────────────────────
class PatchedSb3VecEnv(Sb3VecEnvWrapper):
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

# ───────────────────────────────────────────────────────────────────────────────
# 3.  Paths
# ───────────────────────────────────────────────────────────────────────────────
ROOT_PATH  = Path(ROOT)
MODEL_PATH = ROOT_PATH / CHECKPOINT

# ───────────────────────────────────────────────────────────────────────────────
# 4.  Build the environment
# ───────────────────────────────────────────────────────────────────────────────
env_cfg               = FrankaCubeLiftEnvCfg_PLAY()   # ← NEW
env_cfg.scene.num_envs = 1                      # optional: keep it light
# env_cfg.observations.policy.enable_corruption = False
# env_cfg.scene.object.init_state.pos = (0.5, 0.00, 0.005)
# env_cfg.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
env                   = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)

# Convert multi‑agent → single‑agent (not needed here, but harmless)
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

# Wrap into SB3‑compatible VecEnv
env = PatchedSb3VecEnv(env)

# ───────────────────────────────────────────────────────────────────────────────
# 5.  Load trained policy
# ───────────────────────────────────────────────────────────────────────────────
model = PPO.load(str(MODEL_PATH))
print(f"[INFO] Loaded policy from {MODEL_PATH}")

# ───────────────────────────────────────────────────────────────────────────────
# 6.  Roll‑out loop
# ───────────────────────────────────────────────────────────────────────────────
for ep in range(EPISODES):
    obs      = env.reset()
    done     = np.zeros(env.num_envs, dtype=bool)
    ret_ep   = np.zeros(env.num_envs)

    while not done.any():                       # stop when the *first* env finishes
        act, _      = model.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(act)

        ret_ep += rew
        if HEADLESS:
            time.sleep(0.01)                  # slow down for better visualisation
            simulation_app.update()

    print(f"Episode {ep+1}:  mean return = {ret_ep.mean():.2f}")

# ───────────────────────────────────────────────────────────────────────────────
# 7.  Clean‑up
# ───────────────────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
print("✅ Playback finished.")
