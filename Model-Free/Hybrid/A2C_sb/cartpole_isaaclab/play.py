#!/usr/bin/env python3
# Play a trained A2C policy on Isaac-Lab’s Cartpole-Direct task.
# All settings are hard-coded: edit the constants section to change behaviour.

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS – tweak values here, no CLI parsing required
# ──────────────────────────────────────────────────────────────────────────────
HEADLESS      = False          # True ➜ no GUI, faster; False ➜ show viewport
EPISODES      = 1             # how many roll-outs to play
CHECKPOINT    = "best_model.zip"  # file name inside ROOT / "models" (or ROOT)

ROOT = (
    "/home/chitra/Documents/Reinforcement-Learning/Model-Free/"
    "Hybrid/A2C_sb/cartpole_isaaclab"
)
TASK_NAME = "Isaac-Cartpole-Direct-v0"

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Launch Isaac Sim
# ──────────────────────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

app_launcher   = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app               # keep alive during playback

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import A2C

import numpy as np

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Patch so SB3 helpers work with Isaac-Lab VecEnv
# ──────────────────────────────────────────────────────────────────────────────
class PatchedSb3VecEnv(Sb3VecEnvWrapper):
    def env_is_wrapped(self, wrapper_class, indices=None):
        # needed because VecMonitor queries this method
        return [False] * self.num_envs

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT_PATH   = Path(ROOT)
MODEL_PATH  = ROOT_PATH / CHECKPOINT

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Build the environment
# ──────────────────────────────────────────────────────────────────────────────
env_cfg = CartpoleEnvCfg()                     # default config
env_cfg.scene.num_envs = 1  
env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)

# Convert multi-agent → single-agent for SB3
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

# Wrap into vectorised interface (1 env)
env = PatchedSb3VecEnv(env)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Load trained policy
# ──────────────────────────────────────────────────────────────────────────────
model = A2C.load(str(MODEL_PATH))
print(f"[INFO] Loaded policy from {MODEL_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Roll-out loop
# ──────────────────────────────────────────────────────────────────────────────
# ── Roll-out loop ────────────────────────────────────────────────
for ep in range(EPISODES):
    obs = env.reset()            # obs shape: (4096, obs_dim)
    done  = np.zeros(env.num_envs, dtype=bool)
    ep_return = np.zeros(env.num_envs)

    # run until *all* sub‑envs finish this episode
    while not done.any():           
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        ep_return += reward         # vector add
        if HEADLESS:
            simulation_app.update()

    print(f"Episode {ep+1}: mean return = {ep_return.mean():.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Clean-up
# ──────────────────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
print("✅  Playback finished.")
