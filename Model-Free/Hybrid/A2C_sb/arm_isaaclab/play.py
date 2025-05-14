import time

HEADLESS      = False
EPISODES      = 1
CHECKPOINT    = "best_model.zip"

ROOT = (
    "/home/chitra/Documents/Reinforcement-Learning/Model-Free/"
    "Hybrid/A2C_sb/arm_isaaclab"
)
TASK_NAME = "Isaac-Lift-Cube-Franka-v0" 

from isaaclab.app import AppLauncher
app_launcher   = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg_PLAY,
)

# sb3 wrapper
class PatchedSb3VecEnv(Sb3VecEnvWrapper):
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

ROOT_PATH  = Path(ROOT)
MODEL_PATH = ROOT_PATH / CHECKPOINT

# building the environment
env_cfg = FrankaCubeLiftEnvCfg_PLAY() 
env_cfg.scene.num_envs = 1

env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)

if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

# Wrap into SB3‑compatible VecEnv
env = PatchedSb3VecEnv(env)

model = PPO.load(str(MODEL_PATH))
print(f"[INFO] Loaded policy from {MODEL_PATH}")

# rollout loop
for ep in range(EPISODES):
    obs = env.reset()
    done = np.zeros(env.num_envs, dtype=bool)
    ret_ep = np.zeros(env.num_envs)

    while not done.any(): # stop when first env finishes
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(act)

        ret_ep += rew
        if HEADLESS:
            time.sleep(0.01)                  # slow down for better visualisation
            simulation_app.update()

    print(f"Episode {ep+1}:  mean return = {ret_ep.mean():.2f}")

env.close()
simulation_app.close()
print("Playback finished.")
