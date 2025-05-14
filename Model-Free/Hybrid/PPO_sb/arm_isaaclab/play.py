import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from isaaclab.app import AppLauncher
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg_PLAY,
)

HEADLESS = False
EPISODES = 1
CHECKPOINT = "best_model.zip"
ROOT = (
    "/home/chitra/Documents/Reinforcement-Learning/Model-Free/"
    "Hybrid/PPO_sb/arm_isaaclab"
)
TASK_NAME = "Isaac-Lift-Cube-Franka-v0"

# Launch Isaac‑Sim
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

class PatchedSb3VecEnv(Sb3VecEnvWrapper):
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

# Paths & env construction
ROOT_PATH = Path(ROOT)
MODEL_PATH = ROOT_PATH / CHECKPOINT

env_cfg = FrankaCubeLiftEnvCfg_PLAY()
env_cfg.scene.num_envs = 1
env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)

if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

env = PatchedSb3VecEnv(env)

# Load PPO policy
model = PPO.load(str(MODEL_PATH))
print(f"[INFO] Loaded PPO policy from {MODEL_PATH}")

# Roll‑out loop
for ep in range(EPISODES):
    obs = env.reset()
    done = np.zeros(env.num_envs, dtype=bool)
    ret_ep = np.zeros(env.num_envs)

    while not done.any():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ret_ep += reward
        if not HEADLESS:
            time.sleep(0.01)
            simulation_app.update()

    print(f"Episode {ep+1}: mean return = {ret_ep.mean():.2f}")

# Clean‑up
env.close()
simulation_app.close()
print("Playback finished.")
