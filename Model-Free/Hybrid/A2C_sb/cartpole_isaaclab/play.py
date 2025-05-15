HEADLESS      = False
EPISODES      = 1
CHECKPOINT    = "best_model.zip"

ROOT = (
    "/home/chitra/Documents/Reinforcement-Learning/Model-Free/"
    "Hybrid/A2C_sb/cartpole_isaaclab"
)
TASK_NAME = "Isaac-Cartpole-Direct-v0"

from isaaclab.app import AppLauncher

app_launcher   = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app             

from pathlib import Path
import gymnasium as gym
from stable_baselines3 import A2C

import numpy as np

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

# sb3 wrapper
class PatchedSb3VecEnv(Sb3VecEnvWrapper):
    def env_is_wrapped(self, wrapper_class, indices=None):
        # needed because VecMonitor queries this method
        return [False] * self.num_envs

ROOT_PATH   = Path(ROOT)
MODEL_PATH  = ROOT_PATH / CHECKPOINT

env_cfg = CartpoleEnvCfg()
env_cfg.scene.num_envs = 1  
env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)

# Convert multi-agent → single-agent for SB3
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

# Wrap into vectorised interface (1 env)
env = PatchedSb3VecEnv(env)

model = A2C.load(str(MODEL_PATH))
print(f"[INFO] Loaded policy from {MODEL_PATH}")

for ep in range(EPISODES):
    obs = env.reset()            # obs shape: (4096, obs_dim)
    done  = np.zeros(env.num_envs, dtype=bool)
    ep_return = np.zeros(env.num_envs)

    # run until any one env finishes
    while not done.any():           
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        ep_return += reward 
        if HEADLESS:
            simulation_app.update()

    print(f"Episode {ep+1}: mean return = {ep_return.mean():.2f}")
    
env.close()
simulation_app.close()
print("Playback finished.")
