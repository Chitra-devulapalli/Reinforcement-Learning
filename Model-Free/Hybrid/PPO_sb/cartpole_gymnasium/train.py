import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ENV_NAME = "CartPole-v1"
MODEL_PATH = "ppo_cartpole_best"
PLOT_PATH = Path.home() / "Documents/Reinforcement-Learning/media/rewards_ppo2.png"

class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

def make_env():
    return gym.make(ENV_NAME)

vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)

eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

reward_callback = RewardTracker()

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.0,
    learning_rate=3e-4,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
    render=False,
)

model.learn(total_timesteps=10000, callback=[eval_callback, reward_callback])

PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(reward_callback.episode_rewards)
plt.title("PPO on CartPole-v1")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()
print(f"Saved reward curve at {PLOT_PATH}")
