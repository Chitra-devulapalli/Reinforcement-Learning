import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_NAME = "CartPole-v1"
MODEL_PATH = "best_model"

# Create environment with rendering
env = DummyVecEnv([lambda: gym.make(ENV_NAME, render_mode="human")])

# Load trained PPO model
model = PPO.load(MODEL_PATH)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done[0]:
        obs = env.reset()
