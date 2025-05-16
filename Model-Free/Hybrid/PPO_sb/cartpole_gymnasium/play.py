import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import imageio

ENV_NAME = "CartPole-v1"
MODEL_PATH = "best_model"
GIF_PATH = "/home/chitra/Documents/Reinforcement-Learning/media/ppo2_cartpole.gif"

# Create environment with rendering
env = DummyVecEnv([lambda: gym.make(ENV_NAME, render_mode="rgb_array")])

# Load trained PPO model
model = PPO.load(MODEL_PATH)

frames = []

obs = env.reset()
while True:
    frame = env.render()
    frames.append(frame)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done[0]:
        obs = env.reset()
        break

env.close()
imageio.mimsave(GIF_PATH, frames, fps=30, loop=0)
print(f"Saved animation as '{GIF_PATH}'")