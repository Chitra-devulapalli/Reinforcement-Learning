import gymnasium as gym
import imageio
import torch
from stable_baselines3 import A2C

ENV_NAME = "CartPole-v1"
MODEL_PATH = "best_model.zip"
GIF_PATH = "/home/chitra/Documents/Reinforcement-Learning/media/a2c_2cartpole.gif"

# Create environment with rgb_array rendering
env = gym.make(ENV_NAME, render_mode="rgb_array")
model = A2C.load(MODEL_PATH)

obs, _ = env.reset(seed=42)
frames = []
done = False

while not done:
    frame = env.render()
    frames.append(frame)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()

# Save the collected frames as a GIF
imageio.mimsave(GIF_PATH, frames, fps=30)
print(f" GIF saved at: {GIF_PATH}")
