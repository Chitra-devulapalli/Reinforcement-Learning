import gymnasium as gym
import torch
import torch.nn as nn
import imageio
import numpy as np
import os

# Setup
ENV_NAME = "CartPole-v1"
MODEL_PATH = "best_a2c_model.pth"
GIF_PATH = "/home/chitra/Documents/Reinforcement-Learning/media/a2c_1cartpole.gif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.shared(obs)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# loading the environment and model
env = gym.make(ENV_NAME, render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = ActorCritic(obs_dim, act_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

frames = []
obs, _ = env.reset(seed=42)
obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
episode_reward = 0
done = False

while not done:
    frame = env.render()
    frames.append(frame)

    with torch.no_grad():
        logits, _ = model(obs)
        action = torch.argmax(logits, dim=-1).item()

    obs, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward
    done = terminated or truncated
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

env.close()

imageio.mimsave(GIF_PATH, frames, fps=30)
print(f"Final Episode Reward: {episode_reward:.2f}")
print(f"Saved animation as '{GIF_PATH}'")
