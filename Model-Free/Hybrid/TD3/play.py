import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import imageio

ENV_NAME = "Pendulum-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GIF_PATH = "/home/chitra/Documents/Reinforcement-Learning/media/td3_pendulum1.gif"

env = gym.make(ENV_NAME, render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# Actor Network
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, act_dim), nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs) * act_limit

actor = Actor().to(DEVICE)
actor.load_state_dict(torch.load("td3_actor_best.pth", map_location=DEVICE))
actor.eval()

obs, _ = env.reset()
done = False
total_reward = 0
frames = []

while not done:
    frame = env.render()
    frames.append(frame)

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        act = actor(obs_tensor).cpu().numpy()[0]
    obs, reward, terminated, truncated, _ = env.step(act)
    total_reward += reward
    done = terminated or truncated

print(f"Total Reward: {total_reward:.2f}")
env.close()

imageio.mimsave(GIF_PATH, frames, fps=30, loop=0)
print(f"Saved animation as '{GIF_PATH}'")