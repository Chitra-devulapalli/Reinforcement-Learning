import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import imageio

ENV_NAME = "Pendulum-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GIF_PATH = "/home/chitra/Documents/Reinforcement-Learning/media/sac1_pendulum.gif"


env = gym.make(ENV_NAME, render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

actor = Actor().to(DEVICE)
actor.load_state_dict(torch.load("sac_actor_best.pth", map_location=DEVICE))
actor.eval()

obs, _ = env.reset()
done = False
total_reward = 0.0
frames = []
while not done:
    frame = env.render()
    frames.append(frame)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        mean, _ = actor(obs_tensor)
        action = torch.tanh(mean) * act_limit
        action = action.cpu().numpy()[0]
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated
print(f"Episode finished with total reward: {total_reward:.2f}")

env.close()
imageio.mimsave(GIF_PATH, frames, fps=30, loop=0)
print(f"Saved animation as '{GIF_PATH}'")