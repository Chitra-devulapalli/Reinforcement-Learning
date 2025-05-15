import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

ENV_NAME = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "ppo_cartpole_best.pth"

#ActorCritic Network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.shared(obs)
        return self.actor(x), self.critic(x)

    def act(self, obs):
        logits, _ = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

env = gym.make(ENV_NAME, render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = ActorCritic(obs_dim, act_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

obs, _ = env.reset(seed=42)
episode_reward = 0

while True:
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    action = model.act(obs_tensor).item()

    obs, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward

    if terminated or truncated:
        print(f"Final Episode Reward: {episode_reward:.2f}")
        break  # Exit after one episode

env.close()
