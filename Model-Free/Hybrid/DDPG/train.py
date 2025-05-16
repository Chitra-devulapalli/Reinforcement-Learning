import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ENV_NAME = "Pendulum-v1"
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
MAX_EPISODES = 200
MAX_STEPS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "ddpg_actor_best.pth"
REWARD_PLOT_PATH = Path.home() / "Documents/Reinforcement-Learning/media/ddpg_rewards1.png"

env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

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

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.tensor(obs, dtype=torch.float32, device=DEVICE),
            torch.tensor(act, dtype=torch.float32, device=DEVICE),
            torch.tensor(rew, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(next_obs, dtype=torch.float32, device=DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class OUNoise:
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.zeros(act_dim)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(act_dim)
        self.state += dx
        return self.state

actor = Actor().to(DEVICE)
critic = Critic().to(DEVICE)
target_actor = Actor().to(DEVICE)
target_critic = Critic().to(DEVICE)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
replay_buffer = ReplayBuffer(BUFFER_SIZE)
noise = OUNoise()
best_reward = -float("inf")
episode_rewards = []

def update():
    if len(replay_buffer) < BATCH_SIZE:
        return
    obs, act, rew, next_obs, done = replay_buffer.sample(BATCH_SIZE)
    with torch.no_grad():
        next_act = target_actor(next_obs)
        target_q = target_critic(next_obs, next_act)
        target = rew + GAMMA * (1 - done) * target_q
    current_q = critic(obs, act)
    critic_loss = nn.MSELoss()(current_q, target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    actor_loss = -critic(obs, actor(obs)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    for param, target_param in zip(critic.parameters(), target_critic.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

for ep in range(MAX_EPISODES):
    obs, _ = env.reset(seed=42)
    noise.reset()
    episode_reward = 0
    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            act = actor(obs_tensor).cpu().numpy()[0]
        act = np.clip(act + noise.sample(), -act_limit, act_limit)
        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        replay_buffer.store((obs, act, reward, next_obs, float(done)))
        obs = next_obs
        episode_reward += reward
        update()
        if done:
            break
    print(f"Episode {ep+1} | Reward: {episode_reward:.2f}")
    episode_rewards.append(episode_reward)
    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(actor.state_dict(), SAVE_PATH)
        print(f"Saved best model with reward: {best_reward:.2f}")

env.close()

REWARD_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.title("DDPG on Pendulum-v1")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(REWARD_PLOT_PATH, dpi=300)
plt.close()
print(f"Saved reward curve at {REWARD_PLOT_PATH}")
