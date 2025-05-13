
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- Hyperparameters ---
ENV_NAME = "Pendulum-v1"
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 100
MAX_EPISODES = 200
MAX_STEPS = 200
POLICY_DELAY = 2
ACT_NOISE = 0.1
TARGET_NOISE = 0.2
NOISE_CLIP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment ---
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# --- Actor ---
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

# --- Critic ---
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

# --- Replay Buffer ---
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

# --- Initialize ---
actor = Actor().to(DEVICE)
actor_target = Actor().to(DEVICE)
actor_target.load_state_dict(actor.state_dict())

critic_1 = Critic().to(DEVICE)
critic_2 = Critic().to(DEVICE)
critic_target_1 = Critic().to(DEVICE)
critic_target_2 = Critic().to(DEVICE)
critic_target_1.load_state_dict(critic_1.state_dict())
critic_target_2.load_state_dict(critic_2.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_opt_1 = optim.Adam(critic_1.parameters(), lr=LR_CRITIC)
critic_opt_2 = optim.Adam(critic_2.parameters(), lr=LR_CRITIC)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

# --- Training ---
best_reward = -np.inf

def soft_update(source, target):
    for src, tgt in zip(source.parameters(), target.parameters()):
        tgt.data.copy_(TAU * src.data + (1 - TAU) * tgt.data)

def train():
    if len(replay_buffer) < BATCH_SIZE:
        return

    obs, act, rew, next_obs, done = replay_buffer.sample(BATCH_SIZE)

    with torch.no_grad():
        noise = (torch.randn_like(act) * TARGET_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (actor_target(next_obs) + noise).clamp(-act_limit, act_limit)
        target_q1 = critic_target_1(next_obs, next_action)
        target_q2 = critic_target_2(next_obs, next_action)
        target_q = torch.min(target_q1, target_q2)
        target = rew + GAMMA * (1 - done) * target_q

    current_q1 = critic_1(obs, act)
    current_q2 = critic_2(obs, act)
    critic_loss_1 = nn.MSELoss()(current_q1, target)
    critic_loss_2 = nn.MSELoss()(current_q2, target)

    critic_opt_1.zero_grad()
    critic_loss_1.backward()
    critic_opt_1.step()

    critic_opt_2.zero_grad()
    critic_loss_2.backward()
    critic_opt_2.step()

    global updates
    if updates % POLICY_DELAY == 0:
        actor_loss = -critic_1(obs, actor(obs)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        soft_update(actor, actor_target)

    soft_update(critic_1, critic_target_1)
    soft_update(critic_2, critic_target_2)

# --- Main Loop ---
updates = 0
for ep in range(MAX_EPISODES):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            act = actor(obs_tensor).cpu().numpy()[0]
        act = (act + np.random.normal(0, ACT_NOISE, size=act_dim)).clip(-act_limit, act_limit)

        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        replay_buffer.store((obs, act, reward, next_obs, float(done)))
        obs = next_obs
        total_reward += reward

        train()
        updates += 1

        if done:
            break

    print(f"Episode {ep + 1}, Reward: {total_reward:.2f}")

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(actor.state_dict(), "td3_actor_best.pth")
        print(f"✅ Saved new best model with reward: {best_reward:.2f}")

env.close()