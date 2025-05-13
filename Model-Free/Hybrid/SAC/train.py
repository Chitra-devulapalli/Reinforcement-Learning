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
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
ALPHA = 0.2  # Entropy coefficient
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
MAX_EPISODES = 300
MAX_STEPS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# --- Actor Network (outputs mean and log_std) ---
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

    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action * act_limit, log_prob

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
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

# --- Initialize Networks ---
actor = Actor().to(DEVICE)
critic1 = Critic().to(DEVICE)
critic2 = Critic().to(DEVICE)
target_critic1 = Critic().to(DEVICE)
target_critic2 = Critic().to(DEVICE)

target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=LR_CRITIC)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

best_reward = -float("inf")

# --- Training Loop ---
def update():
    if len(replay_buffer) < BATCH_SIZE:
        return

    obs, act, rew, next_obs, done = replay_buffer.sample(BATCH_SIZE)

    with torch.no_grad():
        next_action, log_prob = actor.sample(next_obs)
        target_q1 = target_critic1(next_obs, next_action)
        target_q2 = target_critic2(next_obs, next_action)
        target_q = torch.min(target_q1, target_q2) - ALPHA * log_prob
        target = rew + GAMMA * (1 - done) * target_q

    current_q1 = critic1(obs, act)
    current_q2 = critic2(obs, act)
    critic1_loss = nn.MSELoss()(current_q1, target)
    critic2_loss = nn.MSELoss()(current_q2, target)

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    action_sampled, log_prob = actor.sample(obs)
    q1_pi = critic1(obs, action_sampled)
    q2_pi = critic2(obs, action_sampled)
    min_q = torch.min(q1_pi, q2_pi)
    actor_loss = (ALPHA * log_prob - min_q).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update
    for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

for ep in range(MAX_EPISODES):
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _ = actor.sample(obs_tensor)
        action = action.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.store((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        episode_reward += reward

        update()
        if done:
            break

    print(f"Episode {ep+1} | Reward: {episode_reward:.2f}")

    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(actor.state_dict(), "sac_actor_best.pth")
        print(f"✅ Saved best actor with reward: {best_reward:.2f}")

env.close()
