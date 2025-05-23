import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPDATE_STEPS = 2048
PPO_EPOCHS = 10
CLIP_EPS = 0.2
BATCH_SIZE = 64
MAX_TOTAL_STEPS = 100000
SAVE_PATH = "ppo_cartpole_best.pth"
REWARD_PLOT_PATH = Path.home() / "Documents/Reinforcement-Learning/media/ppo1_rewards.png"

env = gym.make(ENV_NAME, render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.shared(obs)
        return self.actor(x), self.critic(x)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

model = ActorCritic().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def store(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_advantages(self, last_value, gamma=GAMMA, lam=0.95):
        returns, advs = [], []
        gae = 0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[t])
        self.advantages = torch.tensor(advs, dtype=torch.float32).to(DEVICE)
        self.returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

    def to_tensor(self):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32).to(DEVICE),
            torch.tensor(self.actions).to(DEVICE),
            torch.tensor(self.log_probs).to(DEVICE)
        )

obs, _ = env.reset(seed=42)
buffer = RolloutBuffer()
episode_reward = 0
reward_history = []
best_avg_reward = -float("inf")
total_env_steps = 0

while total_env_steps < MAX_TOTAL_STEPS:
    steps_collected = 0
    while steps_collected < UPDATE_STEPS:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        action, log_prob, _, value = model.act(obs_tensor)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        buffer.store(obs, action.item(), log_prob.item(), reward, done, value.item())
        episode_reward += reward
        steps_collected += 1
        total_env_steps += 1
        obs = next_obs
        if done:
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-10:])
            print(f"[Steps {total_env_steps}] Episode Reward: {episode_reward:.2f} | Avg(10): {avg_reward:.2f}")
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"Saved best model with avg reward: {avg_reward:.2f}")
            episode_reward = 0
            obs, _ = env.reset()

    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        _, _, _, last_value = model.act(obs_tensor)
    buffer.compute_returns_advantages(last_value.item())
    obs_batch, act_batch, old_log_probs = buffer.to_tensor()
    returns = buffer.returns
    advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        for i in range(0, UPDATE_STEPS, BATCH_SIZE):
            idx = slice(i, i + BATCH_SIZE)
            logits, values = model(obs_batch[idx])
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(act_batch[idx])
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns[idx] - values.squeeze()).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

env.close()
REWARD_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(reward_history)
plt.title("PPO on CartPole-v1")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(REWARD_PLOT_PATH, dpi=300)
plt.close()
print(f"Saved reward curve at {REWARD_PLOT_PATH}")
