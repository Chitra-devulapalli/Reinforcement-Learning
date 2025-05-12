import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 5e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 10000
SAVE_PATH = "best_a2c_model.pth"

# --- Create Gym Environment ---
env = gym.make(ENV_NAME, render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# --- A2C Actor-Critic Network ---
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

model = ActorCritic(obs_dim, act_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Training Loop ---
obs, _ = env.reset(seed=42)
obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
episode_reward = 0
reward_history = []
best_avg_reward = -float("inf")

for step in range(MAX_STEPS):
    env.render()

    logits, value = model(obs)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    next_obs, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated
    episode_reward += reward

    reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
    _, next_value = model(next_obs_tensor)

    target = reward + GAMMA * (1 - float(done)) * next_value.squeeze()
    advantage = target - value.squeeze()

    actor_loss = -(log_prob * advantage.detach())
    critic_loss = advantage.pow(2)
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if done:
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-10:])
        print(f"[Step {step}] Episode Reward: {episode_reward:.2f} | Avg(10): {avg_reward:.2f}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best model with avg reward: {avg_reward:.2f}")

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        episode_reward = 0
    else:
        obs = next_obs_tensor

env.close()
