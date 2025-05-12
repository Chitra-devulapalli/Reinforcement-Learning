import gymnasium as gym
import torch
import torch.nn as nn

# --- Setup ---
ENV_NAME = "CartPole-v1"
MODEL_PATH = "best_a2c_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Define Model ---
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

# --- Load Environment and Model ---
env = gym.make(ENV_NAME, render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = ActorCritic(obs_dim, act_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

episode_reward = 0
done = False
while not done:
    with torch.no_grad():
        logits, _ = model(obs)
        action = torch.argmax(logits, dim=-1).item()

    obs, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward
    done = terminated or truncated
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    if done:
        print(f"🎉 Final Episode Reward: {episode_reward:.2f}")
        break

env.close()
