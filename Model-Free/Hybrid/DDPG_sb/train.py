import argparse, gymnasium as gym, numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_reward = 0

    def _on_step(self):
        reward = self.locals.get("rewards", [0])[0]
        self.episode_reward += reward
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        return True

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=10000, help="total training timesteps")
p.add_argument("--save",  type=str, default="ddpg_pendulum", help="model basename")
p.add_argument("--seed",  type=int, default=123)
args = p.parse_args()

env = gym.make("Pendulum-v1")
env.reset(seed=args.seed)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1 * np.ones(n_actions))

callback = RewardTracker()

model = DDPG(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    seed=args.seed,
)

print(f"Training for {args.steps:,} steps")
model.learn(total_timesteps=args.steps, log_interval=10, callback=callback)

model.save(args.save)
print(f"Model saved to {args.save}.zip")

env.close()

plot_path = Path.home() / "Documents/Reinforcement-Learning/media/rewards2_ddpg.png"
plot_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(callback.episode_rewards)
plt.title("DDPG on Pendulum-v1")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved reward curve at {plot_path}")
