import argparse, gymnasium as gym, numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10000, help="total training timesteps")
parser.add_argument("--save",  type=str, default="td3_pendulum", help="model basename")
parser.add_argument("--seed",  type=int, default=123)
args = parser.parse_args()

PLOT_PATH = Path.home() / "Documents/Reinforcement-Learning/media/td3_rewards2.png"

class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

env = gym.make("Pendulum-v1")
env.reset(seed=args.seed)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1 * np.ones(n_actions))

reward_callback = RewardTracker()

model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    seed=args.seed,
)

print(f"Training for {args.steps:,} steps")
model.learn(total_timesteps=args.steps, log_interval=10, callback=reward_callback)

model.save(args.save)
print(f"Model saved to {args.save}.zip")

env.close()
PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(reward_callback.episode_rewards)
plt.title("TD3 on Pendulum-v1")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()
print(f"Saved reward curve at {PLOT_PATH}")
