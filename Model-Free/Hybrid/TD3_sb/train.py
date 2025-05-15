import argparse, gymnasium as gym, numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

# CLI input
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10000, help="total training timesteps")
parser.add_argument("--save",  type=str, default="td3_pendulum", help="model basename")
parser.add_argument("--seed",  type=int, default=123)
args = parser.parse_args()

env = gym.make("Pendulum-v1")
env.reset(seed=args.seed)

# Gaussian exploration noise (TD3 default choice)
n_actions   = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1 * np.ones(n_actions))

# TD3 agent
model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    seed=args.seed,
    # keep all other hyper‑parameters at SB3 defaults
)

print(f"▶ Training for {args.steps:,} steps")
model.learn(total_timesteps=args.steps, log_interval=10)

model.save(args.save)
print(f"Model saved to {args.save}.zip")

env.close()
