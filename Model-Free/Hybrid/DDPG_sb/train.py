#!/usr/bin/env python3
"""
Minimal DDPG training script for Pendulum‑v1 (Gymnasium + SB3).

* identical hyper‑params to your interactive snippet
* no TensorBoard, no Monitor wrapper
* saves ``<save>.zip`` (default: ddpg_pendulum.zip)
"""

import argparse, gymnasium as gym, numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# ── CLI ---------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=10_000, help="total training timesteps")
p.add_argument("--save",  type=str, default="ddpg_pendulum", help="model basename")
p.add_argument("--seed",  type=int, default=123)
args = p.parse_args()

# ── Environment -------------------------------------------------------------
env = gym.make("Pendulum-v1")
env.reset(seed=args.seed)

# ── Gaussian exploration noise ---------------------------------------------
n_actions   = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1 * np.ones(n_actions))

# ── DDPG agent --------------------------------------------------------------
model = DDPG(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    seed=args.seed,
    # keep all other hyper‑parameters at SB3 defaults
)

print(f"▶ Training for {args.steps:,} steps …")
model.learn(total_timesteps=args.steps, log_interval=10)

model.save(args.save)
print(f"✅ Model saved to {args.save}.zip")

env.close()
