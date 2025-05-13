#!/usr/bin/env python3
"""
Train an SAC agent on Pendulum‑v1 with Stable‑Baselines 3.
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

# --------------------------------------------------------------------------- #
#                               Hyper‑params                                  #
# --------------------------------------------------------------------------- #
ENV_ID       = "Pendulum-v1"
TOTAL_STEPS  = 10000              # ≈ 750 episodes × 200 steps
GAMMA        = 0.99
TAU          = 0.005
POLICY_ARCH  = [256, 256]           # default SAC net
LR_ACTOR     = 3e-4
LR_CRITIC    = 3e-4
SAVE_PATH    = Path("sac_pendulum")  # SB3 will append .zip

# --------------------------------------------------------------------------- #
#                         PyTorch ≥ 2.5 nightly fix                           #
#   (remove this block if you run torch ≤ 2.4, because it isn’t needed)       #
# --------------------------------------------------------------------------- #
import torch, stable_baselines3.common.torch_layers as tl          # noqa: E402

if torch.__version__.startswith("2.5"):
    _orig = tl.create_mlp
    def _create_mlp_strip_none(*a, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return _orig(*a, **kw)
    tl.create_mlp = _create_mlp_strip_none
# --------------------------------------------------------------------------- #

# ------------------- Environment -------------------------------------------
env = Monitor(gym.make(ENV_ID))

# ------------------- SAC model --------------------------------------------
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate = 3e-4,
    gamma=GAMMA,
    tau=TAU,
    buffer_size=1000000,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1,
    policy_kwargs=dict(net_arch=POLICY_ARCH),
    verbose=1,
    device="auto",
)

model.learn(total_timesteps=TOTAL_STEPS, log_interval=10)
model.save(SAVE_PATH)
print(f"✅  Trained SAC saved to {SAVE_PATH}.zip")
env.close()
