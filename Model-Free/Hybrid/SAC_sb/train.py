import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, stable_baselines3.common.torch_layers as tl

ENV_ID = "Pendulum-v1"
TOTAL_STEPS = 10000
GAMMA = 0.99
TAU = 0.005
POLICY_ARCH = [256, 256]
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
SAVE_PATH = Path("sac_pendulum")
PLOT_PATH = Path.home() / "Documents/Reinforcement-Learning/media/sac_rewards2.png"

if torch.__version__.startswith("2.5"):
    _orig = tl.create_mlp
    def _create_mlp_strip_none(*a, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return _orig(*a, **kw)
    tl.create_mlp = _create_mlp_strip_none

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

env = Monitor(gym.make(ENV_ID))
reward_callback = RewardTracker()

model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
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

model.learn(total_timesteps=TOTAL_STEPS, log_interval=10, callback=reward_callback)
model.save(SAVE_PATH)
print(f"Trained SAC saved to {SAVE_PATH}.zip")
env.close()

PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(reward_callback.episode_rewards)
plt.title("SAC on Pendulum-v1")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()
print(f"Saved reward curve at {PLOT_PATH}")
