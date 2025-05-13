import time
import gymnasium as gym
from stable_baselines3 import TD3

MODEL_FILE = "td3_pendulum.zip"        # must exist in this directory
RENDER     = True                      # set False for head‑less run

# ── Environment & model -----------------------------------------------------
env   = gym.make("Pendulum-v1", render_mode="human" if RENDER else None)
model = TD3.load(MODEL_FILE, env=env)   # SB3 will wrap env as needed

# ── Single roll‑out ---------------------------------------------------------
obs, _ = env.reset(seed=42)
done   = False
episode_return = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    episode_return += reward
    if RENDER:
        env.render()
        time.sleep(1 / 60)             # ~60 FPS for a smooth window

print(f"\n✅  Episode return: {episode_return:.2f}")
env.close()