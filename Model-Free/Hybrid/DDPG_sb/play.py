import time
import gymnasium as gym
from stable_baselines3 import DDPG

# --------------------------------------------------------------------------- #
#                         Hard‑wired settings                                 #
# --------------------------------------------------------------------------- #
MODEL_FILE = "ddpg_pendulum.zip"   # must exist in the current directory
RENDER     = True                  # set False for headless evaluation

# --------------------------------------------------------------------------- #
#                         Environment & model                                 #
# --------------------------------------------------------------------------- #
env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None)
model = DDPG.load(MODEL_FILE, env=env)    # SB3 handles any needed wrappers

# --------------------------------------------------------------------------- #
#                         Single roll‑out                                     #
# --------------------------------------------------------------------------- #
obs, _   = env.reset(seed=42)
done     = False
ep_ret   = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done      = terminated or truncated
    ep_ret   += reward
    if RENDER:
        env.render()
        time.sleep(1 / 60)        # ~60 FPS to keep it smooth

print(f"\n✅  Episode return: {ep_ret:.2f}")
env.close()