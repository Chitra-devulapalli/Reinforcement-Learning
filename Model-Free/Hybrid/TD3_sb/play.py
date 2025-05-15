import time
import gymnasium as gym
from stable_baselines3 import TD3

MODEL_FILE = "td3_pendulum.zip"
RENDER = True

env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None)
model = TD3.load(MODEL_FILE, env=env)

obs, _ = env.reset(seed=42)
done = False
episode_return = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    episode_return += reward
    if RENDER:
        env.render()
        time.sleep(1 / 60)

print(f"\n Episode return: {episode_return:.2f}")
env.close()
