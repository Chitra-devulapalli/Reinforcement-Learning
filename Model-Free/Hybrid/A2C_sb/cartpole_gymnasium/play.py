import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_NAME = "CartPole-v1"
MODEL_PATH = "best_model"

# Create environment
env = DummyVecEnv([lambda: gym.make(ENV_NAME, render_mode="human")])
model = A2C.load(MODEL_PATH)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done[0]:
        obs = env.reset()
