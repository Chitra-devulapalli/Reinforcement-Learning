import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

ENV_NAME = "CartPole-v1"
MODEL_PATH = "best_model"
LOG_PATH = "./logs/"

# Create vectorized training and eval environments
def make_env():
    return gym.make(ENV_NAME)

vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)

eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

# Define and train model
model = A2C("MlpPolicy", vec_env, verbose=1)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./",
    log_path=LOG_PATH,
    eval_freq=1000,
    deterministic=True,
    render=False,
)

# Train the model
model.learn(total_timesteps=25000, callback=eval_callback)

# Plot evaluation rewards
npz_path = os.path.join(LOG_PATH, "evaluations.npz")
if os.path.exists(npz_path):
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: (evals, 1)
    mean_rewards = results.mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_rewards, label="Mean Eval Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("A2C Evaluation Rewards on CartPole")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.show()
else:
    print(f"Could not find log file at {npz_path}")
