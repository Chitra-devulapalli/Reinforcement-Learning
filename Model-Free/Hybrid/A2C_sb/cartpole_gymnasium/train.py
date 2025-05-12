import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

ENV_NAME = "CartPole-v1"
MODEL_PATH = "a2c_cartpole_best"

# Create vectorized environment
def make_env():
    return gym.make(ENV_NAME)

vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)

# Evaluation environment
eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

# Define model
model = A2C("MlpPolicy", vec_env, verbose=1)

# Callback to save the best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
    render=False,
)

# Train model
model.learn(total_timesteps=25000, callback=eval_callback)
