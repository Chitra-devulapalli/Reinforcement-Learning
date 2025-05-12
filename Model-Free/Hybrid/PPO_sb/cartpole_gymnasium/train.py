import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

ENV_NAME = "CartPole-v1"
MODEL_PATH = "ppo_cartpole_best"

# Create vectorized training environment
def make_env():
    return gym.make(ENV_NAME)

vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)

# Evaluation environment
eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=2048,       # rollout buffer size
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.0,
    learning_rate=3e-4,
)

# Evaluation callback to save best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
    render=False,
)

# Train the model
model.learn(total_timesteps=10_000, callback=eval_callback)
