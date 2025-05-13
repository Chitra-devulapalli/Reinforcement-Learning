#!/usr/bin/env python3
from tqdm import tqdm
import rich

# ── 0. Isaac Sim ─────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher
app_launcher   = AppLauncher(headless=True)      # flip to True once happy
simulation_app = app_launcher.app                 # keep alive

# ── 1. Std / third-party imports ─────────────────────────────────────────────
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

# ── 2. Small patch: env_is_wrapped() for VecMonitor ──────────────────────────
class PatchedSb3VecEnv(Sb3VecEnvWrapper):
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

# ── 3. Fixed output paths ────────────────────────────────────────────────────
ROOT = Path("/home/chitra/Documents/Reinforcement-Learning/Model-Free/Hybrid/PPO_sb/arm_isaaclab")
MODEL_DIR = ROOT / ""
LOG_DIR   = ROOT / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True,  exist_ok=True)

TASK_NAME = "Isaac-Lift-Cube-Franka-v0"

# ── 4. Hydra entry point ─────────────────────────────────────────────────────
@hydra_task_config(TASK_NAME, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.headless = True                        # GUI on
    # env_cfg.randomization = None
    # env_cfg.observations.policy.enable_corruption = False
    # env_cfg.scene.object.init_state.pos = (0.5, 0.00, 0.005)
    # env_cfg.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    # -------- Training environment ------------------------------------------------
    train_env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)
    if isinstance(train_env.unwrapped, DirectMARLEnv):
        train_env = multi_agent_to_single_agent(train_env)
    train_env = PatchedSb3VecEnv(train_env)
    train_env = VecMonitor(train_env, filename=str(LOG_DIR / "monitor.csv"))

    # -------- Evaluation environment ----------------------------------------------
    eval_env =train_env

    # -------- Agent ----------------------------------------------------------------
    # -------- Agent ----------------------------------------------------------------
    env_cfg.scene.num_envs = 128   # 128 envs
    env_cfg.observations.policy.enable_corruption = False

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        device="cuda",            # or "cpu"
        learning_rate=3e-4,       # PPO usually likes a smaller LR than A2C
        n_steps=512,             # rollout length before an update
        batch_size=2048,            # minibatch size for SGD
        n_epochs=10,              # gradient passes over each batch
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,           # policy clip
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        policy_kwargs={"net_arch": [64, 64]},
    )


    model.set_logger(configure(str(LOG_DIR), ["stdout", "tensorboard"]))

    # -------- Callbacks ------------------------------------------------------------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=20000,          # was 1000
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=100000,         # was 5000
        save_path=str(MODEL_DIR),
        name_prefix="checkpoint",
        verbose=2,
    )


    # -------- Train ----------------------------------------------------------------
    model.learn(total_timesteps=128 * 512 * 80, callback=[eval_cb, ckpt_cb], progress_bar = True)

    # always save final weights
    model.save(MODEL_DIR / "best_model")

    # -------- Clean-up -------------------------------------------------------------
    train_env.close()
    eval_env.close()
    simulation_app.close()
    print("\n✅  Training finished. Outputs are in:", ROOT)

# ── 5. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
