"""Tabular Q-learning training for the pursuer agent.

This adapts the simple Q-learning example from
https://github.com/asack20/RL-in-Pursuit-Evasion-Game to the continuous
pursuit--evasion environment by discretising the relative pursuer--evader
position.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import yaml
from gymnasium import Env
from torch.utils.tensorboard import SummaryWriter

from pursuit_evasion import PursuerOnlyEnv, load_config

# Discretisation parameters
N_BINS = 10
MAX_DIST = 5000.0  # metres
BINS = np.linspace(-MAX_DIST, MAX_DIST, N_BINS - 1)

# Action set: [acceleration magnitude, yaw, pitch]
ACTIONS = np.array([
    [0.0, 0.0, 0.0],  # do nothing
    [10.0, 0.0, 0.0],  # thrust forward
    [10.0, -0.2, 0.0],  # yaw left
    [10.0, 0.2, 0.0],  # yaw right
    [10.0, 0.0, 0.2],  # pitch up
    [10.0, 0.0, -0.2],  # pitch down
], dtype=np.float32)

@dataclass
class QConfig:
    episodes: int = 5000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    max_steps: int = 500
    log_dir: str | None = None


def discretise(obs: np.ndarray) -> int:
    """Map continuous observation to a discrete state index."""
    diff = obs[6:9] - obs[0:3]
    bins = np.digitize(diff, BINS)
    return int(bins[0] * N_BINS * N_BINS + bins[1] * N_BINS + bins[2])


def evaluate(env: Env, q_table: np.ndarray, episodes: int = 5) -> float:
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            s = discretise(obs)
            a = int(np.argmax(q_table[s]))
            obs, r, done, _, _ = env.step(ACTIONS[a])
            total += r
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards))


def train(cfg: QConfig, env_cfg: dict) -> np.ndarray:
    env = PursuerOnlyEnv(env_cfg, max_steps=cfg.max_steps)
    num_states = N_BINS ** 3
    q_table = np.zeros((num_states, len(ACTIONS)), dtype=np.float32)
    writer = SummaryWriter(log_dir=cfg.log_dir) if cfg.log_dir else None
    epsilon = cfg.epsilon
    for ep in range(cfg.episodes):
        obs, _ = env.reset()
        total = 0.0
        for step in range(cfg.max_steps):
            state = discretise(obs)
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(len(ACTIONS))
            else:
                action_idx = int(np.argmax(q_table[state]))
            next_obs, reward, done, _, _ = env.step(ACTIONS[action_idx])
            next_state = discretise(next_obs)
            q_old = q_table[state, action_idx]
            q_next = np.max(q_table[next_state])
            q_table[state, action_idx] = q_old + cfg.alpha * (
                reward + cfg.gamma * q_next - q_old
            )
            obs = next_obs
            total += reward
            if done:
                break
        epsilon = max(cfg.min_epsilon, epsilon * cfg.epsilon_decay)
        if writer:
            writer.add_scalar("train/episode_reward", total, ep)
            writer.add_scalar("train/epsilon", epsilon, ep)
        if (ep + 1) % 50 == 0:
            avg_r = evaluate(PursuerOnlyEnv(env_cfg, max_steps=cfg.max_steps), q_table)
            print(f"Episode {ep+1}: avg_reward={avg_r:.2f} eps={epsilon:.3f}")
            if writer:
                writer.add_scalar("eval/avg_reward", avg_r, ep)
    if writer:
        writer.close()
    return q_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pursuer with tabular Q-learning")
    parser.add_argument("--config", type=str, default="training.yaml", help="YAML config file")
    parser.add_argument("--save-path", type=str, default="pursuer_q.npy", help="output Q-table")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)
    q_cfg = QConfig(**full_cfg.get("q_learning", {}))
    env_cfg = load_config()
    q_table = train(q_cfg, env_cfg)
    np.save(args.save_path, q_table)
    print(f"Saved Q-table to {args.save_path}")


if __name__ == "__main__":
    main()
