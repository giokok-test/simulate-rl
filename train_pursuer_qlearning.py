"""Deep Q-learning trainer for the pursuer agent.

The continuous observation is processed by a small neural network that
approximates the action-value function :math:`Q(s, a)`.  A fixed set of
discrete actions controls thrust and orientation, making the method a direct
application of ``DQN`` [Mnih et al., 2015].  Experience replay and a target
network enable batched updates while TensorBoard records training metrics.

Example
-------
```bash
python train_pursuer_qlearning.py --episodes 10 --log-dir runs/dqn
```

References
----------
* Mnih et al., 2015. *Human-level control through deep reinforcement learning*
* Watkins & Dayan, 1992. *Q-learning*
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from gymnasium import Env
from torch.utils.tensorboard import SummaryWriter

from pursuit_evasion import load_config
from curriculum import Curriculum, initialize_gym


# Action set: [acceleration magnitude, yaw, pitch]
ACTIONS = np.array(
    [
        [0.0, 0.0, 0.0],  # do nothing
        [10.0, 0.0, 0.0],  # thrust forward
        [0.0, -0.2, 0.0],  # yaw left
        [0.0, 0.2, 0.0],  # yaw right
        [0.0, 0.0, 0.2],  # pitch up
        [0.0, 0.0, -0.2],  # pitch down
    ],
    dtype=np.float32,
)


@dataclass
class QConfig:
    """Hyper-parameters for Q-learning."""

    episodes: int = 5_000
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update: int = 200
    eval_freq: int = 50
    max_steps: int = 500
    log_dir: str | None = None
    capture_bonus: float = 0.0
    checkpoint_every: int | None = None


class ReplayBuffer:
    """Cyclic experience replay buffer supporting random mini-batches."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(
        self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.done[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.choice(max_idx, batch_size, replace=False)
        return (
            self.obs[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_obs[idxs],
            self.done[idxs],
        )


class QNetwork(nn.Module):
    """Simple MLP producing Q-values for all actions."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def _extract_parameter_ranges(cfg: dict, prefix: str = "") -> dict[str, tuple[float, float]]:
    """Return numeric ranges (min, max) discovered in ``cfg``.

    The helper looks for list/tuple values of length two as well as matching
    ``min_*``/``max_*`` pairs. Keys are concatenated with ``.`` to form a
    hierarchical identifier.
    """

    ranges: dict[str, tuple[float, float]] = {}
    min_candidates: dict[str, float] = {}
    max_candidates: dict[str, float] = {}

    for key, value in cfg.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            ranges.update(_extract_parameter_ranges(value, new_prefix))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            if all(isinstance(v, (int, float)) for v in value):
                low, high = float(value[0]), float(value[1])
                if low > high:
                    low, high = high, low
                ranges[new_prefix] = (low, high)
        elif isinstance(value, (int, float)):
            if key.startswith("min_"):
                min_candidates[key[4:]] = float(value)
            elif key.startswith("max_"):
                max_candidates[key[4:]] = float(value)

    for base, min_val in min_candidates.items():
        if base in max_candidates:
            max_val = max_candidates[base]
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            range_name = f"{prefix}.{base}" if prefix else base
            ranges[range_name] = (low, high)

    return ranges


def _log_parameter_ranges(
    writer: SummaryWriter, cfg: dict, prefix: str, step: int
) -> None:
    """Write environment parameter ranges to TensorBoard."""

    for name, (low, high) in _extract_parameter_ranges(cfg).items():
        tag = name.replace(".", "/")
        writer.add_scalar(f"{prefix}/param_range/{tag}/min", low, step)
        writer.add_scalar(f"{prefix}/param_range/{tag}/max", high, step)
        writer.add_scalar(f"{prefix}/param_range/{tag}/width", high - low, step)


def _log_training_batch_stats(
    batch_index: int,
    episodes: int,
    outcomes: Counter[str],
    avg_reward: float,
    avg_duration: float,
) -> None:
    """Print aggregate statistics for a completed batch of episodes."""

    if episodes == 0:
        return
    outcome_summary = " ".join(
        f"{name}:{count}" for name, count in sorted(outcomes.items())
    )
    logging.info(
        "Train batch %d (%d episodes): outcomes=%s avg_reward=%.2f avg_duration=%.1f",
        batch_index,
        episodes,
        outcome_summary or "none",
        avg_reward,
        avg_duration,
    )


def evaluate(
    env: Env, policy: QNetwork, device: torch.device, episodes: int = 5
) -> tuple[float, dict[str, float]]:
    """Greedy policy evaluation returning mean reward and auxiliary metrics."""

    rewards = []
    min_start_ratios: list[float] = []
    min_distances: list[float] = []
    episode_steps: list[int] = []
    captures = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        info: dict | None = None
        for _ in range(env.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy(obs_t)
            action = ACTIONS[int(torch.argmax(q_values, dim=1).item())]
            obs, reward, done, _, info = env.step(action)
            total += reward
            if done:
                break
        rewards.append(total)
        if info:
            min_d = info.get("min_distance")
            start_d = info.get("start_distance", 1.0)
            if min_d is not None:
                min_distances.append(float(min_d))
                min_start_ratios.append(float(min_d) / max(start_d, 1e-8))
            episode_steps.append(info.get("episode_steps", env.max_steps))
            if info.get("outcome") == "capture":
                captures += 1
    metrics = {
        "avg_min_distance": float(np.mean(min_distances)) if min_distances else 0.0,
        "avg_min_start_ratio": float(np.mean(min_start_ratios)) if min_start_ratios else 0.0,
        "avg_episode_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
        "capture_rate": captures / episodes if episodes > 0 else 0.0,
    }
    return float(np.mean(rewards)), metrics


def compute_loss(
    batch: Tuple[np.ndarray, ...],
    policy: QNetwork,
    target: QNetwork,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the mean squared Bellman error for ``batch``."""

    obs, action, reward, next_obs, done = batch
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
    action_t = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(1)
    reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
    done_t = torch.tensor(done, dtype=torch.float32, device=device)

    q = policy(obs_t).gather(1, action_t).squeeze(1)
    with torch.no_grad():
        max_next_q = target(next_obs_t).max(1).values
        target_q = reward_t + gamma * (1.0 - done_t) * max_next_q
    return nn.functional.mse_loss(q, target_q)


def train(
    cfg: QConfig,
    env_cfg: dict,
    curriculum: Curriculum | None = None,
    *,
    save_path: str | None = None,
    resume_from: str | None = None,
) -> QNetwork:
    """Run training loop and return the trained Q-network."""

    logging.info("Training for %d episodes", cfg.episodes)
    # Create a temporary env to determine observation dimensions.
    tmp_env = initialize_gym(env_cfg, curriculum=curriculum, max_steps=cfg.max_steps)
    obs_dim = tmp_env.observation_space.shape[0]
    n_actions = len(ACTIONS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = QNetwork(obs_dim, n_actions).to(device)
    target = QNetwork(obs_dim, n_actions).to(device)
    if resume_from:
        state = torch.load(resume_from, map_location=device)
        policy.load_state_dict(state)
        target.load_state_dict(state)
        logging.info("Loaded checkpoint from %s", resume_from)
    target.load_state_dict(policy.state_dict())
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim)
    writer = SummaryWriter(log_dir=cfg.log_dir) if cfg.log_dir else None

    epsilon = cfg.epsilon_start
    global_step = 0
    success_window: Deque[int] = deque(maxlen=cfg.batch_size)
    batch_window = max(cfg.batch_size, 1)
    batch_outcomes: Counter[str] = Counter()
    batch_reward_sum = 0.0
    batch_duration_sum = 0.0
    batch_episode_count = 0
    batch_index = 0
    for ep in range(cfg.episodes):
        if curriculum is not None:
            curriculum.advance(ep, cfg.episodes)
        env = initialize_gym(
            env_cfg,
            curriculum=curriculum,
            max_steps=cfg.max_steps,
            capture_bonus=cfg.capture_bonus,
        )
        obs, _ = env.reset()
        total_reward = 0.0
        q_sum = 0.0
        q_count = 0
        info = {}
        for _ in range(cfg.max_steps):
            global_step += 1
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy(obs_t)
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(n_actions)
            else:
                action_idx = int(torch.argmax(q_values, dim=1).item())

            q_sum += float(q_values.max().item())
            q_count += 1

            next_obs, reward, done, _, info = env.step(ACTIONS[action_idx])
            buffer.add(obs, action_idx, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

            if buffer.full or buffer.idx >= cfg.batch_size:
                batch = buffer.sample(cfg.batch_size)
                loss = compute_loss(batch, policy, target, cfg.gamma, device)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)

            if global_step % cfg.target_update == 0:
                target.load_state_dict(policy.state_dict())

            if done:
                break

        success = bool(info.get("outcome") == "capture") if info else False
        if curriculum is not None:
            curriculum.update(success)
        success_window.append(1 if success else 0)
        batch_success_rate = (
            sum(success_window) / len(success_window) if success_window else 0.0
        )

        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
        if writer:
            writer.add_scalar("train/episode_reward", total_reward, ep)
            writer.add_scalar("train/epsilon", epsilon, ep)
            writer.add_scalar("train/avg_q", q_sum / max(q_count, 1), ep)
            if curriculum is not None:
                writer.add_scalar("train/curriculum_progress", curriculum.progress, ep)
            replay_size = buffer.capacity if buffer.full else buffer.idx
            writer.add_scalar("train/replay_size", replay_size, ep)
            writer.add_scalar("train/success_rate_batch", batch_success_rate, ep)
            if info:
                start_d = info.get("start_distance", 0.0)
                min_d = info.get("min_distance")
                if min_d is not None and start_d > 0:
                    writer.add_scalar("train/min_distance", min_d, ep)
                    writer.add_scalar("train/start_distance", start_d, ep)
                    writer.add_scalar("train/min_start_ratio", min_d / start_d, ep)
                writer.add_scalar("train/episode_steps", info.get("episode_steps", 0), ep)
                writer.add_scalar("train/final_distance", info.get("final_distance", 0.0), ep)
                writer.add_scalar("train/evader_to_target", info.get("evader_to_target", 0.0), ep)
                writer.add_scalar("train/pursuer_acc_delta", info.get("pursuer_acc_delta", 0.0), ep)
                writer.add_scalar("train/pursuer_yaw_delta", info.get("pursuer_yaw_delta", 0.0), ep)
                writer.add_scalar("train/pursuer_pitch_delta", info.get("pursuer_pitch_delta", 0.0), ep)
                writer.add_scalar("train/pursuer_vel_delta", info.get("pursuer_vel_delta", 0.0), ep)
                writer.add_scalar("train/pursuer_yaw_diff", info.get("pursuer_yaw_diff", 0.0), ep)
                writer.add_scalar("train/pursuer_pitch_diff", info.get("pursuer_pitch_diff", 0.0), ep)
                writer.add_scalar("train/evader_yaw_diff", info.get("evader_yaw_diff", 0.0), ep)
                writer.add_scalar("train/evader_pitch_diff", info.get("evader_pitch_diff", 0.0), ep)
                writer.add_scalar(
                    "train/capture", 1.0 if info.get("outcome") == "capture" else 0.0, ep
                )
                writer.add_scalar("train/timing_bonus", info.get("timing_bonus", 0.0), ep)
                r_bd = info.get("reward_breakdown")
                if r_bd:
                    for key, val in r_bd.items():
                        writer.add_scalar(f"train/reward_{key}", val, ep)
            if hasattr(env, "env") and hasattr(env.env, "cfg"):
                _log_parameter_ranges(writer, env.env.cfg, "train", ep)

        if (ep + 1) % cfg.eval_freq == 0:
            eval_env = initialize_gym(
                env_cfg,
                curriculum=curriculum,
                max_steps=cfg.max_steps,
                capture_bonus=cfg.capture_bonus,
            )
            avg_r, eval_metrics = evaluate(eval_env, policy, device)
            logging.info(
                "Episode %d: avg_reward=%.2f epsilon=%.3f curriculum=%.2f",
                ep + 1,
                avg_r,
                epsilon,
                curriculum.progress if curriculum is not None else 0.0,
            )
            if writer:
                writer.add_scalar("eval/avg_reward", avg_r, ep)
                writer.add_scalar(
                    "eval/min_start_ratio", eval_metrics["avg_min_start_ratio"], ep
                )
                writer.add_scalar(
                    "eval/min_distance", eval_metrics["avg_min_distance"], ep
                )
                writer.add_scalar(
                    "eval/episode_steps", eval_metrics["avg_episode_steps"], ep
                )
                writer.add_scalar(
                    "eval/capture_rate", eval_metrics["capture_rate"], ep
                )
                if hasattr(eval_env, "env") and hasattr(eval_env.env, "cfg"):
                    _log_parameter_ranges(writer, eval_env.env.cfg, "eval", ep)

        outcome_name = "unknown"
        episode_steps = float(getattr(env, "max_steps", cfg.max_steps))
        if info:
            outcome_val = info.get("outcome")
            if isinstance(outcome_val, str) and outcome_val:
                outcome_name = outcome_val
            episode_steps = float(info.get("episode_steps", episode_steps))
        batch_outcomes[outcome_name] += 1
        batch_reward_sum += total_reward
        batch_duration_sum += episode_steps
        batch_episode_count += 1
        if batch_episode_count >= batch_window:
            batch_index += 1
            _log_training_batch_stats(
                batch_index,
                batch_episode_count,
                batch_outcomes,
                batch_reward_sum / batch_episode_count,
                batch_duration_sum / batch_episode_count,
            )
            batch_outcomes.clear()
            batch_reward_sum = 0.0
            batch_duration_sum = 0.0
            batch_episode_count = 0

        if (
            cfg.checkpoint_every
            and save_path
            and (ep + 1) % cfg.checkpoint_every == 0
        ):
            base, ext = os.path.splitext(os.path.basename(save_path))
            ckpt_file = f"{base}_ckpt_{ep+1}{ext}"
            if cfg.log_dir:
                ckpt_dir = os.path.join(cfg.log_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            else:
                ckpt_path = os.path.join(os.path.dirname(save_path), ckpt_file)
            torch.save(policy.state_dict(), ckpt_path)
            logging.info("Saved checkpoint to %s", ckpt_path)

    if writer:
        writer.close()
    if batch_episode_count:
        batch_index += 1
        _log_training_batch_stats(
            batch_index,
            batch_episode_count,
            batch_outcomes,
            batch_reward_sum / batch_episode_count,
            batch_duration_sum / batch_episode_count,
        )
    return policy


def main() -> None:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description="Train pursuer with deep Q-learning")
    parser.add_argument(
        "--config", type=str, default="setup/training.yaml", help="YAML config file"
    )
    parser.add_argument(
        "--save-path", type=str, default="pursuer_dqn.pt", help="output weight file"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="save a checkpoint every N episodes",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="load model weights from this checkpoint before training",
    )
    parser.add_argument("--episodes", type=int, default=None, help="override episode count")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    q_cfg = QConfig(**full_cfg.get("q_learning", {}))
    if args.episodes is not None:
        q_cfg.episodes = args.episodes
    if args.log_dir is not None:
        q_cfg.log_dir = args.log_dir
    if args.checkpoint_every is not None:
        q_cfg.checkpoint_every = args.checkpoint_every

    cur_cfg = full_cfg.get("curriculum") or {}
    curriculum = None
    mode = cur_cfg.get("mode")
    if mode:
        curriculum = Curriculum(
            start=cur_cfg.get("start", {}),
            end=cur_cfg.get("end", {}),
            mode=mode,
            stages=cur_cfg.get("stages", 2),
            success_threshold=cur_cfg.get("success_threshold", 0.6),
            window=cur_cfg.get("window", 64),
        )

    env_cfg = load_config()
    model = train(
        q_cfg,
        env_cfg,
        curriculum,
        save_path=args.save_path,
        resume_from=args.resume_from,
    )
    torch.save(model.state_dict(), args.save_path)
    logging.info("Saved model to %s", args.save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()

