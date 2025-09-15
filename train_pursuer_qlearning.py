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
from dataclasses import dataclass
from typing import Tuple

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
        [10.0, -0.2, 0.0],  # yaw left
        [10.0, 0.2, 0.0],  # yaw right
        [10.0, 0.0, 0.2],  # pitch up
        [10.0, 0.0, -0.2],  # pitch down
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


def evaluate(env: Env, policy: QNetwork, device: torch.device, episodes: int = 5) -> float:
    """Greedy policy evaluation returning mean reward over ``episodes``."""

    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy(obs_t)
            action = ACTIONS[int(torch.argmax(q_values, dim=1).item())]
            obs, reward, done, _, _ = env.step(action)
            total += reward
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards))


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

        if curriculum is not None:
            curriculum.update(info.get("outcome") == "capture")

        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
        if writer:
            writer.add_scalar("train/episode_reward", total_reward, ep)
            writer.add_scalar("train/epsilon", epsilon, ep)
            writer.add_scalar("train/avg_q", q_sum / max(q_count, 1), ep)
            if curriculum is not None:
                writer.add_scalar("train/curriculum_progress", curriculum.progress, ep)
            replay_size = buffer.capacity if buffer.full else buffer.idx
            writer.add_scalar("train/replay_size", replay_size, ep)

        if (ep + 1) % cfg.eval_freq == 0:
            eval_env = initialize_gym(
                env_cfg,
                curriculum=curriculum,
                max_steps=cfg.max_steps,
                capture_bonus=cfg.capture_bonus,
            )
            avg_r = evaluate(eval_env, policy, device)
            logging.info(
                "Episode %d: avg_reward=%.2f epsilon=%.3f curriculum=%.2f",
                ep + 1,
                avg_r,
                epsilon,
                curriculum.progress if curriculum is not None else 0.0,
            )
            if writer:
                writer.add_scalar("eval/avg_reward", avg_r, ep)

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

