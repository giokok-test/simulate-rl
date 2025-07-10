from __future__ import annotations


import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Optional

from pursuit_evasion import PursuitEvasionEnv, PursuerPolicy, load_config

# Load configuration and set the evader to be unaware
config = load_config()
config['evader']['awareness_mode'] = 1


def _format_step(env: PursuerOnlyEnv, step: int, target: np.ndarray) -> str:
    """Return formatted row for the table shown in ``play.py``."""

    pe_vec = env.env.evader_pos - env.env.pursuer_pos
    et_vec = target - env.env.evader_pos
    pv = env.env.pursuer_vel
    ev = env.env.evader_vel
    pv_u = pv / (np.linalg.norm(pv) + 1e-8)
    ev_u = ev / (np.linalg.norm(ev) + 1e-8)
    pe_u = pe_vec / (np.linalg.norm(pe_vec) + 1e-8)
    return (
        f"{step:5d} | "
        f"[{pe_vec[0]:7.1f} {pe_vec[1]:7.1f} {pe_vec[2]:7.1f}] | "
        f"[{et_vec[0]:7.1f} {et_vec[1]:7.1f} {et_vec[2]:7.1f}] | "
        f"[{pv[0]:7.1f} {pv[1]:7.1f} {pv[2]:7.1f}] | "
        f"[{ev[0]:7.1f} {ev[1]:7.1f} {ev[2]:7.1f}] | "
        f"[{pv_u[0]:6.2f} {pv_u[1]:6.2f} {pv_u[2]:6.2f}] | "
        f"[{ev_u[0]:6.2f} {ev_u[1]:6.2f} {ev_u[2]:6.2f}] | "
        f"[{pe_u[0]:6.2f} {pe_u[1]:6.2f} {pe_u[2]:6.2f}]"
    )


def evader_policy(env: PursuitEvasionEnv) -> np.ndarray:
    """Evader accelerates toward the target with optional dive profile."""
    pos = env.evader_pos
    target = np.array(env.cfg['target_position'], dtype=np.float32)
    direction = target - pos
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction /= norm
    theta = np.arctan2(direction[1], direction[0])
    phi_target = np.arctan2(direction[2], np.linalg.norm(direction[:2]))
    mode = env.cfg['evader'].get('trajectory', 'direct')
    if mode == 'dive':
        threshold = env.cfg['evader'].get('dive_angle', 0.0)
        if phi_target > -threshold:
            phi = max(0.0, phi_target)
        else:
            phi = phi_target
    else:
        phi = phi_target
    phi = np.clip(phi, -env.cfg['evader']['stall_angle'], env.cfg['evader']['stall_angle'])
    mag = env.cfg['evader']['max_acceleration']
    return np.array([mag, theta, phi], dtype=np.float32)


class PursuerOnlyEnv(gym.Env):
    """Environment exposing only the pursuer. The evader follows ``evader_policy``."""

    def __init__(self, cfg: dict, max_steps: int | None = None):
        super().__init__()
        # Full pursuit-evasion environment internally used
        self.env = PursuitEvasionEnv(cfg)
        self.observation_space = self.env.observation_space['pursuer']
        self.action_space = self.env.action_space['pursuer']
        if max_steps is None:
            duration = cfg.get('episode_duration', 0.1)
            self.max_steps = int(duration * 60.0 / cfg['time_step'])
        else:
            self.max_steps = max_steps
        self.cur_step = 0

    def reset(self, *, seed=None, options=None):
        """Reset the wrapped environment and return the pursuer observation."""

        obs, info = self.env.reset(seed=seed)
        self.cur_step = 0
        self.start_distance = self.env.start_pe_dist
        return obs['pursuer'].astype(np.float32), info

    def step(self, action: np.ndarray):
        """Take a step using the pursuer action while the evader follows its fixed policy."""

        e_action = evader_policy(self.env)
        obs, reward, done, truncated, info = self.env.step({'pursuer': action, 'evader': e_action})
        info.setdefault('start_distance', float(self.env.start_pe_dist))
        self.cur_step += 1
        if self.cur_step >= self.max_steps and not done:
            done = True
            info.setdefault('episode_steps', self.cur_step)
            info.setdefault('min_distance', float(self.env.min_pe_dist))
            info.setdefault('final_distance', float(np.linalg.norm(self.env.evader_pos - self.env.pursuer_pos)))
            target = np.array(self.env.cfg['target_position'], dtype=np.float32)
            dist_target = np.linalg.norm(self.env.evader_pos - target)
            info.setdefault('evader_to_target', float(dist_target))
            info.setdefault('start_distance', float(self.env.start_pe_dist))
            info['outcome'] = 'timeout'
        return obs['pursuer'].astype(np.float32), float(reward['pursuer']), done, truncated, info


def evaluate(policy: PursuerPolicy, env: PursuerOnlyEnv, episodes: int = 5) -> tuple[float, float]:
    """Run several evaluation episodes.

    Returns
    -------
    tuple
        Mean reward and success rate over ``episodes`` runs.
    """

    rewards = []
    successes = 0
    min_dists = []
    steps = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        info = {}
        while not done:
            with torch.no_grad():
                action = policy(torch.tensor(obs, device=next(policy.parameters()).device))
            obs, r, done, _, info = env.step(action.cpu().numpy())
            total += r
        rewards.append(total)
        if total > 0:
            successes += 1
        if info:
            min_dists.append(info.get('min_distance', np.nan))
            steps.append(info.get('episode_steps', np.nan))

    if min_dists:
        print(
            f"    eval metrics: mean_min_dist={np.nanmean(min_dists):.2f} "
            f"mean_steps={np.nanmean(steps):.1f}"
        )
    return float(np.mean(rewards)), successes / episodes


def train(
    cfg: dict,
    save_path: Optional[str] = None,
    *,
    checkpoint_every: int | None = None,
    resume_from: str | None = None,
):
    """Train the pursuer policy with REINFORCE.

    Parameters
    ----------
    cfg:
        Configuration dictionary. Expected to contain a ``training`` section
        specifying ``episodes``, ``learning_rate`` and ``eval_freq``.
    save_path:
        File where the final policy weights will be written.
    checkpoint_every:
        Save intermediate checkpoints every this many episodes when not ``None``.
    resume_from:
        Optional path to a checkpoint file to start from.
    """

    training_cfg = cfg.get('training', {})
    num_episodes = training_cfg.get('episodes', 100)
    learning_rate = training_cfg.get('learning_rate', 1e-3)
    eval_freq = training_cfg.get('eval_freq', 10)
    if checkpoint_every is None:
        checkpoint_every = training_cfg.get('checkpoint_steps')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PursuerOnlyEnv(cfg)
    policy = PursuerPolicy(env.observation_space.shape[0]).to(device)
    if resume_from:
        state_dict = torch.load(resume_from, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {resume_from}")
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    gamma = 0.99

    header = (
        f"{'step':>5} | {'pursuer→evader [m]':>26} | "
        f"{'evader→target [m]':>26} | {'pursuer vel [m/s]':>26} | "
        f"{'evader vel [m/s]':>26} | {'p dir':>18} | {'e dir':>18} | "
        f"{'p→e dir':>18}"
    )

    for episode in range(num_episodes):
        # Collect one episode of experience
        obs, _ = env.reset()
        init_pursuer_pos = env.env.pursuer_pos.copy()
        init_evader_pos = env.env.evader_pos.copy()
        log_probs = []
        rewards = []
        done = False
        info = {}
        start_d = env.start_distance
        target = np.asarray(env.env.cfg["target_position"], dtype=float)
        first_rows: list[str] = []
        last_rows: list[str] = []
        step = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mean = policy(obs_t)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean))
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            obs, r, done, _, info = env.step(action.cpu().numpy())
            log_probs.append(log_prob)
            rewards.append(r)
            row = _format_step(env, step, target)
            if len(first_rows) < 3:
                first_rows.append(row)
            if len(last_rows) >= 3:
                last_rows.pop(0)
            last_rows.append(row)
            step += 1
        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -torch.sum(torch.stack(log_probs) * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Initial pursuer pos: {init_pursuer_pos}")
        print(f"Initial evader pos: {init_evader_pos}")
        print(header)
        print("-" * len(header))
        for row in first_rows:
            print(row)
        for row in last_rows:
            print(row)
        episode_reward = sum(rewards)
        if info:
            print(
                f"Episode {episode+1}: reward={episode_reward:.2f} "
                f"outcome={info.get('outcome', 'timeout')} start={start_d:.2f} "
                f"min={info.get('min_distance', float('nan')):.2f}"
            )
        if (episode + 1) % eval_freq == 0:
            # Periodically report progress on separate evaluation episodes
            avg_r, success = evaluate(policy, PursuerOnlyEnv(config))
            print(f"Episode {episode+1}: avg_reward={avg_r:.2f} success={success:.2f}")
        if checkpoint_every and save_path and (episode + 1) % checkpoint_every == 0:
            base, ext = os.path.splitext(save_path)
            ckpt_path = f"{base}_ckpt_{episode+1}{ext}"
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Final evaluation after training
    avg_r, success = evaluate(policy, PursuerOnlyEnv(config))
    print(f"Final performance: avg_reward={avg_r:.2f} success={success:.2f}")

    if save_path is not None:
        torch.save(policy.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pursuer policy")
    parser.add_argument("--episodes", type=int,
                        help="number of training episodes")
    parser.add_argument("--lr", type=float,
                        help="optimizer learning rate")
    parser.add_argument("--eval-freq", type=int,
                        help="how often to run evaluation episodes")
    parser.add_argument("--time-step", type=float,
                        help="simulation time step override")
    parser.add_argument("--save-path", type=str,
                        default="pursuer_policy.pt",
                        help="where to store the trained weights")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="save a checkpoint every N episodes",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="start training from this checkpoint file",
    )
    args = parser.parse_args()

    training_cfg = config.setdefault('training', {
        'episodes': 5000,
        'learning_rate': 1e-3,
        'eval_freq': 1000,
        'checkpoint_steps': 0,
    })
    if args.episodes is not None:
        training_cfg['episodes'] = args.episodes
    if args.lr is not None:
        training_cfg['learning_rate'] = args.lr
    if args.eval_freq is not None:
        training_cfg['eval_freq'] = args.eval_freq
    if args.checkpoint_every is not None:
        training_cfg['checkpoint_steps'] = args.checkpoint_every
    if args.time_step is not None:
        config['time_step'] = args.time_step

    train(
        config,
        save_path=args.save_path,
        checkpoint_every=training_cfg.get('checkpoint_steps'),
        resume_from=args.resume_from,
    )
