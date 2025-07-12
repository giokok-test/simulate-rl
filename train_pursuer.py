from __future__ import annotations


import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from typing import Optional
from collections import defaultdict
import yaml

TABLE_HEADER = (
    f"{'step':>5} | {'pursuer→evader [m]':>26} | "
    f"{'evader→target [m]':>26} | {'pursuer vel [m/s]':>26} | "
    f"{'evader vel [m/s]':>26} | {'p dir':>18} | "
    f"{'e dir':>18} | {'p→e dir':>18}"
)


def _format_row(step: int, env: PursuitEvasionEnv) -> str:
    """Return a formatted table row showing current vectors and velocities."""
    target_pos = np.asarray(env.cfg["target_position"], dtype=float)
    pe_vec = env.evader_pos - env.pursuer_pos
    et_vec = target_pos - env.evader_pos
    pv = env.pursuer_vel
    ev = env.evader_vel
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

from pursuit_evasion import (
    PursuitEvasionEnv,
    PursuerPolicy,
    load_config,
    apply_curriculum,
)

# Load configuration and set the evader to be unaware
config = load_config()
config['evader']['awareness_mode'] = 1


def _log_curriculum(writer: SummaryWriter, cfg: dict, start: dict, end: dict, step: int, prefix: str = "") -> None:
    for key, s_val in start.items():
        if key not in end or key not in cfg:
            continue
        e_val = end[key]
        cur = cfg[key]
        if isinstance(s_val, dict) and isinstance(e_val, dict):
            _log_curriculum(writer, cur, s_val, e_val, step, prefix + key + "/")
        elif isinstance(s_val, (int, float)) and isinstance(e_val, (int, float)):
            if s_val != e_val:
                writer.add_scalar(f"curriculum/{prefix}{key}", cur, step)
        elif (
            isinstance(s_val, (list, tuple))
            and isinstance(e_val, (list, tuple))
            and len(s_val) == len(e_val)
        ):
            for i, (sv, ev, cv) in enumerate(zip(s_val, e_val, cur)):
                if sv != ev:
                    writer.add_scalar(
                        f"curriculum/{prefix}{key}_{i}", cv, step
                    )


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

    def __init__(self, cfg: dict, max_steps: int | None = None, capture_bonus: float = 0.0):
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
        self.capture_bonus = capture_bonus
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
        r_p = float(reward['pursuer'])
        if done and info.get('outcome') == 'capture':
            steps = info.get('episode_steps', self.cur_step + 1)
            r_p += self.capture_bonus * (self.max_steps - steps)
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
        return obs['pursuer'].astype(np.float32), r_p, done, truncated, info


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
    ratios = []
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
            start_d = info.get('start_distance')
            min_d = info.get('min_distance')
            if start_d and min_d is not None and start_d > 0:
                ratios.append(min_d / start_d)

    if min_dists:
        msg = (
            f"    eval metrics: mean_min_dist={np.nanmean(min_dists):.2f} "
            f"mean_steps={np.nanmean(steps):.1f}"
        )
        if ratios:
            msg += f" min_start_ratio={np.nanmean(ratios):.3f}"
        print(msg)
    return float(np.mean(rewards)), successes / episodes


def train(
    cfg: dict,
    save_path: Optional[str] = None,
    *,
    checkpoint_every: int | None = None,
    resume_from: str | None = None,
    log_dir: str | None = None,
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
    log_dir:
        Optional directory for TensorBoard logs. When ``None`` no logging is
        performed.
    """

    training_cfg = cfg.get('training', {})
    num_episodes = training_cfg.get('episodes', 100)
    learning_rate = training_cfg.get('learning_rate', 1e-3)
    weight_decay = training_cfg.get('weight_decay', 0.0)
    lr_step_size = training_cfg.get('lr_step_size', 0)
    lr_gamma = training_cfg.get('lr_gamma', 0.95)
    hidden_size = training_cfg.get('hidden_size', 64)
    activation = training_cfg.get('activation', 'relu')
    reward_threshold = training_cfg.get('reward_threshold', 0.0)
    eval_freq = training_cfg.get('eval_freq', 10)
    curriculum_stages = training_cfg.get('curriculum_stages', 2)
    outcome_window = training_cfg.get('outcome_window', 100)
    if checkpoint_every is None:
        checkpoint_every = training_cfg.get('checkpoint_steps')
    curriculum_cfg = training_cfg.get('curriculum')
    start_cur = curriculum_cfg.get('start') if curriculum_cfg else None
    end_cur = curriculum_cfg.get('end') if curriculum_cfg else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PursuerOnlyEnv(cfg)
    policy = PursuerPolicy(
        env.observation_space.shape[0], hidden_size=hidden_size, activation=activation
    ).to(device)
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
    if writer:
        writer.add_text("config/full", yaml.dump(cfg), 0)
    if resume_from:
        state_dict = torch.load(resume_from, map_location=device)
        policy.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {resume_from}")
    optimizer = optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = (
        optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        if lr_step_size and lr_step_size > 0
        else None
    )
    gamma = 0.99

    header = (
        f"{'step':>5} | {'pursuer→evader [m]':>26} | "
        f"{'evader→target [m]':>26} | {'pursuer vel [m/s]':>26} | "
        f"{'evader vel [m/s]':>26} | {'p dir':>18} | {'e dir':>18} | "
        f"{'p→e dir':>18}"
    )

    efficiency_logged = False
    outcome_counts = defaultdict(int)
    # ``curriculum_stages`` counts the discrete phases from the starting
    # configuration to the final one. There are ``curriculum_stages - 1``
    # transitions, and ``stage_idx`` selects the active stage.
    num_transitions = max(curriculum_stages - 1, 1)
    for episode in range(num_episodes):
        stage_idx = (episode * num_transitions) // max(num_episodes - 1, 1)
        progress = stage_idx / num_transitions
        if start_cur and end_cur:
            apply_curriculum(env.env.cfg, start_cur, end_cur, progress)
            if writer:
                _log_curriculum(writer, env.env.cfg, start_cur, end_cur, episode)
        # Collect one episode of experience
        obs, _ = env.reset()
        init_pursuer_pos = env.env.pursuer_pos.copy()
        init_evader_pos = env.env.evader_pos.copy()
        log_probs = []
        rewards = []
        done = False
        info = {}
        rows = []
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
            rows.append(_format_row(len(rows), env.env))
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
        if scheduler:
            scheduler.step()

        print(f"Initial pursuer pos: {init_pursuer_pos}")
        print(f"Initial evader pos: {init_evader_pos}")
        print(header)
        print("-" * len(header))
        for row in first_rows:
            print(row)
        for row in last_rows:
            print(row)
        episode_reward = sum(rewards)
        if writer:
            writer.add_scalar("train/episode_reward", episode_reward, episode)
            if info:
                writer.add_scalar(
                    "train/min_distance",
                    info.get("min_distance", float("nan")),
                    episode,
                )
                writer.add_scalar(
                    "train/episode_length",
                    info.get("episode_steps", step),
                    episode,
                )
                start_d = info.get("start_distance")
                min_d = info.get("min_distance")
                if start_d is not None and min_d is not None and start_d > 0:
                    writer.add_scalar(
                        "train/min_start_ratio",
                        float(min_d) / float(start_d),
                        episode,
                    )
                rb = info.get("reward_breakdown", {})
                for k, v in rb.items():
                    writer.add_scalar(f"train/reward_{k}", v, episode)
            writer.add_scalar("train/loss", loss.item(), episode)
        if info:
            outcome = info.get("outcome", "timeout")
            outcome_counts[outcome] += 1
            if (episode + 1) % outcome_window == 0 and writer:
                total = sum(outcome_counts.values())
                for k, c in outcome_counts.items():
                    writer.add_scalar(
                        f"termination/{k}", c / total, episode
                    )
                outcome_counts = defaultdict(int)
        if info:
            print(
                f"Episode {episode+1}: reward={episode_reward:.2f} "
                f"outcome={info.get('outcome', 'timeout')} start={start_d:.2f} "
                f"min={info.get('min_distance', float('nan')):.2f}"
            )
            print(TABLE_HEADER)
            for row in rows[:3]:
                print(row)
            if len(rows) > 6:
                print("...")
            for row in rows[-3:]:
                print(row)
        if (episode + 1) % eval_freq == 0:
            # Periodically report progress on separate evaluation episodes
            eval_cfg = copy.deepcopy(cfg)
            if start_cur and end_cur:
                apply_curriculum(eval_cfg, start_cur, end_cur, progress)
            avg_r, success = evaluate(policy, PursuerOnlyEnv(eval_cfg))
            print(f"Episode {episode+1}: avg_reward={avg_r:.2f} success={success:.2f}")
            if writer:
                writer.add_scalar("eval/avg_reward", avg_r, episode)
                writer.add_scalar("eval/success_rate", success, episode)
                if (
                    reward_threshold > 0
                    and not efficiency_logged
                    and avg_r >= reward_threshold
                ):
                    writer.add_scalar("sweep/episodes_to_reward", episode + 1, 0)
                    efficiency_logged = True
        if checkpoint_every and save_path and (episode + 1) % checkpoint_every == 0:
            base_name, ext = os.path.splitext(os.path.basename(save_path))
            ckpt_file = f"{base_name}_ckpt_{episode+1}{ext}"
            if log_dir:
                ckpt_dir = os.path.join(log_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            else:
                ckpt_path = os.path.join(os.path.dirname(save_path), ckpt_file)
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Final evaluation after training
    eval_cfg = copy.deepcopy(cfg)
    if start_cur and end_cur:
        apply_curriculum(eval_cfg, start_cur, end_cur, 1.0)
    avg_r, success = evaluate(policy, PursuerOnlyEnv(eval_cfg))
    print(f"Final performance: avg_reward={avg_r:.2f} success={success:.2f}")
    if writer:
        writer.add_scalar("eval/final_avg_reward", avg_r, num_episodes)
        writer.add_scalar("eval/final_success_rate", success, num_episodes)

    if save_path is not None:
        torch.save(policy.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pursuer policy")
    parser.add_argument("--episodes", type=int,
                        help="number of training episodes")
    parser.add_argument("--lr", type=float,
                        help="optimizer learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="L2 weight decay for the optimizer",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        help="StepLR schedule interval (0 to disable)",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        help="StepLR decay factor",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        help="width of the MLP hidden layers",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "tanh", "leaky_relu"],
        help="activation function",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        help="log episodes to reach this avg reward",
    )
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
        "--curriculum-stages",
        type=int,
        help="number of discrete curriculum stages including the final one",
    )
    parser.add_argument(
        "--outcome-window",
        type=int,
        help="episodes per bin for termination statistics",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="start training from this checkpoint file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/reinforce",
        help="write TensorBoard logs to this directory",
    )
    args = parser.parse_args()

    training_cfg = config.setdefault('training', {
        'episodes': 5000,
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'lr_step_size': 0,
        'lr_gamma': 0.95,
        'hidden_size': 64,
        'activation': 'relu',
        'reward_threshold': 0.0,
        'eval_freq': 1000,
        'checkpoint_steps': 0,
        'curriculum_stages': 2,
        'outcome_window': 100,
    })
    if args.episodes is not None:
        training_cfg['episodes'] = args.episodes
    if args.lr is not None:
        training_cfg['learning_rate'] = args.lr
    if args.weight_decay is not None:
        training_cfg['weight_decay'] = args.weight_decay
    if args.lr_step_size is not None:
        training_cfg['lr_step_size'] = args.lr_step_size
    if args.lr_gamma is not None:
        training_cfg['lr_gamma'] = args.lr_gamma
    if args.hidden_size is not None:
        training_cfg['hidden_size'] = args.hidden_size
    if args.activation is not None:
        training_cfg['activation'] = args.activation
    if args.reward_threshold is not None:
        training_cfg['reward_threshold'] = args.reward_threshold
    if args.eval_freq is not None:
        training_cfg['eval_freq'] = args.eval_freq
    if args.checkpoint_every is not None:
        training_cfg['checkpoint_steps'] = args.checkpoint_every
    if args.curriculum_stages is not None:
        training_cfg['curriculum_stages'] = args.curriculum_stages
    if args.outcome_window is not None:
        training_cfg['outcome_window'] = args.outcome_window
    if args.time_step is not None:
        config['time_step'] = args.time_step

    train(
        config,
        save_path=args.save_path,
        checkpoint_every=training_cfg.get('checkpoint_steps'),
        resume_from=args.resume_from,
        log_dir=args.log_dir,
    )
