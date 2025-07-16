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
from collections import defaultdict, deque
import yaml
import time

TABLE_HEADER = "{:>5} | {:>26} | {:>26} | {:>26} | {:>26} | {:>18} | {:>18} | {:>18}".format(
    "step",
    "pursuer->evader [m]",
    "evader->target [m]",
    "pursuer vel [m/s]",
    "evader vel [m/s]",
    "p dir",
    "e dir",
    "p->e dir",
)


def _format_row(step: int, env: PursuitEvasionEnv) -> str:
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
    _make_mlp,
    load_config,
    apply_curriculum,
)

# Load configuration and fix the evader policy
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
    """Environment exposing only the pursuer."""

    def __init__(self, cfg: dict, max_steps: int | None = None, capture_bonus: float = 0.0):
        super().__init__()
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
        obs, info = self.env.reset(seed=seed)
        self.cur_step = 0
        self.start_distance = self.env.start_pe_dist
        return obs['pursuer'].astype(np.float32), info

    def step(self, action: np.ndarray):
        e_action = evader_policy(self.env)
        obs, reward, done, truncated, info = self.env.step(
            {'pursuer': action, 'evader': e_action}
        )
        info.setdefault('start_distance', float(self.env.start_pe_dist))
        r_p = float(reward['pursuer'])
        timing_bonus = 0.0
        if done and info.get('outcome') == 'capture':
            steps = info.get('episode_steps', self.cur_step + 1)
            timing_bonus = self.capture_bonus * (self.max_steps - steps)
            r_p += timing_bonus
        info['timing_bonus'] = timing_bonus
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


class ActorCritic(nn.Module):
    """Small actor-critic network."""

    def __init__(self, obs_dim: int, hidden_size: int = 64, activation: str = "relu"):
        super().__init__()
        self.policy_net = _make_mlp(obs_dim, 3, hidden_size, activation)
        self.value_net = _make_mlp(obs_dim, 1, hidden_size, activation)
        # Log standard deviation for the Gaussian policy. Using ``zeros``
        # initialisation mirrors the previous unit variance behaviour.
        self.log_std = nn.Parameter(torch.zeros(3))

    def forward(self, obs: torch.Tensor):
        mean = self.policy_net(obs)
        value = self.value_net(obs).squeeze(-1)
        return mean, value

    @property
    def std(self) -> torch.Tensor:
        """Return the action standard deviation."""
        return self.log_std.exp()


def compute_gae(
    rewards: list[float] | torch.Tensor,
    values: list[torch.Tensor] | torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return discounted returns and advantages using GAE."""

    if not torch.is_tensor(values):
        values = torch.stack(list(values))
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=values.device)
    advantages = torch.zeros_like(rewards, device=values.device)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return returns, advantages


def evaluate(model: ActorCritic, env: PursuerOnlyEnv, episodes: int = 5):
    rewards = []
    successes = 0
    min_dists = []
    steps = []
    ratios = []
    outcome_counts = defaultdict(int)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        info = {}
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, device=next(model.parameters()).device)
                mean, _ = model(obs_t)
                std = model.std.expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                action = dist.mean
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
            outcome_counts[info.get('outcome', 'timeout')] += 1

    if min_dists:
        msg = (
            f"    eval metrics: mean_min_dist={np.nanmean(min_dists):.2f} "
            f"mean_steps={np.nanmean(steps):.1f}"
        )
        if ratios:
            msg += f" min_start_ratio={np.nanmean(ratios):.3f}"
        print(msg)
    if outcome_counts:
        print(f"    termination counts: {dict(outcome_counts)}")
    return float(np.mean(rewards)), successes / episodes


def train(
    cfg: dict,
    save_path: Optional[str] = None,
    *,
    checkpoint_every: int | None = None,
    resume_from: str | None = None,
    log_dir: str | None = None,
    num_envs: int = 8,
    profile: bool = False,
):
    """Train the pursuer policy using PPO.

    Parameters
    ----------
    cfg:
        Configuration dictionary with a ``training`` section.
    save_path:
        Path to store the final model parameters.
    checkpoint_every:
        Interval in episodes between checkpoints when set.
    resume_from:
        Optional checkpoint file to load before starting training.
    log_dir:
        Optional directory for TensorBoard logs. When ``None`` no logging is
        performed.
    num_envs:
        Number of parallel environments to run. Values greater than one use a
        vectorised environment for faster data collection.
    profile:
        Measure time spent collecting rollouts, optimising and evaluating. The
        timings are printed and logged to TensorBoard when a ``log_dir`` is
        provided.
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
    curriculum_mode = training_cfg.get('curriculum_mode', 'linear')
    success_threshold = training_cfg.get('success_threshold', 0.8)
    curriculum_window = training_cfg.get('curriculum_window', 50)
    if checkpoint_every is None:
        checkpoint_every = training_cfg.get('checkpoint_steps')
    curriculum_cfg = training_cfg.get('curriculum')
    start_cur = curriculum_cfg.get('start') if curriculum_cfg else None
    end_cur = curriculum_cfg.get('end') if curriculum_cfg else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_envs > 1:
        def _make() -> PursuerOnlyEnv:
            return PursuerOnlyEnv(cfg)

        env = gym.vector.SyncVectorEnv([_make for _ in range(num_envs)])
        obs_space = env.single_observation_space
    else:
        env = PursuerOnlyEnv(cfg)
        obs_space = env.observation_space

    model = ActorCritic(
        obs_space.shape[0], hidden_size=hidden_size, activation=activation
    ).to(device)
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
    if writer:
        writer.add_text("config/full", yaml.dump(cfg), 0)
    if resume_from:
        state_dict = torch.load(resume_from, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {resume_from}")
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = (
        optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        if lr_step_size and lr_step_size > 0
        else None
    )

    episode_counter = 0
    start_time = time.perf_counter()

    gamma = training_cfg.get('gamma', 0.99)
    clip_ratio = training_cfg.get('clip_ratio', 0.2)
    ppo_epochs = training_cfg.get('ppo_epochs', 4)
    entropy_start = training_cfg.get('entropy_coef_start', 0.01)
    entropy_end = training_cfg.get('entropy_coef_end', entropy_start)

    header = "{:>5} | {:>26} | {:>26} | {:>26} | {:>26} | {:>18} | {:>18} | {:>18}".format(
        "step",
        "pursuer->evader [m]",
        "evader->target [m]",
        "pursuer vel [m/s]",
        "evader vel [m/s]",
        "p dir",
        "e dir",
        "p->e dir",
    )

    efficiency_logged = False
    outcome_counts = defaultdict(int)
    # ``curriculum_stages`` counts the discrete phases from the starting
    # configuration to the final one. There are ``curriculum_stages - 1``
    # transitions. ``stage_idx`` selects the active stage.
    num_transitions = max(curriculum_stages - 1, 1)
    stage_idx = 0
    recent = deque(maxlen=curriculum_window)

    for episode in range(num_episodes):
        if curriculum_mode == 'linear':
            stage_idx = (episode * num_transitions) // max(num_episodes - 1, 1)
        progress = stage_idx / num_transitions
        episode_progress = episode / max(num_episodes - 1, 1)
        entropy_coef = entropy_start + (entropy_end - entropy_start) * episode_progress
        if start_cur and end_cur:
            if num_envs == 1:
                apply_curriculum(env.env.cfg, start_cur, end_cur, progress)
                if writer:
                    _log_curriculum(writer, env.env.cfg, start_cur, end_cur, episode)
            else:
                for e in env.envs:
                    apply_curriculum(e.env.cfg, start_cur, end_cur, progress)
                if writer:
                    _log_curriculum(writer, env.envs[0].env.cfg, start_cur, end_cur, episode)
        if profile:
            collect_start = time.perf_counter()
        if num_envs == 1:
            obs, _ = env.reset()
            init_pursuer_pos = env.env.pursuer_pos.copy()
            init_evader_pos = env.env.evader_pos.copy()
            done = False
            log_probs = []
            values = []
            rewards = []
            obs_list = []
            actions = []
            info = {}
            rows = []
            start_d = env.start_distance
            target = np.asarray(env.env.cfg["target_position"], dtype=float)
            first_rows: list[str] = []
            last_rows: list[str] = []
            step = 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                mean, value = model(obs_t)
                std = model.std.expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                next_obs, r, done, _, info = env.step(action.cpu().numpy())
                rows.append(_format_row(len(rows), env.env))
                log_probs.append(log_prob.detach())
                values.append(value.detach())
                rewards.append(r)
                obs_list.append(obs_t)
                actions.append(action)
                row = _format_step(env, step, target)
                if len(first_rows) < 3:
                    first_rows.append(row)
                if len(last_rows) >= 3:
                    last_rows.pop(0)
                last_rows.append(row)
                obs = next_obs
                step += 1

            values_t = torch.stack(values)
            returns, advantages = compute_gae(
                rewards, values_t, gamma=gamma, lam=0.95
            )
            # Normalize advantages for more stable updates
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            obs_batch = torch.stack(obs_list)
            action_batch = torch.stack(actions)
            old_log_probs = torch.stack(log_probs)
            if profile:
                collect_time = time.perf_counter() - collect_start
                update_start = time.perf_counter()

        else:
            obs, _ = env.reset()
            done = np.zeros(num_envs, dtype=bool)
            log_probs = [[] for _ in range(num_envs)]
            values = [[] for _ in range(num_envs)]
            rewards = [[] for _ in range(num_envs)]
            obs_list = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
            infos = [None for _ in range(num_envs)]
            step = 0
            while not np.all(done):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                mean, value = model(obs_t)
                std = model.std.expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=1)
                next_obs, r, d, _, info = env.step(action.cpu().numpy())
                if isinstance(info, dict):
                    # info is something like {'episode_steps': array([ 12,  34, ...]),
                    #                        'min_distance': array([10.2,  5.6, ...]), ...}
                    info_list = []
                    for idx in range(num_envs):
                        single = {}
                        for key, val in info.items():
                            # if val is array-like, index it; otherwise, copy directly
                            if hasattr(val, "__len__") and not isinstance(val, dict):
                                try:
                                    single[key] = val[idx]
                                except Exception:
                                    single[key] = val         # fallback if indexing fails
                            else:
                                single[key] = val             # nested dicts, scalars, etc.
                        info_list.append(single)
                else:
                    info_list = info
                for i in range(num_envs):
                    if not done[i]:
                        log_probs[i].append(log_prob[i].detach())
                        values[i].append(value[i].detach())
                        rewards[i].append(r[i])
                        obs_list[i].append(obs_t[i])
                        actions[i].append(action[i])
                    if d[i] and infos[i] is None:
                        # now pull from our per-env dict
                        infos[i] = info_list[i]
                done = np.logical_or(done, d)
                obs = next_obs
                step += 1

            ret_list = []
            val_list = []
            log_list = []
            obs_stack = []
            action_stack = []
            adv_list = []
            for i in range(num_envs):
                vals_i = torch.stack(values[i])
                rets_i, adv_i = compute_gae(
                    rewards[i], vals_i, gamma=gamma, lam=0.95
                )
                ret_list.append(rets_i)
                adv_list.append(adv_i)
                val_list.append(vals_i)
                log_list.append(torch.stack(log_probs[i]))
                obs_stack.append(torch.stack(obs_list[i]))
                action_stack.append(torch.stack(actions[i]))
            returns = torch.cat(ret_list)
            values_t = torch.cat(val_list)
            advantages = torch.cat(adv_list)
            # Normalize advantages for more stable updates
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )
            obs_batch = torch.cat(obs_stack)
            action_batch = torch.cat(action_stack)
            old_log_probs = torch.cat(log_list)
            if profile:
                collect_time = time.perf_counter() - collect_start
                update_start = time.perf_counter()

        for _ in range(ppo_epochs):
            mean, value = model(obs_batch)
            std = model.std.expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            log_probs_new = dist.log_prob(action_batch).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            ratio = torch.exp(log_probs_new - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        if profile:
            update_time = time.perf_counter() - update_start

        if writer:
            writer.add_scalar("train/loss", loss.item(), episode)
            writer.add_scalar("train/entropy_coef", entropy_coef, episode)

        if num_envs == 1:
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
                batch_step = episode_counter + 1
                writer.add_scalar("batch/episode_reward", episode_reward, batch_step)
                writer.add_scalar("episode/reward", episode_reward, episode_counter)
                if info:
                    writer.add_scalar(
                        "batch/min_distance",
                        info.get("min_distance", float("nan")),
                        batch_step,
                    )
                    writer.add_scalar(
                        "episode/min_distance",
                        info.get("min_distance", float("nan")),
                        episode_counter,
                    )
                writer.add_scalar(
                    "batch/episode_length",
                    info.get("episode_steps", step),
                    batch_step,
                )
                writer.add_scalar(
                    "episode/length",
                    info.get("episode_steps", step),
                    episode_counter,
                )
                start_d = info.get("start_distance")
                min_d = info.get("min_distance")
                if start_d is not None and min_d is not None and start_d > 0:
                    writer.add_scalar(
                        "batch/min_start_ratio",
                        float(min_d) / float(start_d),
                        batch_step,
                    )
                    writer.add_scalar(
                        "episode/min_start_ratio",
                        float(min_d) / float(start_d),
                        episode_counter,
                    )
                writer.add_scalar(
                    "batch/acc_delta",
                    info.get("pursuer_acc_delta", float("nan")),
                    batch_step,
                )
                writer.add_scalar(
                    "episode/acc_delta",
                    info.get("pursuer_acc_delta", float("nan")),
                    episode_counter,
                )
                writer.add_scalar(
                    "batch/yaw_delta",
                    info.get("pursuer_yaw_delta", float("nan")),
                    batch_step,
                )
                writer.add_scalar(
                    "episode/yaw_delta",
                    info.get("pursuer_yaw_delta", float("nan")),
                    episode_counter,
                )
                writer.add_scalar(
                    "batch/pitch_delta",
                    info.get("pursuer_pitch_delta", float("nan")),
                    batch_step,
                )
                writer.add_scalar(
                    "episode/pitch_delta",
                    info.get("pursuer_pitch_delta", float("nan")),
                    episode_counter,
                )
                writer.add_scalar(
                    "batch/vel_delta",
                    info.get("pursuer_vel_delta", float("nan")),
                    batch_step,
                )
                writer.add_scalar(
                    "episode/vel_delta",
                    info.get("pursuer_vel_delta", float("nan")),
                    episode_counter,
                )
                writer.add_scalar(
                    "batch/yaw_diff",
                    info.get("pursuer_yaw_diff", float("nan")),
                    batch_step,
                )
                writer.add_scalar(
                    "episode/yaw_diff",
                    info.get("pursuer_yaw_diff", float("nan")),
                    episode_counter,
                )
                writer.add_scalar(
                    "batch/pitch_diff",
                    info.get("pursuer_pitch_diff", float("nan")),
                    batch_step,
                )
                writer.add_scalar(
                    "episode/pitch_diff",
                    info.get("pursuer_pitch_diff", float("nan")),
                    episode_counter,
                )
                rb = info.get("reward_breakdown", {})
                for k, v in rb.items():
                    scalar_reward = float(v)
                    writer.add_scalar(
                        f"episode/reward_{k}", scalar_reward, episode_counter
                    )
                episode_counter += 1
            if info:
                outcome = info.get('outcome', 'timeout')
                outcome_counts[outcome] += 1
                if curriculum_mode == 'adaptive':
                    recent.append(1 if outcome == 'capture' else 0)
                    if (
                        len(recent) >= curriculum_window
                        and sum(recent) / len(recent) >= success_threshold
                        and stage_idx < num_transitions
                    ):
                        stage_idx += 1
                        recent.clear()
                if (episode + 1) % outcome_window == 0 and writer:
                    total = sum(outcome_counts.values())
                    for k, c in outcome_counts.items():
                        writer.add_scalar(
                            f"termination/{k}", c / total, episode
                        )
                    print(
                        f"Termination counts (last {outcome_window} episodes): "
                        f"{dict(outcome_counts)}"
                    )
                    outcome_counts = defaultdict(int)
                print(
                    f"Episode {episode+1}: reward={episode_reward:.2f} "
                    f"outcome={outcome} start={start_d:.2f} "
                    f"min={info.get('min_distance', float('nan')):.2f}"
                )
                print(TABLE_HEADER)
                for row in rows[:3]:
                    print(row)
                if len(rows) > 6:
                    print("...")
                for row in rows[-3:]:
                    print(row)
        else:
            episode_reward = sum(sum(r) for r in rewards) / num_envs
            episode_outcomes = defaultdict(int)
            if writer:
                batch_step = episode_counter + num_envs
                writer.add_scalar("batch/episode_reward", episode_reward, batch_step)
                for i in range(num_envs):
                    env_r = sum(rewards[i])
                    writer.add_scalar(
                        "episode/reward", env_r, episode_counter + i
                    )
                md_vals = [inf.get("min_distance", float("nan")) for inf in infos if inf]
                step_vals = [inf.get("episode_steps", float("nan")) for inf in infos if inf]
                if md_vals:
                    writer.add_scalar(
                        "batch/min_distance",
                        float(np.nanmean(md_vals)),
                        batch_step,
                    )
                for i, inf in enumerate(infos):
                    if inf and "min_distance" in inf:
                        writer.add_scalar(
                            "episode/min_distance",
                            float(inf["min_distance"]),
                            episode_counter + i,
                        )
                if step_vals:
                    writer.add_scalar(
                        "batch/episode_length",
                        float(np.nanmean(step_vals)),
                        batch_step,
                    )
                for i, inf in enumerate(infos):
                    if inf and "episode_steps" in inf:
                        writer.add_scalar(
                            "episode/length",
                            float(inf["episode_steps"]),
                            episode_counter + i,
                        )
                    if inf:
                        start_d = inf.get("start_distance")
                        min_d = inf.get("min_distance")
                        if start_d is not None and min_d is not None and start_d > 0:
                            writer.add_scalar(
                                "episode/min_start_ratio",
                                float(min_d) / float(start_d),
                                episode_counter + i,
                            )
                        writer.add_scalar(
                            "episode/acc_delta",
                            float(inf.get("pursuer_acc_delta", float("nan"))),
                            episode_counter + i,
                        )
                        writer.add_scalar(
                            "episode/yaw_delta",
                            float(inf.get("pursuer_yaw_delta", float("nan"))),
                            episode_counter + i,
                        )
                        writer.add_scalar(
                            "episode/pitch_delta",
                            float(inf.get("pursuer_pitch_delta", float("nan"))),
                            episode_counter + i,
                        )
                        writer.add_scalar(
                            "episode/vel_delta",
                            float(inf.get("pursuer_vel_delta", float("nan"))),
                            episode_counter + i,
                        )
                rb_sum = defaultdict(float)
                n_info = 0
                min_list = []
                len_list = []
                start_list = []
                acc_list = []
                yaw_list = []
                pitch_list = []
                vel_list = []
                yaw_diff_list = []
                pitch_diff_list = []
                for inf in infos:
                    if inf:
                        n_info += 1
                        for k, v in inf.get("reward_breakdown", {}).items():
                            arr = np.asarray(v)    
                            scalar = float(arr.mean())
                            rb_sum[k] += scalar
                        outcome = inf.get("outcome", "timeout")
                        outcome_counts[outcome] += 1
                        episode_outcomes[outcome] += 1
                        if curriculum_mode == 'adaptive':
                            recent.append(1 if outcome == 'capture' else 0)
                            if (
                                len(recent) >= curriculum_window
                                and sum(recent) / len(recent) >= success_threshold
                                and stage_idx < num_transitions
                            ):
                                stage_idx += 1
                                recent.clear()
                        if "min_distance" in inf:
                            min_list.append(inf["min_distance"])
                        if "episode_steps" in inf:
                            len_list.append(inf["episode_steps"])
                        if "start_distance" in inf:
                            start_list.append(inf["start_distance"])
                        if "pursuer_acc_delta" in inf:
                            acc_list.append(inf["pursuer_acc_delta"])
                        if "pursuer_yaw_delta" in inf:
                            yaw_list.append(inf["pursuer_yaw_delta"])
                        if "pursuer_pitch_delta" in inf:
                            pitch_list.append(inf["pursuer_pitch_delta"])
                        if "pursuer_vel_delta" in inf:
                            vel_list.append(inf["pursuer_vel_delta"])
                        if "pursuer_yaw_diff" in inf:
                            yaw_diff_list.append(inf["pursuer_yaw_diff"])
                        if "pursuer_pitch_diff" in inf:
                            pitch_diff_list.append(inf["pursuer_pitch_diff"])
                if n_info:
                    for k, v in rb_sum.items():
                        avg = v / n_info
                        scalar_reward = float(avg.item() if hasattr(avg, "item") else avg)
                        writer.add_scalar(f"batch/reward_{k}", scalar_reward, episode)
                if min_list:
                    writer.add_scalar("batch/min_distance", float(np.mean(min_list)), batch_step)
                if len_list:
                    writer.add_scalar("batch/episode_length", float(np.mean(len_list)), batch_step)
                if acc_list:
                    writer.add_scalar("batch/acc_delta", float(np.mean(acc_list)), batch_step)
                if yaw_list:
                    writer.add_scalar("batch/yaw_delta", float(np.mean(yaw_list)), batch_step)
                if pitch_list:
                    writer.add_scalar("batch/pitch_delta", float(np.mean(pitch_list)), batch_step)
                if vel_list:
                    writer.add_scalar("batch/vel_delta", float(np.mean(vel_list)), batch_step)
                if yaw_diff_list:
                    writer.add_scalar("batch/yaw_diff", float(np.mean(yaw_diff_list)), batch_step)
                if pitch_diff_list:
                    writer.add_scalar("batch/pitch_diff", float(np.mean(pitch_diff_list)), batch_step)
                if min_list and start_list:
                    ratios = [m / s for m, s in zip(min_list, start_list) if s > 0]
                    if ratios:
                        writer.add_scalar("batch/min_start_ratio", float(np.mean(ratios)), batch_step)
                for i, inf in enumerate(infos):
                    if inf:
                        rb_env = inf.get("reward_breakdown", {})
                        for k, v in rb_env.items():
                            arr = np.asarray(v)
                            scalar = float(arr.mean())  # or .sum() if you’d rather aggregate that way
                            writer.add_scalar(f"episode/reward_{k}", scalar, episode_counter + i)
                # per-environment min_start_ratio logged earlier
                episode_counter += num_envs
                if (episode + 1) % outcome_window == 0:
                    total = sum(outcome_counts.values())
                    for k, c in outcome_counts.items():
                        writer.add_scalar(
                            f"termination/{k}", c / total, episode
                        )
                    print(
                        f"Termination counts (last {outcome_window} episodes): "
                        f"{dict(outcome_counts)}"
                    )
                    outcome_counts = defaultdict(int)
            if episode_outcomes:
                print(f"Episode {episode+1}: outcomes={dict(episode_outcomes)} reward={episode_reward:.2f}")

        if writer:
            eps_sec = episode_counter / max(time.perf_counter() - start_time, 1e-8)
            writer.add_scalar("timing/episodes_per_sec", eps_sec, episode_counter)
            if profile:
                writer.add_scalar("timing/collect", collect_time, episode)
                writer.add_scalar("timing/update", update_time, episode)
        if profile:
            print(
                f"Episode {episode+1}: collect={collect_time:.3f}s update={update_time:.3f}s"
            )

        if (episode + 1) % eval_freq == 0:
            if profile:
                eval_start = time.perf_counter()
            eval_cfg = copy.deepcopy(cfg)
            if start_cur and end_cur:
                # Use final curriculum parameters when evaluating to track
                # performance in the target environment regardless of the
                # current training stage.
                apply_curriculum(eval_cfg, start_cur, end_cur, 1.0)
            avg_r, success = evaluate(model, PursuerOnlyEnv(eval_cfg))
            print(
                f"Episode {episode+1}: avg_reward={avg_r:.2f} success={success:.2f}"
            )
            if profile:
                eval_time = time.perf_counter() - eval_start
            if writer:
                writer.add_scalar("eval/avg_reward", avg_r, episode)
                writer.add_scalar("eval/success_rate", success, episode)
                if profile:
                    writer.add_scalar("timing/eval", eval_time, episode)
            if profile:
                print(f"Evaluation time: {eval_time:.3f}s")
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
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    eval_cfg = copy.deepcopy(cfg)
    if start_cur and end_cur:
        apply_curriculum(eval_cfg, start_cur, end_cur, 1.0)
    avg_r, success = evaluate(model, PursuerOnlyEnv(eval_cfg))
    print(f"Final performance: avg_reward={avg_r:.2f} success={success:.2f}")
    if writer:
        writer.add_scalar("eval/final_avg_reward", avg_r, num_episodes)
        writer.add_scalar("eval/final_success_rate", success, num_episodes)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pursuer policy using PPO")
    parser.add_argument("--episodes", type=int, help="number of training episodes")
    parser.add_argument("--lr", type=float, help="optimizer learning rate")
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
    parser.add_argument("--eval-freq", type=int, help="how often to run evaluation")
    parser.add_argument("--time-step", type=float, help="simulation time step override")
    parser.add_argument("--save-path", type=str, default="pursuer_ppo.pt",
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
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/ppo",
        help="write TensorBoard logs to this directory",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="number of parallel environments",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="measure time spent in major training phases",
    )
    parser.add_argument("--gamma", type=float, help="discount factor")
    parser.add_argument("--clip-ratio", type=float, help="PPO clipping ratio")
    parser.add_argument(
        "--ppo-epochs", type=int, help="number of optimisation epochs per batch"
    )
    parser.add_argument(
        "--curriculum-stages",
        type=int,
        help="number of discrete curriculum stages including the final one",
    )
    parser.add_argument(
        "--curriculum-mode",
        type=str,
        choices=["linear", "adaptive"],
        help="curriculum progression mode",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        help="success rate required to advance the curriculum",
    )
    parser.add_argument(
        "--curriculum-window",
        type=int,
        help="episodes used to compute adaptive success rate",
    )
    parser.add_argument(
        "--outcome-window",
        type=int,
        help="episodes per bin for termination statistics",
    )
    parser.add_argument(
        "--entropy-coef-start",
        type=float,
        help="initial entropy bonus weight",
    )
    parser.add_argument(
        "--entropy-coef-end",
        type=float,
        help="final entropy bonus weight",
    )
    args = parser.parse_args()

    training_cfg = config.setdefault(
        'training', {
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
            'curriculum_mode': 'linear',
            'success_threshold': 0.8,
            'curriculum_window': 50,
            'curriculum_stages': 2,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'ppo_epochs': 4,
            'entropy_coef_start': 0.01,
            'entropy_coef_end': 0.01,
            'outcome_window': 100,
        },
    )
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
    if args.curriculum_mode is not None:
        training_cfg['curriculum_mode'] = args.curriculum_mode
    if args.success_threshold is not None:
        training_cfg['success_threshold'] = args.success_threshold
    if args.curriculum_window is not None:
        training_cfg['curriculum_window'] = args.curriculum_window
    if args.gamma is not None:
        training_cfg['gamma'] = args.gamma
    if args.clip_ratio is not None:
        training_cfg['clip_ratio'] = args.clip_ratio
    if args.ppo_epochs is not None:
        training_cfg['ppo_epochs'] = args.ppo_epochs
    if args.curriculum_stages is not None:
        training_cfg['curriculum_stages'] = args.curriculum_stages
    if args.outcome_window is not None:
        training_cfg['outcome_window'] = args.outcome_window
    if args.entropy_coef_start is not None:
        training_cfg['entropy_coef_start'] = args.entropy_coef_start
    if args.entropy_coef_end is not None:
        training_cfg['entropy_coef_end'] = args.entropy_coef_end
    if args.time_step is not None:
        config['time_step'] = args.time_step

    train(
        config,
        save_path=args.save_path,
        checkpoint_every=training_cfg.get('checkpoint_steps'),
        resume_from=args.resume_from,
        log_dir=args.log_dir,
        num_envs=args.num_envs,
        profile=args.profile,
    )
