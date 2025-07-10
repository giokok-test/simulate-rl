from __future__ import annotations


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Optional

from pursuit_evasion import (
    PursuitEvasionEnv,
    PursuerPolicy,
    _make_mlp,
    load_config,
)

# Load configuration and fix the evader policy
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
    """Evader accelerates toward the target."""
    pos = env.evader_pos
    target = np.array(env.cfg['target_position'], dtype=np.float32)
    direction = target - pos
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction /= norm
    theta = np.arctan2(direction[1], direction[0])
    phi = np.arctan2(direction[2], np.linalg.norm(direction[:2]))
    phi = np.clip(phi, -env.cfg['evader']['stall_angle'], env.cfg['evader']['stall_angle'])
    mag = env.cfg['evader']['max_acceleration']
    return np.array([mag, theta, phi], dtype=np.float32)


class PursuerOnlyEnv(gym.Env):
    """Environment exposing only the pursuer."""

    def __init__(self, cfg: dict, max_steps: int | None = None):
        super().__init__()
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


class ActorCritic(nn.Module):
    """Small actor-critic network."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.policy_net = _make_mlp(obs_dim, 3)
        self.value_net = _make_mlp(obs_dim, 1)

    def forward(self, obs: torch.Tensor):
        mean = self.policy_net(obs)
        value = self.value_net(obs).squeeze(-1)
        return mean, value


def evaluate(model: ActorCritic, env: PursuerOnlyEnv, episodes: int = 5):
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
                obs_t = torch.tensor(obs, device=next(model.parameters()).device)
                mean, _ = model(obs_t)
                dist = torch.distributions.Normal(mean, torch.ones_like(mean))
                action = dist.mean
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


def train(cfg: dict, save_path: Optional[str] = None):
    training_cfg = cfg.get('training', {})
    num_episodes = training_cfg.get('episodes', 100)
    learning_rate = training_cfg.get('learning_rate', 1e-3)
    eval_freq = training_cfg.get('eval_freq', 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PursuerOnlyEnv(cfg)
    model = ActorCritic(env.observation_space.shape[0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.99
    clip_ratio = 0.2
    ppo_epochs = 4
    entropy_coef = 0.01

    header = (
        f"{'step':>5} | {'pursuer→evader [m]':>26} | "
        f"{'evader→target [m]':>26} | {'pursuer vel [m/s]':>26} | "
        f"{'evader vel [m/s]':>26} | {'p dir':>18} | {'e dir':>18} | "
        f"{'p→e dir':>18}"
    )

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        obs_list = []
        actions = []
        info = {}
        start_d = env.start_distance
        target = np.asarray(env.env.cfg["target_position"], dtype=float)
        first_rows: list[str] = []
        last_rows: list[str] = []
        step = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mean, value = model(obs_t)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean))
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            next_obs, r, done, _, info = env.step(action.cpu().numpy())
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

        # compute returns and advantages
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.stack(values)
        advantages = returns - values_t

        obs_batch = torch.stack(obs_list)
        action_batch = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)

        for _ in range(ppo_epochs):
            mean, value = model(obs_batch)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean))
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

        print(header)
        print("-" * len(header))
        for row in first_rows:
            print(row)
        for row in last_rows:
            print(row)
        if info:
            print(
                f"Episode {episode+1}: outcome={info.get('outcome', 'timeout')} "
                f"start={start_d:.2f} min={info.get('min_distance', float('nan')):.2f}"
            )

        if (episode + 1) % eval_freq == 0:
            avg_r, success = evaluate(model, PursuerOnlyEnv(cfg))
            print(
                f"Episode {episode+1}: avg_reward={avg_r:.2f} success={success:.2f}"
            )

    avg_r, success = evaluate(model, PursuerOnlyEnv(cfg))
    print(f"Final performance: avg_reward={avg_r:.2f} success={success:.2f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pursuer policy using PPO")
    parser.add_argument("--episodes", type=int, help="number of training episodes")
    parser.add_argument("--lr", type=float, help="optimizer learning rate")
    parser.add_argument("--eval-freq", type=int, help="how often to run evaluation")
    parser.add_argument("--time-step", type=float, help="simulation time step override")
    parser.add_argument("--save-path", type=str, default="pursuer_ppo.pt",
                        help="where to store the trained weights")
    args = parser.parse_args()

    training_cfg = config.setdefault(
        'training', {'episodes': 5000, 'learning_rate': 1e-3, 'eval_freq': 1000}
    )
    if args.episodes is not None:
        training_cfg['episodes'] = args.episodes
    if args.lr is not None:
        training_cfg['learning_rate'] = args.lr
    if args.eval_freq is not None:
        training_cfg['eval_freq'] = args.eval_freq
    if args.time_step is not None:
        config['time_step'] = args.time_step

    train(config, save_path=args.save_path)
