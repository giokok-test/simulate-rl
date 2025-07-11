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

TABLE_HEADER = (
    f"{'step':>5} | {'pursuer→evader [m]':>26} | "
    f"{'evader→target [m]':>26} | {'pursuer vel [m/s]':>26} | "
    f"{'evader vel [m/s]':>26} | {'p dir':>18} | "
    f"{'e dir':>18} | {'p→e dir':>18}"
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
    PursuerPolicy,
    _make_mlp,
    load_config,
    apply_curriculum,
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

    def __init__(self, obs_dim: int, hidden_size: int = 64, activation: str = "relu"):
        super().__init__()
        self.policy_net = _make_mlp(obs_dim, 3, hidden_size, activation)
        self.value_net = _make_mlp(obs_dim, 1, hidden_size, activation)

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


def train(
    cfg: dict,
    save_path: Optional[str] = None,
    *,
    checkpoint_every: int | None = None,
    resume_from: str | None = None,
    log_dir: str | None = None,
    num_envs: int = 1,
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
    if checkpoint_every is None:
        checkpoint_every = training_cfg.get('checkpoint_steps')
    curriculum_cfg = training_cfg.get('curriculum')
    start_cur = curriculum_cfg.get('start') if curriculum_cfg else None
    end_cur = curriculum_cfg.get('end') if curriculum_cfg else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_envs > 1:
        def _make() -> PursuerOnlyEnv:
            return PursuerOnlyEnv(cfg)

        env = gym.vector.AsyncVectorEnv([_make for _ in range(num_envs)])
        obs_space = env.single_observation_space
    else:
        env = PursuerOnlyEnv(cfg)
        obs_space = env.observation_space

    model = ActorCritic(
        obs_space.shape[0], hidden_size=hidden_size, activation=activation
    ).to(device)
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
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

    efficiency_logged = False

    for episode in range(num_episodes):
        progress = episode / max(num_episodes - 1, 1)
        if start_cur and end_cur:
            if num_envs == 1:
                apply_curriculum(env.env.cfg, start_cur, end_cur, progress)
            else:
                for e in env.envs:
                    apply_curriculum(e.env.cfg, start_cur, end_cur, progress)
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
                dist = torch.distributions.Normal(mean, torch.ones_like(mean))
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
                dist = torch.distributions.Normal(mean, torch.ones_like(mean))
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=1)
                next_obs, r, d, _, info = env.step(action.cpu().numpy())
                for i in range(num_envs):
                    if not done[i]:
                        log_probs[i].append(log_prob[i].detach())
                        values[i].append(value[i].detach())
                        rewards[i].append(r[i])
                        obs_list[i].append(obs_t[i])
                        actions[i].append(action[i])
                    if d[i] and infos[i] is None:
                        infos[i] = info[i]
                done = np.logical_or(done, d)
                obs = next_obs
                step += 1

            ret_list = []
            val_list = []
            log_list = []
            obs_stack = []
            action_stack = []
            for i in range(num_envs):
                G = 0.0
                returns_i = []
                for rr in reversed(rewards[i]):
                    G = rr + gamma * G
                    returns_i.insert(0, G)
                ret_list.append(torch.tensor(returns_i, dtype=torch.float32, device=device))
                val_list.append(torch.stack(values[i]))
                log_list.append(torch.stack(log_probs[i]))
                obs_stack.append(torch.stack(obs_list[i]))
                action_stack.append(torch.stack(actions[i]))
            returns = torch.cat(ret_list)
            values_t = torch.cat(val_list)
            advantages = returns - values_t
            obs_batch = torch.cat(obs_stack)
            action_batch = torch.cat(action_stack)
            old_log_probs = torch.cat(log_list)

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
        if scheduler:
            scheduler.step()

        if writer:
            writer.add_scalar("train/loss", loss.item(), episode)

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
        else:
            episode_reward = sum(sum(r) for r in rewards) / num_envs
            if writer:
                writer.add_scalar("train/episode_reward", episode_reward, episode)

        if (episode + 1) % eval_freq == 0:
            eval_cfg = copy.deepcopy(cfg)
            if start_cur and end_cur:
                apply_curriculum(eval_cfg, start_cur, end_cur, progress)
            avg_r, success = evaluate(model, PursuerOnlyEnv(eval_cfg))
            print(
                f"Episode {episode+1}: avg_reward={avg_r:.2f} success={success:.2f}"
            )
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
            base, ext = os.path.splitext(save_path)
            ckpt_path = f"{base}_ckpt_{episode+1}{ext}"
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
        default=1,
        help="number of parallel environments",
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
    if args.time_step is not None:
        config['time_step'] = args.time_step

    train(
        config,
        save_path=args.save_path,
        checkpoint_every=training_cfg.get('checkpoint_steps'),
        resume_from=args.resume_from,
        log_dir=args.log_dir,
        num_envs=args.num_envs,
    )
