import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from pursuit_evasion import PursuitEvasionEnv, PursuerPolicy, load_config

# Load configuration and set the evader to be unaware
config = load_config()
config['evader']['awareness_mode'] = 1


def evader_policy(env: PursuitEvasionEnv) -> np.ndarray:
    """Evader accelerates toward the target."""
    pos = env.evader_pos
    target = np.array(env.cfg['target_position'], dtype=np.float32)
    direction = target - pos
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction /= norm
    theta = np.arctan2(direction[1], direction[0])
    phi = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.clip(phi, 0.0, env.cfg['evader']['stall_angle'])
    mag = env.cfg['evader']['max_acceleration']
    return np.array([mag, theta, phi], dtype=np.float32)


class PursuerOnlyEnv(gym.Env):
    """Environment exposing only the pursuer. The evader follows ``evader_policy``."""

    def __init__(self, cfg: dict, max_steps: int = 20):
        super().__init__()
        # Full pursuit-evasion environment internally used
        self.env = PursuitEvasionEnv(cfg)
        self.observation_space = self.env.observation_space['pursuer']
        self.action_space = self.env.action_space['pursuer']
        self.max_steps = max_steps
        self.cur_step = 0

    def reset(self, *, seed=None, options=None):
        """Reset the wrapped environment and return the pursuer observation."""

        obs, info = self.env.reset(seed=seed)
        self.cur_step = 0
        return obs['pursuer'].astype(np.float32), info

    def step(self, action: np.ndarray):
        """Take a step using the pursuer action while the evader follows its fixed policy."""

        e_action = evader_policy(self.env)
        obs, reward, done, truncated, info = self.env.step({'pursuer': action, 'evader': e_action})
        self.cur_step += 1
        if self.cur_step >= self.max_steps:
            done = True
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
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                action = policy(torch.tensor(obs, device=next(policy.parameters()).device))
            obs, r, done, _, _ = env.step(action.cpu().numpy())
            total += r
        rewards.append(total)
        if total > 0:
            successes += 1
    return float(np.mean(rewards)), successes / episodes


def train(cfg: dict):
    """Train the pursuer policy with REINFORCE.

    Parameters
    ----------
    cfg:
        Configuration dictionary. Expected to contain a ``training`` section
        specifying ``episodes``, ``learning_rate`` and ``eval_freq``.
    """

    training_cfg = cfg.get('training', {})
    num_episodes = training_cfg.get('episodes', 100)
    learning_rate = training_cfg.get('learning_rate', 1e-3)
    eval_freq = training_cfg.get('eval_freq', 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PursuerOnlyEnv(cfg)
    policy = PursuerPolicy(env.observation_space.shape[0]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    gamma = 0.99

    for episode in range(num_episodes):
        # Collect one episode of experience
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mean = policy(obs_t)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean))
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            obs, r, done, _, _ = env.step(action.cpu().numpy())
            log_probs.append(log_prob)
            rewards.append(r)
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
        if (episode + 1) % eval_freq == 0:
            # Periodically report progress on separate evaluation episodes
            avg_r, success = evaluate(policy, PursuerOnlyEnv(config))
            print(f"Episode {episode+1}: avg_reward={avg_r:.2f} success={success:.2f}")

    # Final evaluation after training
    avg_r, success = evaluate(policy, PursuerOnlyEnv(config))
    print(f"Final performance: avg_reward={avg_r:.2f} success={success:.2f}")


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
    args = parser.parse_args()

    training_cfg = config.setdefault('training', {
        'episodes': 100,
        'learning_rate': 1e-3,
        'eval_freq': 10,
    })
    if args.episodes is not None:
        training_cfg['episodes'] = args.episodes
    if args.lr is not None:
        training_cfg['learning_rate'] = args.lr
    if args.eval_freq is not None:
        training_cfg['eval_freq'] = args.eval_freq
    if args.time_step is not None:
        config['time_step'] = args.time_step

    train(config)
