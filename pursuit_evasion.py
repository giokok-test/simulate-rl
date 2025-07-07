import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# Global configuration dictionary for the environment and agents
config = {
    'evader': {
        'mass': 500.0,
        'max_acceleration': 20.0,
        'top_speed': 300.0,
        'drag_coefficient': 0.02,
        'awareness_mode': 1,   # 1=unaware,2=vague,3=directional,4=full
        'turn_rate': np.pi,    # rad/s
        'up_vector': (0.0, 0.0, 1.0),
        'stall_angle': np.deg2rad(60),  # max angle from up vector
    },
    'pursuer': {
        'mass': 1000.0,
        'max_acceleration': 30.0,
        'top_speed': 350.0,
        'drag_coefficient': 0.03,
        'turn_rate': np.pi * 1.5,
        'up_vector': (0.0, 0.0, 1.0),
        'stall_angle': np.deg2rad(75),
    },
    'gravity': 9.81,
    'time_step': 0.1,
    'capture_radius': 1.0,
    # weight applied to per-step shaping rewards
    'shaping_weight': 0.05,
    'target_position': (1000.0, 0.0, 0.0),
    # parameters controlling the pursuer initial position and orientation
    # The pursuer is sampled in a cone beneath the evader. "cone_half_angle"
    # controls how wide the cone is, while the range limits specify how far
    # from the evader (in metres) the pursuer can start. "force_target_radius"
    # defines the radius of the sphere around the evader that the initial force
    # vector will be pointed toward. These values influence how early or late
    # interceptions can occur during an episode.
    'pursuer_start': {
        'cone_half_angle': np.deg2rad(45.0),
        'min_range': 1000.0,
        'max_range': 5000.0,
        'force_target_radius': 500.0,
        'initial_speed_range': (0.0, 50.0),
    },
    'initial_positions': {
        'evader': (0.0, 0.0, 3000.0),
    }
}


def sample_pursuer_start(evader_pos: np.ndarray, cfg: dict):
    """Sample initial pursuer state and force direction."""
    params = cfg['pursuer_start']
    cone = params['cone_half_angle']
    r = np.random.uniform(params['min_range'], params['max_range'])
    yaw = np.random.uniform(0.0, 2 * np.pi)
    pitch = np.random.uniform(0.0, cone)
    # direction from evader to pursuer in world frame (beneath the evader)
    dir_vec = np.array([
        np.sin(pitch) * np.cos(yaw),
        np.sin(pitch) * np.sin(yaw),
        -np.cos(pitch),
    ], dtype=np.float32)
    pos = evader_pos + dir_vec * r

    speed = np.random.uniform(*params['initial_speed_range'])
    vel = dir_vec * speed

    # point the initial force vector toward a random point near the evader
    tgt_offset = np.random.randn(3).astype(np.float32)
    tgt_offset /= np.linalg.norm(tgt_offset) + 1e-8
    tgt_offset *= np.random.uniform(0.0, params['force_target_radius'])
    tgt = evader_pos + tgt_offset
    to_tgt = tgt - pos
    to_tgt /= np.linalg.norm(to_tgt) + 1e-8
    return pos, vel, to_tgt


class PursuitEvasionEnv(gym.Env):
    """Gym-compatible 3D pursuit-evasion environment."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg['time_step']
        self.shaping_weight = cfg.get('shaping_weight', 0.05)
        # observation sizes depend on awareness mode
        self.evader_obs_dim = 9
        mode = cfg['evader'].get('awareness_mode', 1)
        if mode == 2:
            self.evader_obs_dim += 1  # distance to pursuer
        elif mode == 3:
            self.evader_obs_dim += 3  # directional unit vector
        elif mode >= 4:
            self.evader_obs_dim += 3  # pursuer position
        self.pursuer_obs_dim = 9  # pursuer observes evader position

        self.observation_space = gym.spaces.Dict({
            'pursuer': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.pursuer_obs_dim,), dtype=np.float32),
            'evader': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.evader_obs_dim,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Dict({
            # actions: [acceleration magnitude, azimuth, polar]
            'pursuer': gym.spaces.Box(
                low=np.array([0.0, -np.pi, 0.0], dtype=np.float32),
                high=np.array([
                    cfg['pursuer']['max_acceleration'],
                    np.pi,
                    cfg['pursuer']['stall_angle'],
                ], dtype=np.float32),
            ),
            'evader': gym.spaces.Box(
                low=np.array([0.0, -np.pi, 0.0], dtype=np.float32),
                high=np.array([
                    cfg['evader']['max_acceleration'],
                    np.pi,
                    cfg['evader']['stall_angle'],
                ], dtype=np.float32),
            ),
        })
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.evader_pos = np.array(self.cfg['initial_positions']['evader'], dtype=np.float32)
        self.evader_vel = np.zeros(3, dtype=np.float32)
        self.evader_force_dir = np.array(self.cfg['evader']['up_vector'], dtype=np.float32)
        self.evader_force_dir /= np.linalg.norm(self.evader_force_dir) + 1e-8
        self.evader_force_mag = 0.0

        p_pos, p_vel, p_dir = sample_pursuer_start(self.evader_pos, self.cfg)
        self.pursuer_pos = p_pos.astype(np.float32)
        self.pursuer_vel = p_vel.astype(np.float32)
        self.pursuer_force_dir = p_dir.astype(np.float32)
        self.pursuer_force_mag = 0.0
        # record baseline distances for shaping rewards
        self.prev_pe_dist = np.linalg.norm(self.evader_pos - self.pursuer_pos)
        target = np.array(self.cfg['target_position'], dtype=np.float32)
        self.prev_target_dist = np.linalg.norm(self.evader_pos - target)
        return self._get_obs(), {}

    def step(self, action: dict):
        evader_action = np.array(action['evader'], dtype=np.float32)
        pursuer_action = np.array(action['pursuer'], dtype=np.float32)
        self._update_agent('evader', evader_action)
        self._update_agent('pursuer', pursuer_action)
        # shaping rewards based on change in distances
        dist_pe = np.linalg.norm(self.evader_pos - self.pursuer_pos)
        target = np.array(self.cfg['target_position'], dtype=np.float32)
        dist_target = np.linalg.norm(self.evader_pos - target)
        shape_p = self.prev_pe_dist - dist_pe
        shape_e = self.prev_target_dist - dist_target
        self.prev_pe_dist = dist_pe
        self.prev_target_dist = dist_target

        done, r_e, r_p = self._check_done()
        r_e += self.shaping_weight * shape_e
        r_p += self.shaping_weight * shape_p
        obs = self._get_obs()
        reward = {'evader': r_e, 'pursuer': r_p}
        info = {}
        return obs, reward, done, False, info

    def _update_agent(self, name: str, action: np.ndarray):
        cfg_a = self.cfg[name]
        max_acc = cfg_a['max_acceleration']
        top_speed = cfg_a['top_speed']
        drag_c = cfg_a['drag_coefficient']
        turn_rate = cfg_a['turn_rate']
        stall = cfg_a['stall_angle']
        if name == 'evader':
            pos = self.evader_pos
            vel = self.evader_vel
            dir_vec = self.evader_force_dir
            gravity = np.array([0.0, 0.0, -self.cfg['gravity']], dtype=np.float32)
        else:
            pos = self.pursuer_pos
            vel = self.pursuer_vel
            dir_vec = self.pursuer_force_dir
            gravity = np.zeros(3, dtype=np.float32)

        mag = float(np.clip(action[0], 0.0, max_acc))
        theta = float(action[1])
        phi = float(np.clip(action[2], 0.0, stall))
        target_dir = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ], dtype=np.float32)
        target_dir /= np.linalg.norm(target_dir) + 1e-8

        angle_diff = np.arccos(np.clip(np.dot(dir_vec, target_dir), -1.0, 1.0))
        max_change = turn_rate * self.dt
        if angle_diff > max_change:
            ratio = max_change / (angle_diff + 1e-8)
            new_dir = dir_vec * (1 - ratio) + target_dir * ratio
            new_dir /= np.linalg.norm(new_dir) + 1e-8
        else:
            new_dir = target_dir

        if name == 'evader':
            self.evader_force_dir = new_dir
            self.evader_force_mag = mag
        else:
            self.pursuer_force_dir = new_dir
            self.pursuer_force_mag = mag

        acc_cmd = new_dir * mag
        drag = -drag_c * new_dir
        acc_total = acc_cmd + drag + gravity

        vel[:] = vel + acc_total * self.dt
        speed = np.linalg.norm(vel)
        if speed > top_speed:
            vel[:] = vel / speed * top_speed
        pos[:] = pos + vel * self.dt

    def _check_done(self):
        dist = np.linalg.norm(self.evader_pos - self.pursuer_pos)
        if dist <= self.cfg['capture_radius']:
            return True, -1.0, 1.0
        # check impact with target
        target = np.array(self.cfg['target_position'], dtype=np.float32)
        if (self.evader_pos[2] <= 0.0 and np.linalg.norm(self.evader_pos - target) < self.cfg['capture_radius'] * 5):
            return True, 1.0, -1.0
        return False, 0.0, 0.0

    def _get_obs(self):
        # pursuer observation: own pos/vel + evader pos
        obs_p = np.concatenate([self.pursuer_pos, self.pursuer_vel, self.evader_pos])
        # evader observation
        obs_elems = [self.evader_pos, self.evader_vel, self.cfg['target_position']]
        mode = self.cfg['evader'].get('awareness_mode', 1)
        if mode == 2:
            dist = np.linalg.norm(self.pursuer_pos - self.evader_pos)
            obs_elems.append([dist])
        elif mode == 3:
            direction = self.pursuer_pos - self.evader_pos
            norm = np.linalg.norm(direction) + 1e-8
            obs_elems.append(direction / norm)
        elif mode >= 4:
            obs_elems.append(self.pursuer_pos)
        obs_e = np.concatenate([np.asarray(x).ravel() for x in obs_elems])
        return {'pursuer': obs_p.astype(np.float32), 'evader': obs_e.astype(np.float32)}


def _make_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    """Utility to build a simple two-layer MLP."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )


class EvaderPolicy(nn.Module):
    """Generator network producing evader actions."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = _make_mlp(obs_dim, 3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class PursuerPolicy(nn.Module):
    """Adversary network producing pursuer actions."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = _make_mlp(obs_dim, 3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def sample_trajectories(policy_e: EvaderPolicy, policy_p: PursuerPolicy, env: PursuitEvasionEnv, num_episodes: int):
    """Placeholder for sampling trajectories using current policies."""
    trajs = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode = []
        while not done:
            with torch.no_grad():
                a_e = policy_e(torch.tensor(obs['evader'], dtype=torch.float32)).numpy()
                a_p = policy_p(torch.tensor(obs['pursuer'], dtype=torch.float32)).numpy()
            obs, reward, done, _, _ = env.step({'evader': a_e, 'pursuer': a_p})
            episode.append((obs, reward))
        trajs.append(episode)
    return trajs


def train_autogan(policy_e: EvaderPolicy, policy_p: PursuerPolicy, env: PursuitEvasionEnv, iterations: int = 1000):
    """Placeholder AutoGAN-style training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_e.to(device)
    policy_p.to(device)
    opt_e = optim.Adam(policy_e.parameters(), lr=1e-3)
    opt_p = optim.Adam(policy_p.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(iterations):
        trajs = sample_trajectories(policy_e, policy_p, env, num_episodes=1)
        # Placeholder discriminator loss (pursuer tries to classify success)
        labels = torch.ones(1, 1, device=device)
        output = torch.randn(1, 1, device=device)  # fake output
        d_loss = criterion(output, labels)
        opt_p.zero_grad()
        d_loss.backward()
        opt_p.step()

        # Placeholder generator loss (evader tries to fool the pursuer)
        g_loss = -d_loss
        opt_e.zero_grad()
        g_loss.backward()
        opt_e.step()


def main():
    """Run one episode with random actions to demonstrate the environment."""
    env = PursuitEvasionEnv(config)
    obs, _ = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 100:
        action_e = np.array([
            np.random.uniform(0.0, config['evader']['max_acceleration']),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(0.0, config['evader']['stall_angle']),
        ], dtype=np.float32)
        action_p = np.array([
            np.random.uniform(0.0, config['pursuer']['max_acceleration']),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(0.0, config['pursuer']['stall_angle']),
        ], dtype=np.float32)
        obs, reward, done, _, _ = env.step({'evader': action_e, 'pursuer': action_p})
        step_count += 1
    print(f"Episode finished after {step_count} steps. Reward: {reward}")


if __name__ == '__main__':
    main()
