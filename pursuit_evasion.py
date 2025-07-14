import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import yaml
import os
import copy


def load_config(path: str | None = None) -> dict:
    """Load configuration parameters.

    When ``path`` is ``None`` the function reads ``evader.yaml``,
    ``pursuer.yaml``, ``env.yaml`` and ``training.yaml`` located next to this
    file and merges them into a single dictionary.  If ``path`` points to a
    directory the same file names are loaded from that directory.  Supplying a
    path to a specific YAML file preserves the original behaviour and returns
    its contents directly.
    """

    if path is None:
        base = os.path.dirname(__file__)
    elif os.path.isdir(path):
        base = path
    else:
        with open(path, "r") as fh:
            return yaml.safe_load(fh)

    cfg: dict = {}
    for name in ("evader.yaml", "pursuer.yaml", "env.yaml", "training.yaml"):
        fp = os.path.join(base, name)
        if os.path.exists(fp):
            with open(fp, "r") as fh:
                part = yaml.safe_load(fh) or {}
            cfg.update(part)
    return cfg


# Global configuration dictionary for the environment and agents
config = load_config()


def _interpolate(start: float, end: float, progress: float) -> float:
    """Interpolate between ``start`` and ``end`` logarithmically.

    When both values are positive the interpolation operates on the natural
    logarithm of the numbers which results in smaller changes early on and
    larger steps once ``progress`` approaches one. This is particularly useful
    when the bounds span several orders of magnitude. Negative values or pairs
    that cross zero fall back to simple linear interpolation.
    """

    if start > 0 and end > 0:
        start_l = np.log(float(start))
        end_l = np.log(float(end))
        return float(np.exp(start_l + (end_l - start_l) * progress))
    return start + (end - start) * progress


def apply_curriculum(cfg: dict, start_cfg: dict, end_cfg: dict, progress: float) -> None:
    """Recursively interpolate ``cfg`` between ``start_cfg`` and ``end_cfg``.

    Only keys present in ``start_cfg`` and ``end_cfg`` are touched. Numeric
    values are interpolated logarithmically using :func:`_interpolate` while
    boolean values switch from the start to the end setting halfway through the
    training run.
    """

    for key, start_val in start_cfg.items():
        if key not in end_cfg or key not in cfg:
            continue
        end_val = end_cfg[key]
        if isinstance(start_val, dict) and isinstance(end_val, dict):
            apply_curriculum(cfg[key], start_val, end_val, progress)
        elif isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
            cfg[key] = _interpolate(float(start_val), float(end_val), progress)
        elif (
            isinstance(start_val, (list, tuple))
            and isinstance(end_val, (list, tuple))
            and len(start_val) == len(end_val)
        ):
            cfg[key] = [
                _interpolate(float(s), float(e), progress)
                for s, e in zip(start_val, end_val)
            ]
        elif isinstance(start_val, bool) and isinstance(end_val, bool):
            cfg[key] = end_val if progress >= 0.5 else start_val


def sample_pursuer_start(evader_pos: np.ndarray, heading: np.ndarray, cfg: dict):
    """Sample initial pursuer state and force direction.

    The pursuer spawns inside a truncated cone around the evader. ``heading``
    defines the reference direction for the four 90° spawn quadrants.
    """

    params = cfg["pursuer_start"]
    outer = params["cone_half_angle"]
    inner = params.get("inner_cone_half_angle", 0.0)
    r = np.random.uniform(params["min_range"], params["max_range"])

    base_yaw = np.arctan2(heading[1], heading[0])
    if "yaw_range" in params:
        # Angular range is measured relative to directly behind the evader.
        yaw_min, yaw_max = params["yaw_range"]
        yaw_rel = np.random.uniform(yaw_min, yaw_max)
        yaw = (base_yaw + np.pi + yaw_rel) % (2 * np.pi)
    else:
        sections_cfg = params.get(
            "sections",
            {"front": True, "left": True, "right": True, "back": True},
        )
        quadrants = []
        deg45 = np.deg2rad(45.0)
        if sections_cfg.get("front", True):
            quadrants.append((-deg45, deg45))
        if sections_cfg.get("right", True):
            quadrants.append((-deg45 - np.pi / 2, deg45 - np.pi / 2))
        if sections_cfg.get("back", True):
            quadrants.append((np.pi - deg45, np.pi + deg45))
        if sections_cfg.get("left", True):
            quadrants.append((deg45, deg45 + np.pi / 2))
        if not quadrants:
            raise ValueError("No pursuer spawn sections enabled")
        yaw_rel = np.random.uniform(*quadrants[np.random.randint(len(quadrants))])
        yaw = (base_yaw + yaw_rel) % (2 * np.pi)

    pitch = np.random.uniform(inner, outer)
    # direction from the evader to the pursuer in world coordinates
    dir_vec = np.array([
        np.sin(pitch) * np.cos(yaw),
        np.sin(pitch) * np.sin(yaw),
        -np.cos(pitch),
    ], dtype=np.float32)
    pos = evader_pos + dir_vec * r

    # point the initial force vector toward a random point near the evader
    tgt_offset = np.random.randn(3).astype(np.float32)
    tgt_offset /= np.linalg.norm(tgt_offset) + 1e-8
    tgt_offset *= np.random.uniform(0.0, params['force_target_radius'])
    tgt = evader_pos + tgt_offset
    to_tgt = tgt - pos
    to_tgt /= np.linalg.norm(to_tgt) + 1e-8

    speed = np.random.uniform(*params['initial_speed_range'])
    # initial velocity aligned with the chosen target direction
    vel = to_tgt * speed
    return pos, vel, to_tgt


class PursuitEvasionEnv(gym.Env):
    """Gym-compatible 3D pursuit-evasion environment.

    The env simulates two aircraft-like agents. The evader attempts to reach
    a target position while the pursuer tries to intercept it. All physics is
    very simplified but captures the main constraints such as turn rate,
    maximum acceleration and drag. Observations and actions are represented as
    simple ``numpy`` arrays so the environment can be easily used with
    standard RL libraries.

    Episodes terminate if either agent's altitude drops below zero. When the
    evader hits the ground its terminal reward is scaled by the distance to the
    goal using ``target_reward_distance`` from the configuration.
    """

    def __init__(self, cfg: dict):
        """Create the environment.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary defining the physical parameters for the
            evader and pursuer as well as some global environment options.
        """

        super().__init__()
        # Make a copy so we can modify units without affecting the caller
        self.cfg = copy.deepcopy(cfg)
        self.dt = self.cfg['time_step']
        self.shaping_weight = self.cfg.get('shaping_weight', 0.05)
        # Additional shaping when the pursuer decreases its distance to the
        # evader between consecutive steps. This complements the basic shaping
        # reward above which encourages closing the gap proportionally.
        self.closer_weight = self.cfg.get('closer_weight', 0.0)
        # Reward for reducing the angle between the pursuer's force direction
        # and the line of sight to the evader from one step to the next.
        self.angle_weight = self.cfg.get('angle_weight', 0.0)
        # Reward for reducing the difference between the pursuer and
        # evader headings from one step to the next.
        self.heading_weight = self.cfg.get('heading_weight', 0.0)
        self.meas_err = self.cfg.get('measurement_error_pct', 0.0) / 100.0
        # maximum allowed separation before the episode ends
        self.cutoff_factor = self.cfg.get('separation_cutoff_factor', 2.0)
        # Convert stall angles provided in degrees to radians once
        self.cfg['evader']['stall_angle'] = np.deg2rad(
            self.cfg['evader']['stall_angle']
        )
        if 'dive_angle' in self.cfg['evader']:
            self.cfg['evader']['dive_angle'] = np.deg2rad(
                self.cfg['evader']['dive_angle']
            )
        self.cfg['pursuer']['stall_angle'] = np.deg2rad(
            self.cfg['pursuer']['stall_angle']
        )
        # observation sizes depend on awareness mode
        self.evader_obs_dim = 9
        mode = cfg['evader'].get('awareness_mode', 1)
        if mode == 2:
            self.evader_obs_dim += 1  # distance to pursuer
        elif mode == 3:
            self.evader_obs_dim += 3  # directional unit vector
        elif mode >= 4:
            self.evader_obs_dim += 3  # pursuer position
        # pursuer observes evader position and explicit direction
        self.pursuer_obs_dim = 12

        self.observation_space = gym.spaces.Dict({
            'pursuer': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.pursuer_obs_dim,), dtype=np.float32),
            'evader': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.evader_obs_dim,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Dict({
            # actions: [acceleration magnitude, azimuth, pitch]
            # Acceleration is a non-negative scalar with direction determined by
            # the commanded yaw and pitch angles.
            'pursuer': gym.spaces.Box(
                low=np.array([
                    0.0,
                    -np.pi,
                    -cfg['pursuer']['stall_angle'],
                ], dtype=np.float32),
                high=np.array([
                    cfg['pursuer']['max_acceleration'],
                    np.pi,
                    cfg['pursuer']['stall_angle'],
                ], dtype=np.float32),
            ),
            'evader': gym.spaces.Box(
                low=np.array([0.0, -np.pi, -cfg['evader']['stall_angle']], dtype=np.float32),
                high=np.array([
                    cfg['evader']['max_acceleration'],
                    np.pi,
                    cfg['evader']['stall_angle'],
                ], dtype=np.float32),
            ),
        })
        self.reset()

    def reset(self, *, seed=None, options=None):
        """Reset the environment state.

        Returns the initial observation dictionary for both agents. The
        pursuer spawn position is sampled randomly below the evader using
        :func:`sample_pursuer_start`.
        """

        super().reset(seed=seed)
        start_cfg = self.cfg.get('evader_start', {})
        dmin, dmax = start_cfg.get('distance_range', [0.0, 0.0])
        altitude = start_cfg.get('altitude', 3000.0)
        target = np.array(self.cfg['target_position'], dtype=np.float32)
        dist = np.random.uniform(dmin, dmax)
        ang = np.random.uniform(0.0, 2 * np.pi)
        start_xy = target[:2] + dist * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        self.evader_pos = np.array([start_xy[0], start_xy[1], altitude], dtype=np.float32)

        # Initial velocity roughly toward the target in the x-y plane
        dir_xy = target[:2] - start_xy
        dir_xy /= np.linalg.norm(dir_xy) + 1e-8
        # rotate within ±15 degrees to introduce some randomness
        max_off = np.deg2rad(15.0)
        rot_ang = np.random.uniform(-max_off, max_off)
        rot_mat = np.array([
            [np.cos(rot_ang), -np.sin(rot_ang)],
            [np.sin(rot_ang), np.cos(rot_ang)],
        ], dtype=np.float32)
        heading = rot_mat @ dir_xy
        speed = start_cfg.get('initial_speed', 0.0)
        self.evader_vel = np.array([heading[0], heading[1], 0.0], dtype=np.float32) * speed

        dir_vec = np.array([dir_xy[0], dir_xy[1], 0.0], dtype=np.float32)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-8
        self.evader_force_dir = dir_vec
        self.evader_force_mag = 0.0

        p_pos, p_vel, p_dir = sample_pursuer_start(self.evader_pos, heading, self.cfg)
        self.pursuer_pos = p_pos.astype(np.float32)
        self.pursuer_vel = p_vel.astype(np.float32)
        self.pursuer_force_dir = p_dir.astype(np.float32)
        self.pursuer_force_mag = 0.0
        # store initial yaw/pitch of both agents for logging
        self.init_pursuer_yaw = float(np.arctan2(self.pursuer_force_dir[1], self.pursuer_force_dir[0]))
        self.init_pursuer_pitch = float(
            np.arctan2(self.pursuer_force_dir[2], np.linalg.norm(self.pursuer_force_dir[:2]))
        )
        self.init_evader_yaw = float(np.arctan2(self.evader_force_dir[1], self.evader_force_dir[0]))
        self.init_evader_pitch = float(
            np.arctan2(self.evader_force_dir[2], np.linalg.norm(self.evader_force_dir[:2]))
        )
        # record baseline distances for shaping rewards
        self.prev_pe_dist = np.linalg.norm(self.evader_pos - self.pursuer_pos)
        vec_pe = self.evader_pos - self.pursuer_pos
        self.prev_pe_angle = self._angle_between(self.pursuer_force_dir, vec_pe)
        # heading difference between the agents for shaping
        h_e = self.evader_vel / (np.linalg.norm(self.evader_vel) + 1e-8)
        h_p = self.pursuer_vel / (np.linalg.norm(self.pursuer_vel) + 1e-8)
        self.prev_heading_angle = self._angle_between(h_p, h_e)
        # store the starting distance for logging
        self.start_pe_dist = self.prev_pe_dist
        target = np.array(self.cfg['target_position'], dtype=np.float32)
        self.prev_target_dist = np.linalg.norm(self.evader_pos - target)
        # reset reward component totals
        self._reward_breakdown = {
            'terminal': 0.0,
            'shaping': 0.0,
            'closer': 0.0,
            'angle': 0.0,
            'heading': 0.0,
            'time': 0.0,
        }
        # metrics for episode statistics
        self.min_pe_dist = self.prev_pe_dist
        self.cur_step = 0
        self._prev_ev_obs = None
        self._prev_pu_obs = None
        self.prev_pursuer_action = None
        self.pursuer_acc_delta = 0.0
        self.pursuer_yaw_delta = 0.0
        self.pursuer_pitch_delta = 0.0
        # store previous positions to detect capture between steps
        self.prev_pursuer_pos = self.pursuer_pos.copy()
        self.prev_evader_pos = self.evader_pos.copy()
        return self._get_obs(), {}

    def step(self, action: dict):
        """Update the environment one time step.

        A small time penalty of ``-0.001`` is applied to the pursuer's
        reward on every step to encourage faster capture.

        Parameters
        ----------
        action : dict
            Dictionary with ``'evader'`` and ``'pursuer'`` keys mapping to
            their respective action arrays.
        """

        evader_action = np.array(action['evader'], dtype=np.float32)
        pursuer_action = np.array(action['pursuer'], dtype=np.float32)
        if self.prev_pursuer_action is not None:
            diff = pursuer_action - self.prev_pursuer_action
            self.pursuer_acc_delta += abs(diff[0])
            self.pursuer_yaw_delta += abs(np.arctan2(np.sin(diff[1]), np.cos(diff[1])))
            self.pursuer_pitch_delta += abs(diff[2])
        self.prev_pursuer_action = pursuer_action
        prev_p_pos = self.pursuer_pos.copy()
        prev_e_pos = self.evader_pos.copy()
        self._update_agent('evader', evader_action)
        self._update_agent('pursuer', pursuer_action)
        # shaping rewards based on change in distances
        dist_pe = np.linalg.norm(self.evader_pos - self.pursuer_pos)
        target = np.array(self.cfg['target_position'], dtype=np.float32)
        dist_target = np.linalg.norm(self.evader_pos - target)
        dist_target_xy = np.linalg.norm((self.evader_pos - target)[:2])
        success_thresh = self.cfg.get('target_success_distance', 100.0)
        self.min_pe_dist = min(self.min_pe_dist, dist_pe)
        shape_p = self.prev_pe_dist - dist_pe
        shape_e = self.prev_target_dist - dist_target
        # Bonus reward when the pursuer reduces the distance compared to the
        # previous step. This explicitly incentivises consistent progress
        # toward the evader regardless of terminal rewards.
        closer_bonus = 0.0
        if self.closer_weight > 0.0 and dist_pe < self.prev_pe_dist:
            closer_bonus = self.closer_weight * (self.prev_pe_dist - dist_pe)
        # Reward when the pursuer aligns its force direction with the line of
        # sight to the evader compared to the previous step.
        vec_pe = self.evader_pos - self.pursuer_pos
        angle = self._angle_between(self.pursuer_force_dir, vec_pe)
        angle_bonus = 0.0
        if self.angle_weight > 0.0 and angle < self.prev_pe_angle:
            angle_bonus = self.angle_weight * (self.prev_pe_angle - angle)
        self.prev_pe_angle = angle
        # Reward for aligning the pursuer and evader headings
        h_e = self.evader_vel / (np.linalg.norm(self.evader_vel) + 1e-8)
        h_p = self.pursuer_vel / (np.linalg.norm(self.pursuer_vel) + 1e-8)
        head_ang = self._angle_between(h_p, h_e)
        heading_bonus = 0.0
        if self.heading_weight > 0.0 and head_ang < self.prev_heading_angle:
            heading_bonus = self.heading_weight * (
                self.prev_heading_angle - head_ang
            )
        self.prev_heading_angle = head_ang
        self.prev_pe_dist = dist_pe
        self.prev_target_dist = dist_target

        done, r_e, r_p_terminal, outcome = self._check_done(prev_p_pos, prev_e_pos)
        shaping_reward = self.shaping_weight * shape_p
        r_e += self.shaping_weight * shape_e
        r_p = (
            r_p_terminal
            + shaping_reward
            + closer_bonus
            + angle_bonus
            + heading_bonus
            - 0.001
        )
        self._reward_breakdown['terminal'] += r_p_terminal
        self._reward_breakdown['shaping'] += shaping_reward
        self._reward_breakdown['closer'] += closer_bonus
        self._reward_breakdown['angle'] += angle_bonus
        self._reward_breakdown['heading'] += heading_bonus
        self._reward_breakdown['time'] += -0.001
        obs = self._get_obs()
        reward = {'evader': r_e, 'pursuer': r_p}
        info = {}
        if done:
            info = {
                'episode_steps': self.cur_step + 1,
                'min_distance': float(self.min_pe_dist),
                'final_distance': float(dist_pe),
                'evader_to_target': float(dist_target),
                'start_distance': float(self.start_pe_dist),
            }
            if outcome is None:
                if dist_pe <= self.cfg['capture_radius']:
                    outcome = 'capture'
                elif dist_target_xy <= success_thresh and self.evader_pos[2] > 0.0:
                    outcome = 'evader_success'
                elif self.evader_pos[2] < 0.0:
                    outcome = 'evader_ground'
                elif self.pursuer_pos[2] < 0.0:
                    outcome = 'pursuer_ground'
                elif dist_pe >= self.cutoff_factor * self.start_pe_dist:
                    outcome = 'separation_exceeded'
            if outcome:
                info['outcome'] = outcome
            info['reward_breakdown'] = {
                k: float(v) for k, v in self._reward_breakdown.items()
            }
            info['pursuer_acc_delta'] = float(self.pursuer_acc_delta)
            info['pursuer_yaw_delta'] = float(self.pursuer_yaw_delta)
            info['pursuer_pitch_delta'] = float(self.pursuer_pitch_delta)
            # difference between starting and final orientation of both agents
            p_yaw = np.arctan2(self.pursuer_force_dir[1], self.pursuer_force_dir[0])
            p_pitch = np.arctan2(
                self.pursuer_force_dir[2], np.linalg.norm(self.pursuer_force_dir[:2])
            )
            yaw_diff = np.arctan2(np.sin(p_yaw - self.init_pursuer_yaw), np.cos(p_yaw - self.init_pursuer_yaw))
            pitch_diff = p_pitch - self.init_pursuer_pitch
            info['pursuer_yaw_diff'] = float(yaw_diff)
            info['pursuer_pitch_diff'] = float(pitch_diff)
            e_yaw = np.arctan2(self.evader_force_dir[1], self.evader_force_dir[0])
            e_pitch = np.arctan2(
                self.evader_force_dir[2], np.linalg.norm(self.evader_force_dir[:2])
            )
            info['evader_yaw_diff'] = float(
                np.arctan2(np.sin(e_yaw - self.init_evader_yaw), np.cos(e_yaw - self.init_evader_yaw))
            )
            info['evader_pitch_diff'] = float(e_pitch - self.init_evader_pitch)
        # update stored previous positions for next step
        self.prev_pursuer_pos = prev_p_pos
        self.prev_evader_pos = prev_e_pos

        self.cur_step += 1
        return obs, reward, done, False, info

    def _update_agent(self, name: str, action: np.ndarray):
        """Apply an action to either agent and integrate simple physics."""

        cfg_a = self.cfg[name]
        max_acc = cfg_a['max_acceleration']
        top_speed = cfg_a['top_speed']
        drag_c = cfg_a['drag_coefficient']
        yaw_rate = np.deg2rad(cfg_a.get('yaw_rate', cfg_a.get('turn_rate', 0.0)))
        pitch_rate = np.deg2rad(cfg_a.get('pitch_rate', cfg_a.get('turn_rate', 0.0)))
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

        # Acceleration magnitude is a non-negative scalar. The commanded yaw and
        # pitch angles define the direction of the applied force.
        mag = float(np.clip(action[0], 0.0, max_acc))
        theta = float(action[1])
        phi = float(np.clip(action[2], -stall, stall))
        target_dir = np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
        ], dtype=np.float32)
        target_dir /= np.linalg.norm(target_dir) + 1e-8

        # Rotate current force direction toward the commanded yaw/pitch angles
        # separately, respecting independent turn rates
        cur_yaw = np.arctan2(dir_vec[1], dir_vec[0])
        cur_pitch = np.arctan2(dir_vec[2], np.linalg.norm(dir_vec[:2]))
        yaw_diff = np.arctan2(np.sin(theta - cur_yaw), np.cos(theta - cur_yaw))
        pitch_diff = phi - cur_pitch
        max_yaw = yaw_rate * self.dt
        max_pitch = pitch_rate * self.dt
        new_yaw = cur_yaw + np.clip(yaw_diff, -max_yaw, max_yaw)
        new_pitch = cur_pitch + np.clip(pitch_diff, -max_pitch, max_pitch)
        new_pitch = np.clip(new_pitch, -stall, stall)
        new_dir = np.array([
            np.cos(new_pitch) * np.cos(new_yaw),
            np.cos(new_pitch) * np.sin(new_yaw),
            np.sin(new_pitch),
        ], dtype=np.float32)
        new_dir /= np.linalg.norm(new_dir) + 1e-8

        if name == 'evader':
            self.evader_force_dir = new_dir
            self.evader_force_mag = mag
        else:
            self.pursuer_force_dir = new_dir
            self.pursuer_force_mag = mag

        # Compute acceleration in world frame. Drag opposes the current
        # velocity while gravity only affects the evader.
        acc_cmd = new_dir * mag
        drag = -drag_c * vel
        acc_total = acc_cmd + drag + gravity

        # Integrate velocity while preventing a reversal of direction. When the
        # commanded acceleration would overshoot and invert the velocity vector
        # along its original direction, clamp the component in that direction to
        # zero instead of letting the agent move backwards.
        vel_dir = vel / (np.linalg.norm(vel) + 1e-8)
        vel_new = vel + acc_total * self.dt
        if np.dot(vel_new, vel_dir) < 0.0:
            # remove the component opposing the previous velocity
            perp = vel_new - np.dot(vel_new, vel_dir) * vel_dir
            vel_new = perp

        speed = np.linalg.norm(vel_new)
        if speed > top_speed:
            vel_new = vel_new / speed * top_speed
        vel[:] = vel_new
        pos[:] = pos + vel * self.dt

    def _angle_between(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return angle between two vectors in radians."""
        a_n = np.linalg.norm(a) + 1e-8
        b_n = np.linalg.norm(b) + 1e-8
        cos_t = np.clip(np.dot(a, b) / (a_n * b_n), -1.0, 1.0)
        return float(np.arccos(cos_t))

    def _observe_enemy(self, observer_pos: np.ndarray, enemy_pos: np.ndarray,
                        prev_obs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Return noisy observation of the enemy position and velocity."""

        vec = enemy_pos - observer_pos
        r = np.linalg.norm(vec) + 1e-8
        ra = np.arctan2(vec[1], vec[0])
        dec = np.arcsin(vec[2] / r)
        if self.meas_err > 0.0:
            ra += ra * self.meas_err * np.random.randn()
            dec += dec * self.meas_err * np.random.randn()
        direction = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ], dtype=np.float32)
        pos_obs = observer_pos + direction * r
        if prev_obs is None:
            vel_obs = enemy_pos - enemy_pos  # zeros
        else:
            vel_obs = (pos_obs - prev_obs) / self.dt
        return pos_obs.astype(np.float32), vel_obs.astype(np.float32)

    def _check_done(
        self,
        prev_p_pos: np.ndarray | None = None,
        prev_e_pos: np.ndarray | None = None,
    ):
        """Determine if the episode has terminated.

        Episodes end on capture, excessive separation, either agent touching
        the ground or the evader reaching the vicinity of its target while
        still airborne.
        """
        dist = np.linalg.norm(self.evader_pos - self.pursuer_pos)
        target = np.array(self.cfg['target_position'], dtype=np.float32)
        dist_target = np.linalg.norm(self.evader_pos - target)
        dist_target_xy = np.linalg.norm((self.evader_pos - target)[:2])
        success_thresh = self.cfg.get('target_success_distance', 100.0)

        cross_capture = False
        if prev_p_pos is not None and prev_e_pos is not None:
            prev_vec = prev_e_pos - prev_p_pos
            cur_vec = self.evader_pos - self.pursuer_pos
            if np.dot(prev_vec, cur_vec) < 0.0:
                rel_start = prev_p_pos - prev_e_pos
                rel_v = (
                    (self.pursuer_pos - prev_p_pos)
                    - (self.evader_pos - prev_e_pos)
                )
                v_norm_sq = np.dot(rel_v, rel_v)
                if v_norm_sq > 1e-12:
                    t = -np.dot(rel_start, rel_v) / v_norm_sq
                    if 0.0 <= t <= 1.0:
                        closest = rel_start + rel_v * t
                        if np.linalg.norm(closest) <= self.cfg['capture_radius']:
                            cross_capture = True
                elif dist <= self.cfg['capture_radius']:
                    cross_capture = True

        if dist <= self.cfg['capture_radius'] or cross_capture:
            return True, -1.0, 1.0, 'capture'
        if dist_target_xy <= success_thresh and self.evader_pos[2] > 0.0:
            return True, 1.0, 0.0, 'evader_success'
        if dist >= self.cutoff_factor * self.start_pe_dist:
            return True, 0.0, 0.0, 'separation_exceeded'

        # episode ends if either agent goes below ground level
        if self.pursuer_pos[2] < 0.0:
            penalty = self.cfg.get('pursuer_ground_penalty', -1.0)
            return True, 0.0, penalty, 'pursuer_ground'
        if self.evader_pos[2] < 0.0:
            max_d = self.cfg.get('target_reward_distance', 100.0)
            reward = max(0.0, 1.0 - (dist_target / max_d) ** 2)
            return True, reward, 0.0, 'evader_ground'

        return False, 0.0, 0.0, None

    def _get_obs(self):
        """Assemble observations for both agents."""
        # optionally apply sensor error when observing the opposing agent
        if self.meas_err > 0.0:
            ev_pos_obs, ev_vel_obs = self._observe_enemy(
                self.pursuer_pos, self.evader_pos, getattr(self, "_prev_ev_obs", None)
            )
            self._prev_ev_obs = ev_pos_obs
            pu_pos_obs, pu_vel_obs = self._observe_enemy(
                self.evader_pos, self.pursuer_pos, getattr(self, "_prev_pu_obs", None)
            )
            self._prev_pu_obs = pu_pos_obs
        else:
            ev_pos_obs, ev_vel_obs = self.evader_pos, self.evader_vel
            pu_pos_obs, pu_vel_obs = self.pursuer_pos, self.pursuer_vel

        # pursuer observation: own state, evader position and normalized direction
        direction_pe = ev_pos_obs - self.pursuer_pos
        norm_pe = np.linalg.norm(direction_pe) + 1e-8
        obs_p = np.concatenate([
            self.pursuer_pos,
            self.pursuer_vel,
            ev_pos_obs,
            direction_pe / norm_pe,
        ])

        # evader observation starts with its own state and the target
        obs_elems = [self.evader_pos, self.evader_vel, self.cfg['target_position']]
        mode = self.cfg['evader'].get('awareness_mode', 1)
        if mode == 2:
            dist = np.linalg.norm(pu_pos_obs - self.evader_pos)
            obs_elems.append([dist])
        elif mode == 3:
            direction = pu_pos_obs - self.evader_pos
            norm = np.linalg.norm(direction) + 1e-8
            obs_elems.append(direction / norm)
        elif mode >= 4:
            obs_elems.append(pu_pos_obs)
        obs_e = np.concatenate([np.asarray(x).ravel() for x in obs_elems])
        return {'pursuer': obs_p.astype(np.float32), 'evader': obs_e.astype(np.float32)}


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_size: int = 64,
    activation: str = "relu",
) -> nn.Sequential:
    """Utility to build a simple two-layer MLP with configurable width."""

    acts = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    act_cls = acts.get(activation, nn.ReLU)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        act_cls(),
        nn.Linear(hidden_size, hidden_size),
        act_cls(),
        nn.Linear(hidden_size, output_dim),
    )




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
            np.random.uniform(-config['evader']['stall_angle'], config['evader']['stall_angle']),
        ], dtype=np.float32)
        action_p = np.array([
            np.random.uniform(0.0, config['pursuer']['max_acceleration']),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-config['pursuer']['stall_angle'], config['pursuer']['stall_angle']),
        ], dtype=np.float32)
        obs, reward, done, _, _ = env.step({'evader': action_e, 'pursuer': action_p})
        step_count += 1
    print(f"Episode finished after {step_count} steps. Reward: {reward}")


if __name__ == '__main__':
    main()
