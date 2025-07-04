import gym
from gym import spaces
import numpy as np

class Simple3DEnv(gym.Env):
    """A simple 3D environment with two entities X and Y."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps: int = 100, gravity: float = -0.1):
        super().__init__()
        self.max_steps = max_steps
        # Vertical velocity applied each step to mimic gravity pulling X downward
        self.gravity = gravity
        self.step_count = 0

        # Observation: X position (x, y, z), Y position (x, y, z),
        # and the intruder (X) velocity (vx, vy, vz)
        high = np.array(
            [10000.0, 10000.0, 3000.0, 10000.0, 10000.0, 0.0, 10.0, 10.0, 10.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Actions: delta movement for X in 3D, Y moves on ground (2D)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.state = None

    def reset(self):
        x_pos = np.random.uniform(-1000, 1000, size=2)
        x_alt = 3000.0
        y_pos = np.random.uniform(-1000, 1000, size=2)
        self.state = np.array(
            [
                x_pos[0],
                x_pos[1],
                x_alt,
                y_pos[0],
                y_pos[1],
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        self.step_count = 0
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        x_move = action[:3]
        y_move = np.append(action[3:], 0.0)

        # Apply gravity to the intruder's vertical movement
        x_move_with_gravity = np.array(x_move)
        x_move_with_gravity[2] += self.gravity

        # Update positions with gravity adjusted movement
        self.state[:3] += x_move_with_gravity
        self.state[3:6] += y_move

        # Prevent the intruder from going below ground level
        self.state[2] = max(self.state[2], 0.0)

        # Record intruder velocity
        self.state[6:9] = x_move_with_gravity

        self.step_count += 1
        dist = np.linalg.norm(self.state[:3] - self.state[3:6])

        reward = -dist
        done = dist < 1.0 or self.step_count >= self.max_steps
        info = {"distance": dist}
        return self.state, reward, done, info

    def render(self, mode="human"):
        print(
            f"X: {self.state[:3]} Y: {self.state[3:6]} X_vel: {self.state[6:9]}"
        )

    def close(self):
        pass
