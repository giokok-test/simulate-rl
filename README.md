# Simulate RL

This repository contains a small reinforcement learning setup for a simple 3D pursuit scenario. Entity **X** acts as an intruder starting in the air while entity **Y** is on the ground. A lightweight neural network policy controls the actions.

## Environment

`Simple3DEnv` located in `src/environment.py` is a custom `gym.Env` implementation with the following characteristics:

- **Observation Space** (`Box(9,)`):
  - X position `(x, y, z)`
  - Y position `(x, y, z)`
  - X velocity `(vx, vy, vz)` (includes gravity)
- **Action Space** (`Box(5,)`):
  - Movement for X `(dx, dy, dz)`
  - Movement for Y on the ground `(dx, dy)`
- **Gravity**: Each step, the intruder's vertical movement is reduced by `gravity` (default `-0.1`). The altitude is clamped at ground level.
- **Reward**: Negative Euclidean distance between X and Y.
- **Termination**: When distance < 1 m or after `max_steps` steps.

Running `env.render()` prints X and Y positions and the current velocity of X.

## Policy Network

`SimplePolicyNetwork` in `src/model.py` is a basic multilayer perceptron with two hidden layers of size 128 and a `tanh` output. It expects the 9-dimensional state and outputs 5 continuous actions.

## Training

`src/train.py` contains a minimal training loop using the policy network. The script repeatedly interacts with the environment and performs gradient ascent on the immediate reward.

```bash
python -m src.train
```

Training parameters such as the number of episodes can be adjusted by editing `train.py`.

## Installation

This project depends on:

- Python 3.8+
- `gym`
- `numpy`
- `torch`

Install them with pip:

```bash
pip install gym numpy torch
```

## Example

After installing the dependencies, run the following to train for a few episodes:

```bash
python -m src.train
```

You will see output similar to:

```text
Episode 0: reward -XXXX.XX
Episode 1: reward -XXXX.XX
...
```

## Notes

The current setup is deliberately simple. The reward and update rule are minimal examples for demonstration only and are not meant for high-performance RL training.
