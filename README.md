# Simulated Pursuit-Evasion RL

This repository contains a small demonstration of a 3D pursuit--evasion
environment written using `gymnasium`. Two agents (an evader and a pursuer)
move in a simplified physics world.  The current training script implements
Deep Q-learning with a replay buffer and TensorBoard logging.

## Environment setup

1. **Python**: Ensure Python 3.10+ is available. Creating a virtual
environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file lists the packages used in the example scripts.

The environment configuration is split across four YAML files:
``evader.yaml``, ``pursuer.yaml``, ``env.yaml`` and ``training.yaml``. These
files reside in the ``setup/`` directory and are loaded automatically by the
scripts.

## Running the first training

The `train_pursuer_qlearning.py` script trains a Q-network to control the
pursuer while the evader follows a scripted behaviour. To start training run:

```bash
python train_pursuer_qlearning.py --log-dir runs/dqn
```

Weights are saved to `pursuer_dqn.pt` by default. Command line arguments allow
overriding the episode count and log directory. All defaults live in
``setup/training.yaml`` and can be modified directly in that file.

Pass ``--log-dir`` to write TensorBoard metrics such as episode reward,
evaluation return, loss, exploration rate, minimum pursuer--evader distance,
start-distance ratio, action deltas, the sliding success rate over the most
recent training batch and the active curriculum parameter ranges. In addition,
the trainer prints a concise console summary after every
``q_learning.batch_size`` episodes showing the outcome counts together with the
mean reward and mean episode duration for that batch so you can spot drifts
without launching TensorBoard.

### Monitoring training with TensorBoard

When ``--log-dir`` is supplied the trainer writes episode reward, loss,
exploration rate, distance ratios, action deltas, active spawn ranges and the
success rate computed over the last ``q_learning.batch_size`` episodes for
visualisation with TensorBoard:

```bash
tensorboard --logdir runs
```

Periodic checkpoints are stored under ``<log-dir>/checkpoints`` when
``--checkpoint-every`` or ``q_learning.checkpoint_every`` is set.

### Q-learning training

The ``train_pursuer_qlearning.py`` script implements deep Q-learning with a
replay buffer and target network. Continuous observations feed a small MLP
which outputs action-values for the discrete manoeuvre set.

Run training with:

```bash
python train_pursuer_qlearning.py --config setup/training.yaml --log-dir runs/dqn
```

The trained weights are saved as ``pursuer_dqn.pt``. Enable TensorBoard logging
by setting ``q_learning.log_dir`` in ``setup/training.yaml`` or passing ``--log-dir``.

## Curriculum training

The training pipeline supports fixed and adaptive curricula that gradually
increase task difficulty. Parameters live in the ``curriculum`` section of
``setup/training.yaml`` and interpolate between ``start`` and ``end``
configuration blocks. In ``fixed`` mode progress advances linearly over a
number of discrete stages. ``adaptive`` mode only moves to the next stage when
the recent success rate exceeds ``success_threshold``.

Environment creation is centralised in :func:`initialize_gym`, which applies
the current curriculum state before instantiating ``PursuerOnlyEnv``. Both the
training script and ``play.py`` use this helper.

Enable curriculum training by setting ``curriculum.mode`` and specifying the
desired interpolation:

```yaml
curriculum:
  mode: adaptive          # or "fixed" for linear progression
  success_threshold: 0.6  # minimum capture rate to advance (adaptive only)
  window: 64              # episodes in success window
  stages: 120             # number of discrete curriculum stages
  start:
    evader_start:
      distance_range: [10000.0, 10000.0]
      initial_speed: 1.0
    pursuer_start:
      cone_half_angle: 1.5708
      inner_cone_half_angle: 1.5708
      min_range: 10.0
      max_range: 50.0
      yaw_range: [0.0, 0.0]
      initial_speed_range: [50.0, 50.0]
      force_target_radius: 0.0
  end:
    evader_start:
      distance_range: [8000.0, 12000.0]
      initial_speed: 50.0
    pursuer_start:
      cone_half_angle: 1.5
      inner_cone_half_angle: 1.3
      min_range: 1000.0
      max_range: 5000.0
      yaw_range: [-2.0, 2.0]
      initial_speed_range: [50.0, 75.0]
      force_target_radius: 350.0
```

When ``mode`` is set the trainer automatically widens the spawn range and
other parameters as the policy succeeds.

The ``play.py`` utility uses the same configuration and applies the curriculum
state before running a single episode. Pass ``--progress`` to visualise
intermediate stages.

### Q-learning theory

- Learns an action-value function $Q(s,a)$ estimating long-term returns via temporal-difference updates.
- Off-policy: updates use greedy targets while the behaviour policy explores.
- Converges to the optimal policy in tabular settings with sufficient exploration (Watkins & Dayan, 1992).

In this pursuit--evasion task the pursuer observes continuous state vectors and selects from a discrete manoeuvre set.
Q-learning evaluates how each manoeuvre moves the pursuer toward capturing the evader while avoiding penalties for crashes or excessive separation.
Because the state space is continuous we approximate $Q_\theta(s,a)$ with a neural network trained on minibatches sampled from a replay buffer.

```python
import random
from collections import deque

import gymnasium as gym
import torch
from torch import nn, Tensor


class QNet(nn.Module):
    """Small MLP approximating Q(s, a)."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def q_learning(env: gym.Env) -> None:
    """Minimal Deep Q-learning loop."""

    net = QNet(env.observation_space.shape[0], env.action_space.n)
    target = QNet(env.observation_space.shape[0], env.action_space.n)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    buf: deque = deque(maxlen=10_000)
    eps, gamma = 1.0, 0.99
    obs, _ = env.reset()
    for step in range(1000):
        if random.random() < eps:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                act = torch.argmax(net(torch.tensor(obs).float())).item()
        n_obs, reward, term, trunc, _ = env.step(act)
        buf.append((obs, act, reward, n_obs, term or trunc))
        obs = n_obs
        if len(buf) >= 32:
            batch = random.sample(buf, 32)
            o, a, r, n_o, d = map(lambda x: torch.tensor(x).float(), zip(*batch))
            q = net(o).gather(1, a.long().unsqueeze(1)).squeeze()
            with torch.no_grad():
                target_q = target(n_o).max(1).values
            y = r + gamma * target_q * (1 - d)
            loss = torch.nn.functional.mse_loss(q, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if step % 50 == 0:
            target.load_state_dict(net.state_dict())
        eps = max(0.05, eps * 0.995)

```

**Evaluation plan**

- Log episode return, minimum pursuer--evader distance, Q-value estimates and epsilon.
- Ablations: vary learning rate, replay capacity and exploration decay.
- Watch for divergence from overly large step sizes or insufficient exploration causing unstable Q-values.

**Next steps**

- Compare against policy-gradient baselines such as PPO or SAC.
- Investigate prioritised replay and Double/Dueling DQN variants.
- Read: Watkins & Dayan (1992); Mnih et al. (2015).

## Additional scripts

- `pursuit_evasion.py` contains the environment implementation along with a
  `main()` function that runs a single random episode. This can be executed via

```bash
python pursuit_evasion.py
```

which is useful for quickly checking that the environment works.

- `play.py` loads a saved Q-network (``.pt``) or legacy Q-table (``.npy``) and
  runs a single episode. Episodes run for the duration specified by
  `episode_duration` in ``setup/env.yaml`` unless `--steps`` overrides the maximum
  number of simulation steps.
  The plot now highlights the starting and final positions of both agents,
  marks the evader's goal position and draws arrows indicating the initial
  heading of both players. During the run a table prints the distance vectors
  between the players and the goal along with the current velocities for both
  agents. The table header now uses ASCII arrows (``->``) instead of Unicode
  symbols for broader compatibility and is formatted with ``str.format`` so the
  script also runs on older Python versions. The spawn volume for the pursuer is
  outlined with green lines
  only (surface fills were removed to keep the plot responsive).
  Use ``--profile`` to print how long inference, environment stepping and
  plotting take.
- `sweep.py` performs a simple hyperparameter sweep for Q-learning by varying
  the learning rate and exploration decay. Each configuration logs metrics to a
  separate TensorBoard directory.
- `plot_config.py` renders a stand-alone visualisation of the environment
  configuration showing an outline of the spawn volume. The accompanying
  `SpawnVolumeDemo.ipynb` notebook calls this script so you can interactively
  adjust ``setup/env.yaml`` and inspect the effect.

The environment stores several statistics for each episode. When an episode
finishes the ``info`` dictionary returned from ``env.step`` contains the
closest pursuer--evader distance, number of steps and outcome. The outcome can
be ``capture`` (pursuer success), ``evader_success`` (reaches the target in the
air), ``evader_ground`` or ``pursuer_ground`` when a crash occurs,
``separation_exceeded`` if the starting distance has grown beyond the
``separation_cutoff_factor`` multiple (incurring ``separation_penalty``), or ``timeout`` when the step limit is
reached. A capture is also triggered when the pursuer crosses through the
evader's capture sphere between steps, detected by intersecting the line
segment between consecutive pursuer positions with a radius of
``capture_radius`` around the evader. The evaluation helpers in the training
scripts print the average
minimum distance and episode length during periodic evaluations.
The logged ``min_start_ratio`` metric records how close the pursuer got
relative to where it spawned (minimum distance divided by the starting
separation).
The total change in the pursuer's action commands over an episode is
available via the ``*_delta`` metrics described above.

## Adjusting environment parameters

All physical constants and environment options are stored in the
``setup/evader.yaml``, ``setup/pursuer.yaml`` and ``setup/env.yaml`` files.
Simply edit these files to tweak values such as masses, maximum
acceleration or the starting distance of
the pursuer.  The evader's starting position is also randomised using
the `evader_start.distance_range` and `evader_start.altitude` settings,
while `evader_start.initial_speed` controls its initial velocity toward
the target (within ±15° of the exact bearing).

The pursuer's starting position is sampled inside a configurable volume.
`pursuer_start.cone_half_angle` sets the outer limit of the spawn cone
below the evader while `pursuer_start.inner_cone_half_angle` specifies an
inner cutoff to avoid very steep spawn angles.  Horizontal placement can be
controlled either by specifying explicit `sections` (front, right, back and
left quadrants) **or** with `pursuer_start.yaw_range`.  The yaw range is
measured relative to directly behind the evader (0&nbsp;rad points toward the
pursuer approaching from behind).  When `yaw_range` is present it overrides
the section based spawning.  Combined with the `min_range` and
`max_range` distances these options define where the pursuer may appear at
the beginning of an episode.
The initial speed of the pursuer is drawn uniformly from
`pursuer_start.initial_speed_range`, defining the range of possible spawn
velocities.
Both `pursuit_evasion.py` and `train_pursuer_qlearning.py` load the configuration
at runtime, so changes take effect the next time you run the script.
The reward shaping parameters `shaping_weight` and `align_weight` can be
adjusted here to encourage desired behaviour. The `separation_cutoff_factor`
option defines a multiplier of
the initial pursuer--evader distance that ends the episode when the
agents drift farther apart than this threshold. When this occurs the
pursuer receives `separation_penalty` as a terminal reward.
The `capture_bonus` setting adds a time incentive for the pursuer by
increasing its terminal reward when a capture occurs earlier. The final
reward becomes `1 + capture_bonus * (max_steps - episode_steps)` where
`max_steps` is the episode length limit. This bonus is awarded only when
the pursuer actually captures the evader.

The `evader.awareness_mode` option defines how much information the
evader receives about the pursuer:

1. `1` – no knowledge of the pursuer
2. `2` – only the distance to the pursuer
3. `3` – the unit vector pointing to the pursuer
4. `4` – full pursuer position (values above 4 behave the same)

The `yaw_rate` and `pitch_rate` values for both agents are specified in
degrees per second and are converted internally to radians per second.
Similarly, the `stall_angle` parameter in ``setup/evader.yaml`` and
``setup/pursuer.yaml`` is given in
degrees but converted to radians when the environment loads. Actions
specify yaw and **pitch** where pitch is measured relative to the horizontal
x–y plane (positive values command an upward climb). Both agents clamp
their pitch commands to ``[-stall_angle, +stall_angle]``. The first action
component controls the **magnitude** of the acceleration while the yaw and
pitch angles define its direction. Acceleration can slow the agent down but is
clamped so that it never reverses the current velocity. The
``evader.trajectory`` option selects a preset flight profile. When set to
``"dive"`` the evader keeps a non-negative pitch until the line of sight to the
target drops below ``evader.dive_angle`` (specified in degrees). Once past this
threshold the evader begins descending toward the goal. The
`episode_duration` value defines how long each episode lasts in minutes and
is used to compute the maximum number of simulation steps based on the
configured `time_step`.

When either agent touches the ground the episode terminates. If the evader
hits the ground its terminal reward scales with the distance to the target
using ``target_reward_distance``. A reward of one is given when it reaches the
goal and it falls off to zero once the evader is roughly 100&nbsp;m away by
default. Episodes also end successfully when the airborne evader comes within
100&nbsp;m of the goal. The radius for this check can be adjusted via the
``target_success_distance`` setting. The pursuer receives a configurable
penalty, controlled by ``pursuer_ground_penalty``, when it crashes into the
ground.

## Sensor error model

The configuration file defines a `measurement_error_pct` option controlling  
the uncertainty in angular measurements of the opposing agent. When the value  
is greater than zero each agent observes the other via noisy right ascension  
($\alpha$) and declination ($\delta$) angles. The noise is modelled as a  
percentage of the measured angles, where

![sigma](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%20%5Csigma%20%3D%20%5Cfrac%7B%5Cmathrm%7Bmeasurement_e%7D%7D%7B100%7D)

and the perturbed angles satisfy

![alpha delta](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%20%5Calpha'%20%3D%20%5Calpha%20%2B%20%5Calpha%20%5Csigma%20%5Cvarepsilon_%5Calpha%2C%5Cquad%20%5Cdelta'%20%3D%20%5Cdelta%20%2B%20%5Cdelta%20%5Csigma%20%5Cvarepsilon_%5Cdelta)

with $\varepsilon_\alpha,\;\varepsilon_\delta\sim\mathcal{N}(0,1)$.

These are converted into a unit direction vector:

![u prime](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%20u'%20%3D%20%5B%5Ccos%5Cdelta'%5Ccos%5Calpha'%20%5C;%20%5Ccos%5Cdelta'%5Csin%5Calpha'%20%5C;%20%5Csin%5Cdelta'%5D)

If the true range to the target is $R$, the observed position becomes

![position](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%20p%20%2B%20R%20u')

Linearising around the true angles yields the first‐order position‐error

![delta r](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%20%5CDelta%20r%20%5Capprox%20R%28%5CDelta%5Calpha%5Cpartial%20u%2F%5Cpartial%5Calpha%20%2B%20%5CDelta%5Cdelta%5Cpartial%20u%2F%5Cpartial%5Cdelta%29)

where

![partial alpha](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%5Calpha%7D%20%3D%20%5B-%5Ccos%5Cdelta%5Csin%5Calpha%5C;%20%5Ccos%5Cdelta%5Ccos%5Calpha%5C;%200%5D)

![partial delta](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%5Cdelta%7D%20%3D%20%5B-%5Csin%5Cdelta%5Ccos%5Calpha%5C;-%5Csin%5Cdelta%5Csin%5Calpha%5C;%20%5Ccos%5Cdelta%5D)

Velocity is then estimated from successive noisy positions, so the velocity  
error is simply the difference between these position‐errors divided by the  
simulation time‐step.
