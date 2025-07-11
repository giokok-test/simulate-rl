# Simulated Pursuit-Evasion RL

This repository contains a very small demonstration of a 3D pursuit--evasion
environment written using `gymnasium`. Two agents (an evader and a pursuer)
move in a simplified physics world. The provided scripts allow training a
pursuer policy with a basic REINFORCE loop.

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

## Running the first training

The `train_pursuer.py` script trains a small neural network policy for the
pursuer while the evader follows a scripted behaviour. To start training run:

```bash
python train_pursuer.py
```
The script saves the trained policy to `pursuer_policy.pt` by default. You can
override this with `--save-path`.

The script accepts a few command line options to control the training. For
instance, to run 200 episodes with a smaller learning rate and evaluation every
20 episodes while saving checkpoints every 50 episodes:

```bash
python train_pursuer.py --episodes 200 --lr 5e-4 --eval-freq 20 --checkpoint-every 50
```

The defaults for these options live in ``config.yaml`` under the
``training`` section and can be modified directly in that file.

It will print evaluation statistics every ``--eval-freq`` episodes and a final
summary when training finishes.

### Monitoring training with TensorBoard

Both training scripts can write metrics for visualization with TensorBoard.
Pass ``--log-dir`` to specify where logs should be stored. For example:

```bash
python train_pursuer.py --log-dir runs/reinforce
```

Start TensorBoard with

```bash
tensorboard --logdir runs
```

This will show episode rewards, evaluation results and losses during training.
When a log directory is set, periodic checkpoints created with
``--checkpoint-every`` are stored under ``<log-dir>/checkpoints``.

### New training options

Both training scripts now support additional parameters for
weight decay, learning rate scheduling and model size. These can be set via the
`training` section in `config.yaml` or from the command line:

```bash
python train_pursuer.py --weight-decay 1e-4 --lr-step-size 500 --lr-gamma 0.9 \
    --hidden-size 128 --activation tanh
```

The `hidden-size` and `activation` options control the width and activation
function of the two-layer MLP used by the policies (available activations are
`relu`, `tanh` and `leaky_relu`).

### Hyperparameter sweeps

The repository includes a small `sweep.py` utility which iterates over a grid of
hyperparameters and logs each run to its own TensorBoard directory. The sweep
script also records the number of episodes required to reach a configurable
average reward threshold via the `sweep/episodes_to_reward` metric.

### PPO variant

For a more stable actor--critic approach, run the ``train_pursuer_ppo.py``
script which implements a minimal Proximal Policy Optimization loop with an
entropy bonus:

```bash
python train_pursuer_ppo.py
```
The command line options are the same as for ``train_pursuer.py`` and the
trained weights are written to ``pursuer_ppo.pt`` unless ``--save-path`` is
specified. Both training scripts also support ``--checkpoint-every`` to save
periodic checkpoints and ``--resume-from`` to continue from a saved model.
If ``--log-dir`` is supplied these checkpoints are placed in ``<log-dir>/checkpoints``.
The PPO trainer additionally accepts ``--num-envs`` to run several
environment instances in parallel which can significantly speed up data
collection on multi-core machines. All algorithm parameters
(``gamma``, ``clip_ratio``, ``ppo_epochs`` and the entropy bonus weight)
live in the ``training`` section of ``config.yaml`` and may be overridden
via command line flags. The entropy bonus coefficient decays linearly
from ``entropy_coef_start`` to ``entropy_coef_end`` over the course of
training.

### Curriculum training

Both training scripts optionally support gradually increasing the starting
difficulty of each episode. The `training.curriculum` section in
`config.yaml` contains `start` and `end` dictionaries with values that are
interpolated over the course of training. Any numeric field under these
dictionaries will be linearly scaled from the `start` value to the `end`
value as episodes progress. For example, the default configuration narrows
the pursuer's `yaw_range` and initial `force_target_radius` to begin the
agent immediately behind the evader while increasing `evader_start.initial_speed`
from 0&nbsp;m/s to 50&nbsp;m/s before expanding to the full search
area. The curriculum makes it possible to smoothly transition from simple
encounters to more challenging ones.

## Additional scripts

- `pursuit_evasion.py` contains the environment implementation along with a
  `main()` function that runs a single random episode. This can be executed via

```bash
python pursuit_evasion.py
```

which is useful for quickly checking that the environment works.

- `play.py` loads a saved policy and runs a single episode. Use the `--ppo`
  flag when loading a model trained with the PPO script. Episodes run for the
  duration specified by `episode_duration` in `config.yaml` unless `--steps` is
  used to override the maximum number of simulation steps.
  The plot now highlights the starting and final positions of both agents,
  marks the evader's goal position and draws arrows indicating the initial
  heading of both players. During the run a table prints the distance vectors
  between the players and the goal along with the current velocities for both
  agents. The spawn volume for the pursuer is now outlined with green lines
  only (surface fills were removed to keep the plot responsive).
- `plot_config.py` renders a stand-alone visualisation of the environment
  configuration showing an outline of the spawn volume. The accompanying
  `SpawnVolumeDemo.ipynb` notebook calls this script so you can interactively
  adjust `config.yaml` and inspect the effect.

The environment stores several statistics for each episode. When an episode
finishes the ``info`` dictionary returned from ``env.step`` contains the
closest pursuer--evader distance, number of steps and outcome (capture,
evader reaching the target while airborne, separation exceeding a multiple of the starting distance (controlled by `separation_cutoff_factor` in `config.yaml`) or timeout). The evaluation helpers in the training
scripts print the average minimum distance and episode length during
periodic evaluations.

## Adjusting environment parameters

All physical constants and environment options are stored in
`config.yaml` in the repository root.  Simply edit this file to tweak
values such as masses, maximum acceleration or the starting distance of
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
Both `pursuit_evasion.py` and `train_pursuer.py` load the configuration
at runtime, so changes take effect the next time you run the scripts.
The reward shaping parameters `shaping_weight`, `closer_weight` and
`angle_weight` can be adjusted here as well to encourage desired
behaviour. The `separation_cutoff_factor` option defines a multiplier of
the initial pursuer--evader distance that ends the episode when the
agents drift farther apart than this threshold.
The `capture_bonus` setting adds a time incentive for the pursuer by
increasing its terminal reward when a capture occurs earlier. The final
reward becomes `1 + capture_bonus * (max_steps - episode_steps)` where
`max_steps` is the episode length limit.

The `evader.awareness_mode` option defines how much information the
evader receives about the pursuer:

1. `1` – no knowledge of the pursuer
2. `2` – only the distance to the pursuer
3. `3` – the unit vector pointing to the pursuer
4. `4` – full pursuer position (values above 4 behave the same)

The `yaw_rate` and `pitch_rate` values for both agents are specified in
degrees per second and are converted internally to radians per second.
Similarly, the `stall_angle` parameter in `config.yaml` is given in
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
``target_success_distance`` setting.

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
