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
20 episodes:

```bash
python train_pursuer.py --episodes 200 --lr 5e-4 --eval-freq 20
```

The defaults for these options live in ``pursuit_evasion.py`` under
``config['training']`` and can be modified directly in that file.

It will print evaluation statistics every ``--eval-freq`` episodes and a final
summary when training finishes.

### PPO variant

For a more stable actor--critic approach, run the ``train_pursuer_ppo.py``
script which implements a minimal Proximal Policy Optimization loop with an
entropy bonus:

```bash
python train_pursuer_ppo.py
```
The command line options are the same as for ``train_pursuer.py`` and the
trained weights are written to ``pursuer_ppo.pt`` unless ``--save-path`` is
specified.

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
  agents. The spawn volume for the pursuer is drawn using dashed green lines so
  you can verify the configuration visually.
- `plot_config.py` renders a stand-alone visualisation of the environment
  configuration including the spawn volume. The accompanying
  `SpawnVolumeDemo.ipynb` notebook calls this script so you can interactively
  adjust `config.yaml` and inspect the effect.

The environment stores several statistics for each episode. When an episode
finishes the ``info`` dictionary returned from ``env.step`` contains the
closest pursuer--evader distance, number of steps and outcome (capture,
evader reaching the target, separation exceeding twice the starting distance or timeout). The evaluation helpers in the training
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
inner cutoff to avoid very steep spawn angles.  The `sections` dictionary
divides the horizontal plane around the evader into four 90° quadrants
(`front`, `right`, `back`, `left`) relative to the evader's initial
heading.  Each section can be enabled or disabled to further restrict the
spawn volume.  Combined with the `min_range` and `max_range` distances
these options define where the pursuer may appear at the beginning of an
episode.
Both `pursuit_evasion.py` and `train_pursuer.py` load the configuration
at runtime, so changes take effect the next time you run the scripts.

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
their pitch commands to ``[-stall_angle, +stall_angle]``. The
`episode_duration` value defines how long each episode lasts in minutes and
is used to compute the maximum number of simulation steps based on the
configured `time_step`.

When either agent touches the ground the episode terminates. If the evader
hits the ground its terminal reward scales with the distance to the target
using ``target_reward_distance``. A reward of one is given when it reaches the
goal and it falls off to zero once the evader is roughly 100&nbsp;m away by
default.
