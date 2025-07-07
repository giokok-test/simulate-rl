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
pip install numpy torch gymnasium pyyaml
```

These are the only packages used in the example scripts.

## Running the first training

The `train_pursuer.py` script trains a small neural network policy for the
pursuer while the evader follows a scripted behaviour. To start training run:

```bash
python train_pursuer.py
```

The script will print evaluation statistics every few episodes and a final
summary when training finishes.

## Additional scripts

- `pursuit_evasion.py` contains the environment implementation along with a
  `main()` function that runs a single random episode. This can be executed via

```bash
python pursuit_evasion.py
```

which is useful for quickly checking that the environment works.

## Adjusting environment parameters

All physical constants and environment options are stored in
`config.yaml` in the repository root.  Simply edit this file to tweak
values such as masses, maximum acceleration or the starting distance of
the pursuer.  Both `pursuit_evasion.py` and `train_pursuer.py` load the
configuration at runtime, so changes take effect the next time you run
the scripts.
