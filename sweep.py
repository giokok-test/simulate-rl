import argparse
import itertools
import os

from pursuit_evasion import load_config
from train_pursuer_ppo import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple hyperparameter sweep")
    parser.add_argument("--log-dir", type=str, default="runs/sweep", help="base directory for TensorBoard logs")
    args = parser.parse_args()

    weight_decays = [0.0, 1e-4]
    hidden_sizes = [64, 128]

    for i, (wd, hs) in enumerate(itertools.product(weight_decays, hidden_sizes)):
        cfg = load_config()
        t_cfg = cfg.setdefault("training", {})
        t_cfg["episodes"] = t_cfg.get("episodes", 500)
        t_cfg["weight_decay"] = wd
        t_cfg["hidden_size"] = hs
        run_dir = os.path.join(args.log_dir, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        train(cfg, log_dir=run_dir)


if __name__ == "__main__":
    main()
