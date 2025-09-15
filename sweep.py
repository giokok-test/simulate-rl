import argparse
import itertools
import os

from pursuit_evasion import load_config
from train_pursuer_qlearning import QConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple hyperparameter sweep")
    parser.add_argument("--log-dir", type=str, default="runs/sweep", help="base directory for TensorBoard logs")
    args = parser.parse_args()

    lrs = [1e-3, 5e-4]
    eps_decays = [0.995, 0.99]

    for i, (lr, eps_decay) in enumerate(itertools.product(lrs, eps_decays)):
        cfg = load_config()
        q_cfg_dict = cfg.get("q_learning", {}).copy()
        q_cfg_dict["lr"] = lr
        q_cfg_dict["epsilon_decay"] = eps_decay
        q_cfg = QConfig(**q_cfg_dict)
        run_dir = os.path.join(args.log_dir, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        q_cfg.log_dir = run_dir
        train(q_cfg, cfg)


if __name__ == "__main__":
    main()
