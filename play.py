import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pursuit_evasion import PursuerPolicy, load_config
from train_pursuer import PursuerOnlyEnv
from train_pursuer_ppo import ActorCritic


def run_episode(model_path: str, use_ppo: bool = False) -> None:
    cfg = load_config()
    cfg['evader']['awareness_mode'] = 1
    env = PursuerOnlyEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_ppo:
        model = ActorCritic(env.observation_space.shape[0])
    else:
        model = PursuerPolicy(env.observation_space.shape[0])
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    obs, _ = env.reset()
    # collect positions for plotting
    pursuer_traj = [env.env.pursuer_pos.copy()]
    evader_traj = [env.env.evader_pos.copy()]

    done = False
    total_reward = 0.0
    info = {}
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            if use_ppo:
                mean, _ = model(obs_t)
            else:
                mean = model(obs_t)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean))
            action = dist.mean
        obs, r, done, _, info = env.step(action.cpu().numpy())
        pursuer_traj.append(env.env.pursuer_pos.copy())
        evader_traj.append(env.env.evader_pos.copy())
        total_reward += r

    print(f"Episode reward: {total_reward:.2f}")
    if info:
        min_d = info.get('min_distance')
        closest = f"{min_d:.2f}" if min_d is not None else "n/a"
        print(
            f"Steps: {info.get('episode_steps', 'n/a')}  "
            f"closest={closest}  "
            f"outcome={info.get('outcome', 'unknown')}"
        )

    # Plot the trajectories in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = np.stack(pursuer_traj)
    e = np.stack(evader_traj)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], label="pursuer")
    ax.plot(e[:, 0], e[:, 1], e[:, 2], label="evader")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single episode using a saved pursuer model"
    )
    parser.add_argument(
        "--model", type=str, default="pursuer_policy.pt", help="path to weight file"
    )
    parser.add_argument(
        "--ppo", action="store_true", help="load weights from PPO training"
    )
    args = parser.parse_args()

    run_episode(args.model, use_ppo=args.ppo)
