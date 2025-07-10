import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pursuit_evasion import PursuerPolicy, load_config
from train_pursuer import PursuerOnlyEnv
from train_pursuer_ppo import ActorCritic


def run_episode(model_path: str, use_ppo: bool = False, max_steps: int | None = None) -> None:
    cfg = load_config()
    cfg['evader']['awareness_mode'] = 1
    env = PursuerOnlyEnv(cfg, max_steps=max_steps)
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
            f"start={info.get('start_distance', float('nan')):.2f}  "
            f"outcome={info.get('outcome', 'unknown')}"
        )

    # Plot the trajectories in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = np.stack(pursuer_traj)
    e = np.stack(evader_traj)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], label="pursuer")
    ax.plot(e[:, 0], e[:, 1], e[:, 2], label="evader")
    # draw arrows showing the initial heading for both agents
    arrow_len = 1000.0
    p_dir = env.env.pursuer_vel
    p_norm = np.linalg.norm(p_dir)
    if p_norm > 1e-6:
        p_dir = p_dir / p_norm
    e_dir = env.env.evader_force_dir
    e_dir = e_dir / (np.linalg.norm(e_dir) + 1e-8)
    ax.quiver(
        p[0, 0],
        p[0, 1],
        p[0, 2],
        p_dir[0],
        p_dir[1],
        p_dir[2],
        length=arrow_len,
        color="blue",
        arrow_length_ratio=0.1,
        label="pursuer heading",
    )
    ax.quiver(
        e[0, 0],
        e[0, 1],
        e[0, 2],
        e_dir[0],
        e_dir[1],
        e_dir[2],
        length=arrow_len,
        color="orange",
        arrow_length_ratio=0.1,
        label="evader heading",
    )
    # mark the evader's target position
    target = np.asarray(env.env.cfg["target_position"], dtype=float)
    ax.scatter(*target, color="red", marker="*", s=100, label="goal")
    # mark starting and final positions for both agents
    ax.scatter(
        p[0, 0],
        p[0, 1],
        p[0, 2],
        color="blue",
        marker="o",
        s=60,
        label="pursuer start",
    )
    ax.scatter(
        e[0, 0],
        e[0, 1],
        e[0, 2],
        color="orange",
        marker="o",
        s=60,
        label="evader start",
    )
    ax.scatter(
        p[-1, 0],
        p[-1, 1],
        p[-1, 2],
        color="blue",
        marker="X",
        s=80,
        label="pursuer end",
    )
    ax.scatter(
        e[-1, 0],
        e[-1, 1],
        e[-1, 2],
        color="orange",
        marker="X",
        s=80,
        label="evader end",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    # keep all three axes scaled equally so movement is not distorted
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:  # matplotlib < 3.3 fallback
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        radius = (limits[:, 1] - limits[:, 0]).max() / 2
        centers = limits.mean(axis=1)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
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
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override maximum episode steps",
    )
    args = parser.parse_args()

    run_episode(args.model, use_ppo=args.ppo, max_steps=args.steps)
