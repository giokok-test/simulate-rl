import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pursuit_evasion import load_config
from curriculum import Curriculum, initialize_gym
from train_pursuer_qlearning import QNetwork, ACTIONS

# Discretisation for legacy Q-table models
N_BINS = 10
MAX_DIST = 5000.0
BINS = np.linspace(-MAX_DIST, MAX_DIST, N_BINS - 1)

def _discretise(obs: np.ndarray) -> int:
    diff = obs[6:9] - obs[0:3]
    bins = np.digitize(diff, BINS)
    return int(bins[0] * N_BINS * N_BINS + bins[1] * N_BINS + bins[2])


def draw_spawn_volume(
    ax,
    apex,
    inner,
    outer,
    r_min,
    r_max,
    yaw_ranges,
    *,
    color="green",
    alpha=0.15,
    linestyle="--",
):
    """Draw pursuer spawn cone as closed translucent surfaces."""

    line_kw = {"color": color, "linestyle": linestyle}
    for yaw_start, yaw_end in yaw_ranges:
        theta = np.linspace(yaw_start, yaw_end, 30)
        for r in np.linspace(r_min, r_max, 3):
            x = apex[0] + r * np.sin(outer) * np.cos(theta)
            y = apex[1] + r * np.sin(outer) * np.sin(theta)
            z = apex[2] - r * np.cos(outer)
            ax.plot(x, y, z, **line_kw)
            if inner > 0:
                x = apex[0] + r * np.sin(inner) * np.cos(theta)
                y = apex[1] + r * np.sin(inner) * np.sin(theta)
                z = apex[2] - r * np.cos(inner)
                ax.plot(x, y, z, **line_kw)
        for ang in [yaw_start, yaw_end]:
            x = apex[0] + np.array([r_min, r_max]) * np.sin(outer) * np.cos(ang)
            y = apex[1] + np.array([r_min, r_max]) * np.sin(outer) * np.sin(ang)
            z = apex[2] - np.array([r_min, r_max]) * np.cos(outer)
            ax.plot(x, y, z, **line_kw)
            if inner > 0:
                x = apex[0] + np.array([r_min, r_max]) * np.sin(inner) * np.cos(ang)
                y = apex[1] + np.array([r_min, r_max]) * np.sin(inner) * np.sin(ang)
                z = apex[2] - np.array([r_min, r_max]) * np.cos(inner)
                ax.plot(x, y, z, **line_kw)

    # Surfaces that used to fill the volume were removed to reduce lag
    # in interactive plots. Only the outline is drawn now so the spawn
    # region remains visible without the heavy meshes.


def run_episode(
    model_path: str,
    max_steps: int | None = None,
    profile: bool = False,
    *,
    config_path: str | None = None,
    progress: float = 1.0,
) -> None:
    """Run one episode using ``model_path``.

    When ``profile`` is ``True`` the function records how much time is spent
    on inference, environment stepping and plotting before printing a summary.
    """

    cfg = load_config(config_path)
    cfg['evader']['awareness_mode'] = 1
    cur_cfg = cfg.get("curriculum") or {}
    curriculum = None
    mode = cur_cfg.get("mode")
    if mode:
        curriculum = Curriculum(
            start=cur_cfg.get("start", {}),
            end=cur_cfg.get("end", {}),
            mode=mode,
            stages=cur_cfg.get("stages", 2),
            success_threshold=cur_cfg.get("success_threshold", 0.6),
            window=cur_cfg.get("window", 64),
        )
        curriculum.stage = int(progress * max(curriculum.stages - 1, 1))
    env = initialize_gym(cfg, curriculum=curriculum, max_steps=max_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_q_table = model_path.endswith(".npy")
    if use_q_table:
        q_table = np.load(model_path)
    else:
        model = QNetwork(env.observation_space.shape[0], len(ACTIONS))
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

    obs, _ = env.reset()
    if profile:
        total_start = time.perf_counter()
        infer_time = 0.0
        step_time = 0.0
    # store initial state before stepping
    p_init_dir = env.env.pursuer_force_dir.copy()
    e_init_dir = env.env.evader_force_dir.copy()
    e_start_pos = env.env.evader_pos.copy()
    heading_dir = env.env.evader_vel / (np.linalg.norm(env.env.evader_vel) + 1e-8)
    # collect positions for plotting
    pursuer_traj = [env.env.pursuer_pos.copy()]
    evader_traj = [env.env.evader_pos.copy()]

    # print table header showing distance vectors, velocities and directions
    header = "{:>5} | {:>26} | {:>26} | {:>26} | {:>26} | {:>18} | {:>18} | {:>18}".format(
        "step",
        "pursuer->evader [m]",
        "evader->target [m]",
        "pursuer vel [m/s]",
        "evader vel [m/s]",
        "p dir",
        "e dir",
        "p->e dir",
    )
    print(header)
    print("-" * len(header))

    done = False
    total_reward = 0.0
    info = {}
    step = 0
    target_pos = np.asarray(env.env.cfg["target_position"], dtype=float)
    while not done:
        if profile:
            t0 = time.perf_counter()
        if use_q_table:
            idx = _discretise(obs)
            action = ACTIONS[int(np.argmax(q_table[idx]))]
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = model(obs_t)
            action = ACTIONS[int(torch.argmax(q_values, dim=1).item())]
        if profile:
            infer_time += time.perf_counter() - t0
            t0 = time.perf_counter()
        obs, r, done, _, info = env.step(action)
        if profile:
            step_time += time.perf_counter() - t0
        pursuer_traj.append(env.env.pursuer_pos.copy())
        evader_traj.append(env.env.evader_pos.copy())
        # compute vectors and velocities for table output
        pe_vec = env.env.evader_pos - env.env.pursuer_pos
        et_vec = target_pos - env.env.evader_pos
        pv = env.env.pursuer_vel
        ev = env.env.evader_vel
        pv_u = pv / (np.linalg.norm(pv) + 1e-8)
        ev_u = ev / (np.linalg.norm(ev) + 1e-8)
        pe_u = pe_vec / (np.linalg.norm(pe_vec) + 1e-8)
        print(
            f"{step:5d} | "
            f"[{pe_vec[0]:7.1f} {pe_vec[1]:7.1f} {pe_vec[2]:7.1f}] | "
            f"[{et_vec[0]:7.1f} {et_vec[1]:7.1f} {et_vec[2]:7.1f}] | "
            f"[{pv[0]:7.1f} {pv[1]:7.1f} {pv[2]:7.1f}] | "
            f"[{ev[0]:7.1f} {ev[1]:7.1f} {ev[2]:7.1f}] | "
            f"[{pv_u[0]:6.2f} {pv_u[1]:6.2f} {pv_u[2]:6.2f}] | "
            f"[{ev_u[0]:6.2f} {ev_u[1]:6.2f} {ev_u[2]:6.2f}] | "
            f"[{pe_u[0]:6.2f} {pe_u[1]:6.2f} {pe_u[2]:6.2f}]"
        )
        step += 1
        total_reward += r

    print(f"Episode reward: {total_reward:.2f}")
    if info:
        min_d = info.get('min_distance')
        closest = f"{min_d:.2f}" if min_d is not None else "n/a"
        start_d = info.get('start_distance')
        ratio = (
            f"{min_d / start_d:.3f}" if min_d is not None and start_d and start_d > 0 else "n/a"
        )
        print(
            f"Steps: {info.get('episode_steps', 'n/a')}  "
            f"closest={closest}  "
            f"start={start_d:.2f}  "
            f"ratio={ratio}  "
            f"outcome={info.get('outcome', 'unknown')}"
        )
        acc_d = info.get('pursuer_acc_delta')
        yaw_d = info.get('pursuer_yaw_delta')
        pitch_d = info.get('pursuer_pitch_delta')
        vel_d = info.get('pursuer_vel_delta')
        yaw_diff = info.get('pursuer_yaw_diff')
        pitch_diff = info.get('pursuer_pitch_diff')
        if acc_d is not None and yaw_d is not None and pitch_d is not None:
            print(
                f"acc_delta={acc_d:.2f}  "
                f"yaw_delta={yaw_d:.2f}  "
                f"pitch_delta={pitch_d:.2f}  "
                f"vel_delta={vel_d:.2f}  "
                f"yaw_diff={yaw_diff:.2f}  "
                f"pitch_diff={pitch_diff:.2f}"
            )

    # Plot the trajectories in 3D
    if profile:
        plot_start = time.perf_counter()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # visualize spawn volume
    p_cfg = env.env.cfg["pursuer_start"]
    outer = p_cfg["cone_half_angle"]
    inner = p_cfg.get("inner_cone_half_angle", 0.0)
    sections = p_cfg.get(
        "sections",
        {"front": True, "left": True, "right": True, "back": True},
    )
    deg45 = np.deg2rad(45.0)
    base_yaw = np.arctan2(heading_dir[1], heading_dir[0])
    if "yaw_range" in p_cfg:
        yaw_min, yaw_max = p_cfg["yaw_range"]
        ranges = [(base_yaw + np.pi + yaw_min, base_yaw + np.pi + yaw_max)]
    else:
        ranges = []
        if sections.get("front", True):
            ranges.append((base_yaw - deg45, base_yaw + deg45))
        if sections.get("right", True):
            ranges.append((base_yaw - np.pi / 2 - deg45, base_yaw - np.pi / 2 + deg45))
        if sections.get("back", True):
            ranges.append((base_yaw + np.pi - deg45, base_yaw + np.pi + deg45))
        if sections.get("left", True):
            ranges.append((base_yaw + deg45, base_yaw + np.pi / 2 + deg45))
    draw_spawn_volume(
        ax,
        e_start_pos,
        inner,
        outer,
        p_cfg["min_range"],
        p_cfg["max_range"],
        ranges,
        color="green",
        linestyle="--",
        alpha=0.2,
    )
    p = np.stack(pursuer_traj)
    e = np.stack(evader_traj)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], label="pursuer")
    ax.plot(e[:, 0], e[:, 1], e[:, 2], label="evader")
    # draw arrows showing the initial heading for both agents
    arrow_len = 1000.0
    p_dir = p_init_dir / (np.linalg.norm(p_init_dir) + 1e-8)
    e_dir = e_init_dir / (np.linalg.norm(e_init_dir) + 1e-8)
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
    if profile:
        plot_time = time.perf_counter() - plot_start
        total_time = time.perf_counter() - total_start
        print(
            f"timings: inference={infer_time:.3f}s env_step={step_time:.3f}s "
            f"plot={plot_time:.3f}s total={total_time:.3f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single episode using a saved pursuer model"
    )
    parser.add_argument(
        "--model", type=str, default="pursuer_dqn.pt", help="path to weight file"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override maximum episode steps",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print timing information",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional path to configuration directory or file",
    )
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="curriculum progress in [0,1] when using a curriculum",
    )
    args = parser.parse_args()

    run_episode(
        args.model,
        max_steps=args.steps,
        profile=args.profile,
        config_path=args.config,
        progress=args.progress,
    )
