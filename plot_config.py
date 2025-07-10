import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_config(path="config.yaml"):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def draw_ring(ax, center, radius, z, **kw):
    theta = np.linspace(0, 2 * np.pi, 200)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, z)
    ax.plot(x, y, z, **kw)


def draw_cone(ax, apex, angle, r_min, r_max, **kw):
    # angle in radians, draw approximate cone surface using lines
    theta = np.linspace(0, 2 * np.pi, 36)
    for r in np.linspace(r_min, r_max, 4):
        x = apex[0] + r * np.sin(angle) * np.cos(theta)
        y = apex[1] + r * np.sin(angle) * np.sin(theta)
        z = apex[2] - r * np.cos(angle)
        ax.plot(x, y, z, **kw)


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
    """Visualise the pursuer spawn volume as closed translucent surfaces."""

    line_kw = {"color": color, "linestyle": linestyle}
    for yaw_start, yaw_end in yaw_ranges:
        theta = np.linspace(yaw_start, yaw_end, 30)
        for r in np.linspace(r_min, r_max, 4):
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

    # Surfaces used to fill the volume were removed to improve
    # interactive performance. Only the outline lines are drawn now so the
    # spawn area remains visible without the heavy translucent meshes.


def main():
    cfg = load_config()
    target = np.array(cfg["target_position"], dtype=float)
    start_cfg = cfg["evader_start"]
    d_min, d_max = start_cfg["distance_range"]
    altitude = start_cfg["altitude"]

    # choose one example starting point on the outer radius
    evader_pos = target + np.array([d_max, 0.0, 0.0])
    evader_pos[2] = altitude

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # draw ground
    ground_size = d_max + 500
    gx, gy = np.meshgrid(
        np.linspace(target[0] - ground_size, target[0] + ground_size, 2),
        np.linspace(target[1] - ground_size, target[1] + ground_size, 2),
    )
    gz = np.zeros_like(gx)
    ax.plot_surface(gx, gy, gz, color="lightgray", alpha=0.3)

    # target and evader example position
    ax.scatter(*target, color="red", s=60, label="target")
    ax.scatter(*evader_pos, color="blue", s=60, label="evader start")

    # ring showing possible evader spawn distances
    draw_ring(ax, target[:2], d_min, altitude, color="blue", linestyle="--")
    draw_ring(ax, target[:2], d_max, altitude, color="blue")

    # pursuer spawn volume (outer & inner cone with quadrant selection)
    p_cfg = cfg["pursuer_start"]
    outer = p_cfg["cone_half_angle"]
    inner = p_cfg.get("inner_cone_half_angle", 0.0)
    sections = p_cfg.get(
        "sections",
        {"front": True, "left": True, "right": True, "back": True},
    )
    # direction from the evader toward the target projected onto the x-y plane
    heading = target - evader_pos
    heading[2] = 0.0
    base_yaw = np.arctan2(heading[1], heading[0])
    deg45 = np.deg2rad(45.0)
    ranges = []
    if sections.get("front", True):
        ranges.append((base_yaw - deg45, base_yaw + deg45))
    if sections.get("right", True):
        ranges.append((base_yaw - np.pi/2 - deg45, base_yaw - np.pi/2 + deg45))
    if sections.get("back", True):
        ranges.append((base_yaw + np.pi - deg45, base_yaw + np.pi + deg45))
    if sections.get("left", True):
        ranges.append((base_yaw + deg45, base_yaw + np.pi/2 + deg45))

    draw_spawn_volume(
        ax,
        evader_pos,
        inner,
        outer,
        p_cfg["min_range"],
        p_cfg["max_range"],
        ranges,
        color="green",
        linestyle="--",
        alpha=0.2,
    )
    ax.text(*(evader_pos - [0, 0, p_cfg["min_range"]]), "pursuer spawn volume", color="green")

    # capture radius sphere around evader
    cap_r = cfg["capture_radius"]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = evader_pos[0] + cap_r * np.cos(u) * np.sin(v)
    y = evader_pos[1] + cap_r * np.sin(u) * np.sin(v)
    z = evader_pos[2] + cap_r * np.cos(v)
    ax.plot_wireframe(x, y, z, color="orange", linewidth=0.5)
    ax.text(*evader_pos, "capture radius", color="orange")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    ax.set_title("Pursuit-evasion configuration overview")

    # add parameter summaries as figure text
    evader_txt = (
        f"Evader\n"
        f"mass: {cfg['evader']['mass']} kg\n"
        f"max accel: {cfg['evader']['max_acceleration']} m/s^2\n"
        f"top speed: {cfg['evader']['top_speed']} m/s\n"
        f"yaw rate: {cfg['evader']['yaw_rate']} deg/s\n"
        f"pitch rate: {cfg['evader']['pitch_rate']} deg/s\n"
        f"stall angle: {cfg['evader']['stall_angle']} deg\n"
    )
    pursuer_txt = (
        f"Pursuer\n"
        f"mass: {cfg['pursuer']['mass']} kg\n"
        f"max accel: {cfg['pursuer']['max_acceleration']} m/s^2\n"
        f"top speed: {cfg['pursuer']['top_speed']} m/s\n"
        f"yaw rate: {cfg['pursuer']['yaw_rate']} deg/s\n"
        f"pitch rate: {cfg['pursuer']['pitch_rate']} deg/s\n"
        f"stall angle: {cfg['pursuer']['stall_angle']} deg\n"
    )
    global_txt = (
        f"gravity: {cfg['gravity']} m/s^2\n"
        f"time step: {cfg['time_step']} s\n"
        f"episode duration: {cfg.get('episode_duration', 0.0)} min\n"
        f"shaping weight: {cfg['shaping_weight']}\n"
    )
    fig.text(0.01, 0.95, evader_txt, fontsize=9, va="top")
    fig.text(0.25, 0.95, pursuer_txt, fontsize=9, va="top")
    fig.text(0.5, 0.95, global_txt, fontsize=9, va="top")

    # ensure the 3D plot uses the same scale on all axes
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:  # for older matplotlib versions
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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
