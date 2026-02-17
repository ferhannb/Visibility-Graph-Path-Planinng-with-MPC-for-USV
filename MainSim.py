#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated main module integrating (optionally) the new Quadtree/Lazy Visibility Graph
implementation with the existing MPC path tracking pipeline.

Workflow:
 1) Generate random obstacles (inflated with a safety buffer via Shapely)
 2) Build a visibility graph (classic or quadtree lazy edges)
 3) Extract shortest path → waypoint list
 4) Run curvature‑bounded MPC tracker along the waypoint sequence
 5) Plot environment, path, and closed‑loop trajectory

You can change most parameters from the command line, e.g.:
    python Main.py --algo quadtree --num-obs 40 --seed 1 --goal 40 55 \
                   --v-desired 2.5 --dt 0.4 --N 25 --ds-max 3.0 --mpc-steps 80

If run as a module (import), nothing executes until main() is called.
"""
from __future__ import annotations
import math, time, random, argparse, sys
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon, JOIN_STYLE
# smooth-curve helpers
from plotCurvelinearArc import cartesian_to_curvilinear, sample_arc


# ---- Local modules ---------------------------------------------------------
from MPCTrackingV2 import PathOptimizer
from quadtreeVG import ClassicVisibilityGraph, QuadtreeVisibilityGraph

# ---- Type aliases ----------------------------------------------------------
Point = Tuple[float, float]
Polygon = List[Point]



# =============================================================================
# Geometry helpers (unchanged logic; kept local for quick shape generation)
# =============================================================================
def rotate(pt: Point, ang: float, about: Point = (0.0, 0.0)) -> Point:
    ox, oy = about
    x, y = pt
    ca, sa = math.cos(ang), math.sin(ang)
    return (ox + ca * (x - ox) - sa * (y - oy),
            oy + sa * (x - ox) + ca * (y - oy))

def regular_polygon(center: Point, radius: float, n: int, phase: float = 0.0) -> Polygon:
    cx, cy = center
    return [(cx + radius * math.cos(2 * math.pi * i / n + phase),
             cy + radius * math.sin(2 * math.pi * i / n + phase)) for i in range(n)]

def rectangle(center: Point, w: float, h: float, ang: float = 0.0) -> Polygon:
    cx, cy = center
    dx, dy = w / 2, h / 2
    pts = [(cx - dx, cy - dy), (cx + dx, cy - dy),
           (cx + dx, cy + dy), (cx - dx, cy + dy)]
    return [rotate(p, ang, center) for p in pts]

# =============================================================================
# Configuration dataclass
# =============================================================================
@dataclass
class ScenarioConfig:
    area_size: int = 150
    start_heading: float = 0.0 
    start: Point = (0.0, 0.0)
    goal: Point = (45.0, 25.0)
    num_obs: int = 30
    seed: int = 42
    buffer: float = 1.0               # safety inflation (meters)
    algo: str = "classic"              # 'classic' or 'quadtree'
    quadtree_depth: int = 4
    # MPC / tracking
    v_desired: float = 2.0
    dt: float = 0.3
    horizon_N: int = 30
    ds_max: float = 4.0
    K_max: float = 0.8            # maximum curvature magnitude |K| [1/m]
    mpc_steps: int = 50
    # --- MPC cost weights (exposed via argparse) ---
    w_pos: float = 40.0
    w_theta: float = 10.0
    w_ds_change: float = 1.0
    w_K_change: float = 1.0
    term_X: float = 120.0
    term_Y: float = 120.0
    term_theta: float = 80.0

# =============================================================================
# Obstacle generation (returns original + buffered lists)
# =============================================================================

def generate_obstacles(cfg: ScenarioConfig):
    random.seed(cfg.seed)
    obstacles: List[Polygon] = []          # inflated
    original: List[Polygon] = []           # raw
    for i in range(cfg.num_obs):
        cx = random.uniform(5, cfg.area_size - 5)
        cy = random.uniform(5, cfg.area_size - 5)
        if i % 2 == 0:
            poly = rectangle((cx, cy),
                             w=random.uniform(2, 8),
                             h=random.uniform(2, 8),
                             ang=random.uniform(0, 2 * math.pi))
        else:
            poly = regular_polygon((cx, cy),
                                   radius=random.uniform(2, 6),
                                   n=random.choice([3, 4, 5, 6]),
                                   phase=random.uniform(0, 2 * math.pi))
        original.append(poly)

        # Shapely buffer for safety margin
        shapely_poly = ShapelyPolygon(poly)
        if not shapely_poly.is_valid:
            shapely_poly = shapely_poly.buffer(0)
        buffered = shapely_poly.buffer(cfg.buffer,
                                       join_style=JOIN_STYLE.mitre,
                                       mitre_limit=5.0)
        if buffered.geom_type == "Polygon":
            obstacles.append(list(buffered.exterior.coords)[:-1])
        else:  # MultiPolygon
            for part in buffered.geoms:
                obstacles.append(list(part.exterior.coords)[:-1])
    return original, obstacles

# =============================================================================
# Visibility Graph construction & shortest path
# =============================================================================

def build_path(cfg: ScenarioConfig, inflated_obstacles: List[Polygon]):
    if cfg.algo == "quadtree":
        vg = QuadtreeVisibilityGraph(cfg.start, cfg.goal, inflated_obstacles,
                                     env_bbox=(0, 0, cfg.area_size, cfg.area_size),
                                     max_depth=cfg.quadtree_depth)
    elif cfg.algo == "classic":
        vg = ClassicVisibilityGraph(cfg.start, cfg.goal, inflated_obstacles)
    else:
        raise ValueError(f"Unknown algo '{cfg.algo}'. Choose 'classic' or 'quadtree'.")

    t0 = time.time()
    vg.build()            # (quadtree inherits .build() from classic for node+edge generation)
    path = vg.shortest_path()
    build_time = (time.time() - t0)
    if not path:
        raise RuntimeError("No path found by visibility graph.")
    return np.array(path, dtype=float), vg, build_time

# =============================================================================
# MPC Simulation
# =============================================================================

def run_mpc(cfg: ScenarioConfig, waypoints: np.ndarray):
    if len(waypoints) > 1:
        dx0, dy0 = waypoints[1] - waypoints[0]
        theta0 = math.atan2(dy0, dx0)
    else:
        theta0 = 0.0
    x,y,th=0,5,0
    # x, y, th = waypoints[0, 0], waypoints[0, 1], theta0

    # Build optimizer; include K_max if the class supports it
    # Build optimizer kwargs (include weights)
    optimizer_kwargs = dict(
        X_init=x, Y_init=y, theta_init=th,
        waypoints=waypoints,
        v_desired=cfg.v_desired,
        dt=cfg.dt,
        N=cfg.horizon_N,
        ds_max=cfg.ds_max,
        w_pos=cfg.w_pos,
        
        w_theta=cfg.w_theta,
        w_ds_change=cfg.w_ds_change,
        w_K_change=cfg.w_K_change,
        term_X=cfg.term_X,
        term_Y=cfg.term_Y,
        term_theta=cfg.term_theta,
    )
    # Inject K_max only if PathOptimizer supports it
    try:
        import inspect
        if 'K_max' in inspect.signature(PathOptimizer.__init__).parameters:
            optimizer_kwargs['K_max'] = cfg.K_max
    except Exception as e:
        print(f"[Warn] Could not introspect PathOptimizer signature: {e}")
    
    opt = PathOptimizer(**optimizer_kwargs)
    opt.x_ref_full, opt.y_ref_full, opt.th_ref_full = opt.build_global_ref(opt.dt)
    opt.ref_index = 0
    for k in range(cfg.mpc_steps):
        ds_cmd, K_cmd = opt.mpc_step(x, y, th)
        dth = K_cmd * ds_cmd
        th += dth
        if abs(dth) < 1e-4:
            x += math.cos(th) * ds_cmd
            y += math.sin(th) * ds_cmd
        else:
            R = 1.0 / K_cmd
            x += R * (math.sin(th) - math.sin(th - dth))
            y += -R * (math.cos(th) - math.cos(th - dth))
        if np.linalg.norm([x - cfg.goal[0], y - cfg.goal[1]]) < 0.6:
            print(f"[MPC] Goal reached early at step {k}")
            break
    return opt

# =============================================================================
# Plotting
# =============================================================================
def plot_environment(cfg: ScenarioConfig, original_obs, inflated_obs, opt: PathOptimizer):
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 2,
        "axes.linewidth": 1.2,
    })

    # --------------------------------
    #  Data
    # --------------------------------
    if not opt.mpc_X_hist:
        raise RuntimeError("MPC has not run yet; there is no data to plot.")

    traj = np.column_stack([opt.mpc_X_hist, opt.mpc_Y_hist])
    prog = np.linspace(0, 1, traj.shape[0])

    # --------------------------------
    #  Figure / Axis
    # --------------------------------
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # ---- MPC closed-loop trajectory
    # sc = ax.scatter(traj[:, 0], traj[:, 1],
    #                 s=30, label="Trajectory")
    # sc.set_rasterized(True)


    # ---- Waypoints
    if opt.waypoints is not None:
        ax.plot(opt.waypoints[:, 0], opt.waypoints[:, 1], "r--o",
                markersize=5, linewidth=2, label="Waypoints")



    # ---- Obstacles
    for poly in original_obs:
        xs, ys = zip(*(poly + [poly[0]]))
        ax.fill(xs, ys, facecolor="lightgray", edgecolor="dimgray",
                linewidth=1.0, alpha=0.5)
    for poly in inflated_obs:
        xs, ys = zip(*(poly + [poly[0]]))
        ax.plot(xs, ys, "g--", linewidth=1.2)


    try:
        Xc, Yc = build_smooth_path(traj if len(traj) else opt.waypoints,
                                theta0=0)
        ax.plot(Xc, Yc, color="royalblue", linewidth=1.8,
                alpha=0.85, label="Smooth curve")
    except Exception as e:
        print("[curve‑warn]", e) 

    # ---- Start / Goal
    ax.scatter(cfg.start[0], cfg.start[1], marker="*", color="green", s=180, label="Start")
    ax.scatter(cfg.goal[0],  cfg.goal[1],  marker="X", color="red",   s=220, label="Goal")

    # --------------------------------
    #  Stil / limitler
    # --------------------------------
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"VisibilityGraph ({cfg.algo}) Path Generation and  Smoothed Path")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)

    wx = opt.waypoints[:, 0] if opt.waypoints is not None else np.array([])
    wy = opt.waypoints[:, 1] if opt.waypoints is not None else np.array([])

    obs_xy = np.concatenate([np.array(poly) for poly in original_obs]) \
             if original_obs else np.empty((0, 2))
    all_x = np.concatenate([traj[:, 0], opt.x_ref_full,
                            opt.waypoints[:, 0], obs_xy[:, 0]])
    all_y = np.concatenate([traj[:, 1], opt.y_ref_full,
                            opt.waypoints[:, 1], obs_xy[:, 1]])

    # Dynamic margin: 5% of span + 1 m
    span = max(all_x.max() - all_x.min(),
               all_y.max() - all_y.min())
    margin = 0.05 * span + 1.0

    leg = ax.legend(loc="upper left", markerscale=1.2)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.85)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95)
    plt.show()



def plot_ds_K(opt, mode='dsK', dt=None):
    """Plot histories using stem plots (two stacked subplots) in a version‑agnostic way.

    Parameters
    ----------
    opt : PathOptimizer (expected to have mpc_ds_hist, mpc_K_hist)
    mode : str, 'dsK' or 'vomega'
        * 'dsK'    → top: ds, bottom: K
        * 'vomega' → top: v = ds/dt, bottom: ω = v*K
    dt : float or None
        Needed only if mode == 'vomega'. If None, attempts to read opt.dt.
    """
    if not hasattr(opt, 'mpc_ds_hist') or not hasattr(opt, 'mpc_K_hist'):
        print('[Plot] Optimizer missing histories mpc_ds_hist / mpc_K_hist; skipping plot.')
        return
    ds_hist = np.asarray(opt.mpc_ds_hist, dtype=float)
    K_hist  = np.asarray(opt.mpc_K_hist, dtype=float)
    if ds_hist.size == 0:
        print('[Plot] Empty ds/K histories; nothing to plot.')
        return

    import matplotlib
    from matplotlib.collections import LineCollection

    steps = np.arange(ds_hist.size)

    if mode == 'vomega':
        if dt is None:
            dt = getattr(opt, 'dt', None)
        if not dt or dt <= 0:
            print('[Plot] Invalid dt for v/ω conversion; falling back to dsK.')
            mode = 'dsK'
        else:
            v_hist = ds_hist / dt
            omega_hist = v_hist * K_hist
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    def _style_stem(stem_ret, color_idx=0):
        """Apply styling compatible across Matplotlib versions.
        stem_ret is the 3‑tuple returned by ax.stem.
        """
        markerline, stemlines, baseline = stem_ret
        # marker
        markerline.set_markerfacecolor(f'C{color_idx}')
        markerline.set_markersize(5)
        # stems: could be LineCollection or sequence
        if isinstance(stemlines, LineCollection):
            stemlines.set_linewidth(1.1)
            stemlines.set_color(matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][color_idx])
        else:  # iterable of Line2D
            for ln in stemlines:
                ln.set_linewidth(1.1)
        baseline.set_linewidth(1.0)
        return markerline, stemlines, baseline

    if mode == 'dsK':
        stem_ds = ax1.stem(steps, ds_hist)
        _style_stem(stem_ds, 0)
        ax1.set_title('Optimal ds Control Signal')
        ax1.set_ylabel('ds [m]')

        stem_K = ax2.stem(steps, K_hist)
        _style_stem(stem_K, 1)
        ax2.set_title('Optimal K Control Signal')
        ax2.set_ylabel('Curvature K [1/m]')
    else:  # vomega
        stem_v = ax1.stem(steps, v_hist)
        _style_stem(stem_v, 0)
        ax1.set_title('Optimal V Control Signal')
        ax1.set_ylabel('V [m/s]')

        stem_w = ax2.stem(steps, omega_hist)
        _style_stem(stem_w, 1)
        ax2.set_title('Optimal Omega Control Signal')
        ax2.set_ylabel('Ω [rad/s]')

    ax2.set_xlabel('Step Number')

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    plt.show()

def build_smooth_path(points, theta0: float = 0.0, n_per_seg: int = 80):
    """
    Generates intermediate points from sparse (x, y) samples using
    constant-curvature arc segments.
    points  : [(x0,y0), (x1,y1), ...]  (numpy Nx2 or list)
    theta0  : Initial heading of the first segment (rad), usually vessel heading
    returns : X_curve, Y_curve  (numpy 1-D)
    """
    import numpy as np
    ds, phi, theta_ends, _ = cartesian_to_curvilinear(points, theta0)

    x_curves, y_curves = [], []
    x0, y0 = points[0]
    for i, (ds_i, phi_i) in enumerate(zip(ds, phi)):
        th_start = theta0 if i == 0 else theta_ends[i-1]
        xseg, yseg = sample_arc(x0, y0, th_start, ds_i, phi_i, n=n_per_seg)
        # Skip first point to avoid duplicating segment junctions
        if x_curves:
            xseg, yseg = xseg[1:], yseg[1:]
        x_curves.append(xseg); y_curves.append(yseg)
        x0, y0 = points[i+1]
    return np.concatenate(x_curves), np.concatenate(y_curves)

# =============================================================================
# Main execution
# =============================================================================

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Visibility Graph + MPC demo")
    parser.add_argument('--start-heading', type=float, default=60,
                    help="Initial heading angle [rad]")
    parser.add_argument('--algo', choices=['classic', 'quadtree'], default='quadtree')
    parser.add_argument('--area-size', type=int, default=75)
    parser.add_argument('--num-obs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal', type=float, nargs=2, default=[65.0, 43.0], metavar=('GX','GY'))
    parser.add_argument('--start', type=float, nargs=2, default=[0.0, 0.0], metavar=('SX','SY'))
    parser.add_argument('--buffer', type=float, default=1.0)
    parser.add_argument('--quadtree-depth', type=int, default=4)
    # MPC params
    parser.add_argument('--v-desired', type=float, default=3.0)
    parser.add_argument('--dt', type=float, default=1)
    parser.add_argument('--N', type=int, default=20, help='MPC horizon length')
    parser.add_argument('--ds-max', type=float, default=10.0)
    parser.add_argument('--K-max', type=float, default=0.5, help='Maximum curvature magnitude |K| [1/m]')
    parser.add_argument('--mpc-steps', type=int, default=300)
    # Cost weights
    parser.add_argument('--w-pos', type=float, default=40.0, dest='w_pos', help='Weight for position (X,Y) error term')
    parser.add_argument('--w-theta', type=float, default=40.0, dest='w_theta', help='Weight for heading error')
    parser.add_argument('--w-ds-change', type=float, default=1500.0, dest='w_ds_change', help='Weight for ds smoothness (consecutive difference)')
    parser.add_argument('--w-K-change', type=float, default=50000, dest='w_K_change', help='Weight for curvature smoothness (consecutive difference)')
    parser.add_argument('--term-X', type=float, default=120.0, dest='term_X', help='Terminal X position weight (currently unused / commented)')
    parser.add_argument('--term-Y', type=float, default=120.0, dest='term_Y', help='Terminal Y position weight (unused)')
    parser.add_argument('--term-theta', type=float, default=80.0, dest='term_theta', help='Terminal heading weight (unused)')
    parser.add_argument('--no-dsK', action='store_true', help='Disable ds/K or v/omega history plot')
    parser.add_argument('--controls-mode', choices=['dsK', 'vomega'], default='dsK', help='Choose whether to plot (ds,K) or (v,omega)')

    args = parser.parse_args(argv)

    cfg = ScenarioConfig(
        area_size=args.area_size,
        start=tuple(args.start),
        goal=tuple(args.goal),
        num_obs=args.num_obs,
        seed=args.seed,
        buffer=args.buffer,
        algo=args.algo,
        quadtree_depth=args.quadtree_depth,
        v_desired=args.v_desired,
        dt=args.dt,
        horizon_N=args.N,
        ds_max=args.ds_max,
        mpc_steps=args.mpc_steps,
        K_max=args.K_max,
        w_pos=args.w_pos,

        w_theta=args.w_theta,
        w_ds_change=args.w_ds_change,
        w_K_change=args.w_K_change,
        term_X=args.term_X,
        term_Y=args.term_Y,
        term_theta=args.term_theta,
    )

    print("[Scenario]", cfg)
    original_obs, inflated_obs = generate_obstacles(cfg)
    waypoints, vg, build_time = build_path(cfg, inflated_obs)
    print(f"[VisibilityGraph] nodes={len(vg.nodes)} path_points={len(waypoints)} build+solve_time={build_time:.3f}s")
    opt = run_mpc(cfg, waypoints)
    print("[MPC] Simulation completed. Logged steps:", len(opt.mpc_ds_hist))
    plot_environment(cfg, original_obs, inflated_obs, opt)
    if not args.no_dsK:
        plot_ds_K(opt, mode=args.controls_mode, dt=cfg.dt)

# Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    main()  # pass sys.argv[1:] implicitly
