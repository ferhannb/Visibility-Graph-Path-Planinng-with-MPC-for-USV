import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_to_pi(a: float) -> float:
    """
    Wrap angle to [-π, π).
    """
    return (a + math.pi) % (2 * math.pi) - math.pi


def _delta_theta(theta_prev: float, dx: float, dy: float) -> float:
    """
    Compute the heading change Δθ between the previous heading and the segment
    defined by (dx, dy), matching Excel's midpoint approach.
    """
    theta_mid_raw = math.atan2(dy, dx)
    best = None
    best_abs = float("inf")
    # try shifts to minimize wrap
    for k in (-1, 0, 1):
        theta_mid = theta_mid_raw + k * math.pi
        dtheta = 2 * (theta_mid - theta_prev)
        dtheta = _wrap_to_pi(dtheta)
        if abs(dtheta) < best_abs:
            best = dtheta
            best_abs = abs(dtheta)
    return best

# ---------------------------------------------------------------------------
# Cartesian → Curvilinear
# ---------------------------------------------------------------------------

def cartesian_to_curvilinear(points, theta0: float = 0.0):
    """
    Convert Cartesian waypoints to curvilinear parameters (ds, φ, θ, Δθ).

    Parameters
    ----------
    points : sequence of (x, y)
        Cartesian coordinates of successive waypoints.
    theta0 : float
        Initial heading (rad) at the first point.

    Returns
    -------
    ds            : list of float
        Incremental arc lengths (mm).
    phi           : list of float
        Curvature values (1/mm).
    theta         : list of float
        Absolute headings at each segment end (rad).
    delta_theta   : list of float
        Heading change per segment (rad).
    """
    ds = []
    phi = []
    theta = []
    delta_theta = []

    th = theta0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        # Δθ via midpoint method
        dth = _delta_theta(th, dx, dy)
        # new heading
        th_next = _wrap_to_pi(th + dth)
        # curvature φ: sin difference / dx, or alternative for dx=0
        if abs(dx) < 1e-12:
            phi_i = dth / math.hypot(dx, dy)
        else:
            phi_i = (math.sin(th_next) - math.sin(th)) / dx
        # arc length ds: Excel branch (φ==0 straight vs curved)
        if phi_i == 0.0:
            ds_i = dx / math.cos(th_next)
        else:
            ds_i = dth / phi_i
        # store
        delta_theta.append(dth)
        ds.append(ds_i)
        phi.append(phi_i)
        theta.append(th_next)
        th = th_next
    return ds, phi, theta, delta_theta

# ---------------------------------------------------------------------------
# Curvilinear → Cartesian
# ---------------------------------------------------------------------------

def curvilinear_to_cartesian(ds, phi, theta0: float = 0.0, x0: float = 0.0, y0: float = 0.0):
    """
    Reconstruct Cartesian coordinates from curvilinear parameters,
    matching the Excel 'curvilinear->Cartesian' sheet.
    """
    xs = [x0]
    ys = [y0]
    thetas = [theta0]
    dxs = []
    dys = []
    dthetas = []

    for ds_i, phi_i in zip(ds, phi):
        dth = phi_i * ds_i
        th_prev = thetas[-1]
        th_next = _wrap_to_pi(th_prev + dth)
        if phi_i == 0.0:
            dx = math.cos(th_next) * ds_i
            dy = math.sin(th_next) * ds_i
        else:
            dx = (math.sin(th_next) - math.sin(th_prev)) / phi_i
            dy = -(math.cos(th_next) - math.cos(th_prev)) / phi_i
        xs.append(xs[-1] + dx)
        ys.append(ys[-1] + dy)
        dxs.append(dx)
        dys.append(dy)
        dthetas.append(dth)
        thetas.append(th_next)
    return xs, ys, thetas, dxs, dys, dthetas

# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def sample_arc(x0: float, y0: float, theta0: float, ds: float, phi: float, n: int = 100):
    """Sample points along a constant-curvature segment."""
    s = np.linspace(0.0, ds, n)
    if abs(phi) < 1e-12:
        x = x0 + s * math.cos(theta0)
        y = y0 + s * math.sin(theta0)
    else:
        x = x0 + (1.0 / phi) * (np.sin(theta0 + phi * s) - np.sin(theta0))
        y = y0 - (1.0 / phi) * (np.cos(theta0 + phi * s) - np.cos(theta0))
    return x, y


def plot_constant_curvature_path(waypoints, theta0: float = 0.0, n_per_segment: int = 120):
    """Plot reconstructed path from Cartesian→curvilinear→sampled arcs."""
    ds, phi, theta_ends, _ = cartesian_to_curvilinear(waypoints, theta0=theta0)

    xs, ys = [], []
    for i, (ds_i, phi_i) in enumerate(zip(ds, phi)):
        x0, y0 = waypoints[i]
        th_start = theta0 if i == 0 else theta_ends[i-1]
        xseg, yseg = sample_arc(x0, y0, th_start, ds_i, phi_i, n=n_per_segment)
        if xs:
            xseg, yseg = xseg[1:], yseg[1:]
        xs.append(xseg)
        ys.append(yseg)
    X_path = np.concatenate(xs) if xs else np.array([])
    Y_path = np.concatenate(ys) if ys else np.array([])
    plt.figure(figsize=(8, 6))
    plt.plot(X_path, Y_path)
    plt.scatter(*zip(*waypoints), s=20, marker="x", c="r")
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Constant-curvature reconstruction")
    plt.grid(True)
    plt.show()

# ---------------------------------------------------------------------------
# Utility: Print curvilinear table
# ---------------------------------------------------------------------------
def print_curvilinear_table(points, theta0: float = 0.0):
    """
    Compute and print ds, theta, delta_theta in table form.
    """
    ds, _, theta_vals, delta_theta_vals = cartesian_to_curvilinear(points, theta0)
    # Attempt pandas
    try:
        import pandas as pd
        df = pd.DataFrame({
            'ds': ds,
            'theta (rad)': theta_vals,
            'delta_theta (rad)': delta_theta_vals
        })
        print(df.to_markdown(index=False))
    except ImportError:
        print(f"{'ds':>12} {'theta(rad)':>15} {'delta_theta(rad)':>18}")
        for ds_i, th_i, dth_i in zip(ds, theta_vals, delta_theta_vals):
            print(f"{ds_i:12.6f} {th_i:15.6f} {dth_i:18.6f}")

# ---------------------------------------------------------------------------
# DEMO (if run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    waypoints = [
                [0,0],
                [10*20,10*5],
                [10*35,10*-2],
                [10*40,10*-10],
                [10*60,10*-1],
                [10*75,10*12],
                [10*100,10*4]
    ]
    print("Curvilinear metrics:")
    print_curvilinear_table(waypoints, theta0=0.0)
    # Optionally plot path
    plot_constant_curvature_path(waypoints, theta0=np.deg2rad(0))
#  [(0.000,	0.000),
# (0.974,	0.197),
# (1.793,	0.758),
# (2.330,	1.594),
# (2.548,	2.566),

#     (2.42029680308256, 3.55124104449957),
#     (1.87862298881003, 4.37944934185505),
#     (0.98805103465789, 4.8004473102870),
#     (0.01531694906669, 4.645057176509),
#     (-0.6613194701541, 3.936543970432),
#     (-0.7224028100308, 2.958742031085),
#     (-0.1392046213658, 2.171526889119),
#     (0.80662778377079, 1.89628380321236),
#     (1.80648642015421, 1.91309770369671),
#     (2.80634505653762, 1.92991160418106),
#     (3.80620369292104, 1.94672550466541),
#     (4.80606232930445, 1.96353940514976),
#     (5.80592096568787, 1.98035330563411),
#     (6.77601101601849, 2.19404202600435),
#     (7.61646660969246, 2.72897833897773),
#     (8.29987293534427, 3.45673303803298),
#     (8.86756050313839, 4.27947100954448),
#     (9.39363802051949, 5.12990763017304),
#     (9.9197155379006, 5.98034425080161),
#     (10.4457930552817, 6.83078087143017),
#     (10.9718705726628, 7.68121749205874),
#     (11.4979480900439, 8.5316541126873),
#     (12.1427814964098, 9.2910699025613),
#     (12.9832370900838, 9.82600621553469),
#     (13.9442394759708, 10.0886787839443),
#     (14.9399451730461, 10.0556238495589),
#     (15.8985159758368, 9.77667381106616),
#     (16.7825603646738, 9.31284558154178),
#     (17.5568342795376, 8.68263052880228),
#     (18.2655040538289, 7.97709020323189),
#     (18.9741738281201, 7.2715498776615),
#     (19.6828436024114, 6.56600955209111),
#     (20.0508222674035, 5.66468650489842),
#     (19.6606258944284, 4.77275695397686),
#     (18.7489423674068, 4.43125339638543),
#     (17.8687866802275, 4.84732730762725),
#     (18.9741738281201, 7.2715498776615),
#     (19.6828436024114, 6.56600955209111),
#     (20.0508222674035, 5.66468650489842),
#     (19.6606258944284, 4.77275695397686),
#     (18.7489423674068, 4.43125339638543),
#     (17.8687866802275, 4.84732730762725),
# ]