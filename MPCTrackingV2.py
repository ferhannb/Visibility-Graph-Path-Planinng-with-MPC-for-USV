"""
Curvature-bounded path optimiser + Path-tracking MPC
---------------------------------------------------
• one-shot   : opt.solve()
• MPC step   : ds_cmd, K_cmd = opt.mpc_step(x, y, theta)
"""

from __future__ import annotations
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from plotCurvelinearArc import cartesian_to_curvilinear, sample_arc

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
import matplotlib
matplotlib.use("TkAgg")




# ----------------------------------------------------------------------
#  Helper: generate a global reference sequence by sampling the full path
# ----------------------------------------------------------------------
def _uniform_reference_from_waypoints(wp: np.ndarray,
                                      ds_const: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples waypoint sequence along a single "s" axis with constant Delta s.
    Keeps theta consistent at each route break point.
    """
    # 1) Segment lengths and cumulative distance
    seg_L   = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum_L   = np.insert(np.cumsum(seg_L), 0, 0.0)
    L_tot   = cum_L[-1]

    # 2) Uniformly spaced s-grid
    s_grid  = np.arange(0.0, L_tot, ds_const)
    xs, ys, ths = [], [], []

    # 3) Find the segment corresponding to each s value
    for s in s_grid:
        idx = np.searchsorted(cum_L, s, side="right") - 1
        s0  = cum_L[idx]
        t   = (s - s0) / seg_L[idx]
        p0, p1 = wp[idx], wp[idx+1]
        pos = (1-t)*p0 + t*p1
        theta = np.arctan2(*(p1 - p0)[::-1])  # atan2(dy,dx)
        xs.append(pos[0]); ys.append(pos[1]); ths.append(theta)

    # 4) Always append final waypoint
    xs.append(wp[-1,0]); ys.append(wp[-1,1]); ths.append(ths[-1])
    return np.asarray(xs), np.asarray(ys), np.asarray(ths)


# ----------------------------------------------------------------------
#  Main class
# ----------------------------------------------------------------------
@dataclass
class PathOptimizer:
    # ---- kinematic base --------------------------------------------
    X_init: float
    Y_init: float
    theta_init: float

    # ---- target / path  --------------------------------------------
    target_type: str = "point"          # 'point' (single target) / 'trajectory'
    x_ref: Optional[Union[float, np.ndarray]] = None
    y_ref: Optional[Union[float, np.ndarray]] = None
    theta_ref: Optional[Union[float, np.ndarray]] = None
    waypoints: Optional[np.ndarray] = None   # (M,2) -> path tracking
    v_desired: float = 3.0
    dt: float = 0.3
    tol: float = 4.0                         # segment end tolerance

    # ---- discretization / bounds -----------------------------------
    N: int = 40
    ds_max: float = 5.0

    K_max: float = 1
    max_iter: int = 800

    # ---- weights ----------------------------------------------------
    w_pos: float = 40.0
    w_theta: float = 10.0
    w_ds_change: float = 1.0
    w_K_change: float = 1.0
    w_ds_total:  float = 0.0
    w_K_mag:     float = 0.0
    term_X: float = 120.0
    term_Y: float = 120.0
    term_theta: float = 80.0

    solver_opts: dict | None = field(default_factory=dict)

    # ----------------- internal buffers ------------------------------
    X_sol: np.ndarray | None = field(init=False, default=None)
    Y_sol: np.ndarray | None = field(init=False, default=None)
    TH_sol: np.ndarray | None = field(init=False, default=None)
    ds_sol: np.ndarray | None = field(init=False, default=None)
    K_sol: np.ndarray | None = field(init=False, default=None)

    # MPC logs for plotting
    mpc_X_hist: list[float] = field(init=False, default_factory=list, repr=False)
    mpc_Y_hist: list[float] = field(init=False, default_factory=list, repr=False)
    mpc_TH_hist: list[float] = field(init=False, default_factory=list, repr=False)
    mpc_ds_hist: list[float] = field(init=False, default_factory=list, repr=False)
    mpc_K_hist: list[float] = field(init=False, default_factory=list, repr=False)

        # --- global reference arrays ---
    x_ref_full: np.ndarray | None = field(init=False, default=None, repr=False)
    y_ref_full: np.ndarray | None = field(init=False, default=None, repr=False)
    th_ref_full: np.ndarray | None = field(init=False, default=None, repr=False)
    ref_index: int = field(init=False, default=0, repr=False)   # horizon start index

    # path-tracking state
    current_segment_idx: int = field(init=False, default=0, repr=False)

    # ---- symbolic handles (build_once) ------------------------------
    _built: bool = field(init=False, default=False, repr=False)
    _opti: ca.Opti = field(init=False, repr=False)
    _X: ca.MX = field(init=False, repr=False)
    _Y: ca.MX = field(init=False, repr=False)
    _TH: ca.MX = field(init=False, repr=False)
    _ds: ca.MX = field(init=False, repr=False)
    _K: ca.MX = field(init=False, repr=False)
    _x0_p: ca.MX = field(init=False, repr=False)
    _y0_p: ca.MX = field(init=False, repr=False)
    _th0_p: ca.MX = field(init=False, repr=False)
    _x_ref_p: ca.MX = field(init=False, repr=False)
    _y_ref_p: ca.MX = field(init=False, repr=False)
    _th_ref_p: ca.MX = field(init=False, repr=False)

    def __post_init__(self):
        if self.waypoints is not None:
            ds_const = self.v_desired * self.dt      # or round to a divisible step size
            (self.x_ref_full,
            self.y_ref_full,
            self.th_ref_full) = _uniform_reference_from_waypoints(
                                    self.waypoints, ds_const)


    # ================================================================
    #  MPC STEP  ------------------------------------------------------
    # ================================================================
    def mpc_step(self, x: float, y: float, theta: float) -> Tuple[float, float]:
        """Solve receding-horizon problem and return first control."""
        self._build_nlp()                       # no-op after first time
        self._update_state(x, y, theta)

        # --- generate reference if path tracking is active ------------
        # --- use precomputed global ref, advancing one step each call ---
        # ------------------------------------------------------------
        #  Updated reference block
        # ------------------------------------------------------------
        #  New global reference - projection-based horizon
        # ------------------------------------------------------------
        if self.x_ref_full is not None:
            N = self.N
            p_now = np.array([x, y])

            # 1) Find nearest reference index (across whole array)
            d2_all = (self.x_ref_full - p_now[0])**2 + (self.y_ref_full - p_now[1])**2
            self.ref_index = int(np.argmin(d2_all))

            # 2) Take horizon slice
            i0 = self.ref_index
            i1 = min(i0 + N + 1, len(self.x_ref_full))
            x_h  = self.x_ref_full[i0:i1]
            y_h  = self.y_ref_full[i0:i1]
            th_h = self.th_ref_full[i0:i1]

            # 3) If short, extend in direction of last segment
            if len(x_h) < N + 1:
                if len(x_h) >= 2:
                    dx = x_h[-1] - x_h[-2]
                    dy = y_h[-1] - y_h[-2]
                    if dx == dy == 0:
                        dx, dy = np.cos(th_h[-1]), np.sin(th_h[-1])
                else:
                    dx, dy = np.cos(th_h[-1]), np.sin(th_h[-1])

                while len(x_h) < N + 1:
                    x_h = np.append(x_h,  x_h[-1] + dx)
                    y_h = np.append(y_h,  y_h[-1] + dy)
                    th_h= np.append(th_h, th_h[-1])

            # 4) Load into CasADi parameters
            self._opti.set_value(self._x_ref_p, x_h)
            self._opti.set_value(self._y_ref_p, y_h)
            self._opti.set_value(self._th_ref_p, th_h)
            




        self._initialise_guesses()
        sol = self._opti.solve()
        self._cache_solution(sol)

        ds_cmd = float(sol.value(self._ds[0]))
        K_cmd = float(sol.value(self._K[0]))

        # log
        self.mpc_X_hist.append(x)
        self.mpc_Y_hist.append(y)
        self.mpc_TH_hist.append(theta)
        self.mpc_ds_hist.append(ds_cmd)
        self.mpc_K_hist.append(K_cmd)

        return ds_cmd, K_cmd

    # ================================================================
    #  One-shot solve (optional) --------------------------------------
    # ================================================================
    def solve(self, **state_override):
        self._build_nlp()
        self._update_state(state_override.get("X0", self.X_init),
                           state_override.get("Y0", self.Y_init),
                           state_override.get("TH0", self.theta_init))
        self._initialise_guesses()
        sol = self._opti.solve()
        self._cache_solution(sol)
        return (self.X_sol, self.Y_sol, self.TH_sol, self.ds_sol, self.K_sol)

    # ================================================================
    #  Build NLP model (once) -----------------------------------------
    # ================================================================
    def _build_nlp(self):
        if self._built:
            return

        opti = ca.Opti()

        X = opti.variable(self.N + 1)
        Y = opti.variable(self.N + 1)
        TH = opti.variable(self.N + 1)
        ds = opti.variable(self.N)
        K = opti.variable(self.N)

        x0_p = opti.parameter()
        y0_p = opti.parameter()
        th0_p = opti.parameter()

        # reference parameters (length N+1)
        x_ref_p = opti.parameter(self.N + 1)
        y_ref_p = opti.parameter(self.N + 1)
        th_ref_p = opti.parameter(self.N + 1)

        opti.subject_to([X[0] == x0_p,
                         Y[0] == y0_p,
                         TH[0] == th0_p])

        epsc, epst = 1e-4, 1e-3
        obj_pos = 0.0
        obj_th = 0.0
        obj_smooth = 0.0
        obj_path=0
        obj_Kmag=0
        obj_position_total=0
        for k in range(self.N):
            dtheta = K[k] * ds[k]             

            mid    = TH[k] + 0.5 * dtheta

            def sinc(z):
                return ca.if_else(ca.fabs(z) < 1e-4,
                                1 - z**2 / 6 + z**4 / 120, 
                                ca.sin(z)/z)
            s = sinc(0.5 * dtheta)

         
            opti.subject_to([X[k+1]  == X[k] + ds[k] * s * ca.cos(mid),
                             Y[k+1]  == Y[k] + ds[k] * s * ca.sin(mid),
                             TH[k+1] == TH[k] + dtheta])

            opti.subject_to([0 <= ds[k], ds[k] <= self.ds_max])
            opti.subject_to([-self.K_max <= K[k], K[k] <= self.K_max])


            # pos_err_int = (X[0] - self.x_ref) ** 2 + (Y[0] - self.y_ref) ** 2
            # pos_err = (X[k] - self.x_ref) ** 2 + (Y[k] - self.y_ref) ** 2
            # pert_err_pos = pos_err / pos_err_int
            # th_err = (TH[k] - self.theta_ref) ** 2
            # obj_theta_total    += th_err * self.w_theta * (1 - pert_err_pos)
            # obj_position_total += pert_err_pos * self.w_X

            pos_err = (X[k] - x_ref_p[k])**2 + (Y[k] - y_ref_p[k])**2
            th_err =  (TH[k] - th_ref_p[k])**2
            obj_pos += self.w_pos * pos_err
            obj_th += self.w_theta * th_err
            obj_path += self.w_ds_total * ds[k] 
            obj_Kmag += self.w_K_mag * K[k]**2           # curvature magnitude penalty

            if k < self.N - 1:
                obj_smooth += self.w_ds_change * (ds[k+1] - ds[k])**2
                obj_smooth += self.w_K_change  * (K[k+1] - K[k])**2

        # terminal
        # obj_pos += self.term_X * (X[-1] - x_ref_p[-1])**2 \
        #          + self.term_Y * (Y[-1] - y_ref_p[-1])**2
        # obj_th  += self.term_theta * (TH[-1] - th_ref_p[-1])**2

        opti.minimize(obj_pos + obj_th + obj_smooth + obj_path + obj_Kmag)

        opts = {"ipopt": {"print_level": 0,
                          "tol": 1e-6,
                          "max_iter": self.max_iter}}
        for sec, d in self.solver_opts.items():
            opts.setdefault(sec, {}).update(d)
        opti.solver("ipopt", opts)

        # store handles
        self._opti = opti
        self._X, self._Y, self._TH = X, Y, TH
        self._ds, self._K = ds, K
        self._x0_p, self._y0_p, self._th0_p = x0_p, y0_p, th0_p
        self._x_ref_p, self._y_ref_p, self._th_ref_p = x_ref_p, y_ref_p, th_ref_p
        self._built = True

        # initial reference values (auto-fill if empty)
        N1 = self.N + 1
        if self.target_type == "point":
            xr = np.full(N1, self.x_ref if self.x_ref is not None else self.X_init)
            yr = np.full(N1, self.y_ref if self.y_ref is not None else self.Y_init)
            tr = np.full(N1, self.theta_ref if self.theta_ref is not None else self.theta_init)

        else:  # 'trajectory' mode or waypoints
            # If incoming arrays are missing, use temporary zero vectors
            xr = np.zeros(N1) if self.x_ref is None else np.asarray(self.x_ref, dtype=float)
            yr = np.zeros(N1) if self.y_ref is None else np.asarray(self.y_ref, dtype=float)
            tr = np.zeros(N1) if self.theta_ref is None else np.asarray(self.theta_ref, dtype=float)

        # assign CasADi parameters
        opti.set_value(x_ref_p, xr)
        opti.set_value(y_ref_p, yr)
        opti.set_value(th_ref_p, tr)

    # ================================================================
    #  Helpers --------------------------------------------------------
    # ================================================================
    def _update_state(self, x, y, th):
        self._opti.set_value(self._x0_p, x)
        self._opti.set_value(self._y0_p, y)
        self._opti.set_value(self._th0_p, th)



    def _initialise_guesses(self):
        if self.ds_sol is None:           # cold-start
            self._opti.set_initial(self._ds, np.append([self.ds_max], np.zeros(self.N-1)))
            self._opti.set_initial(self._K, 0)
            self._opti.set_initial(self._X, self.X_init)
            self._opti.set_initial(self._Y, self.Y_init)
            self._opti.set_initial(self._TH, self.theta_init)
        else:                             # warm-start – shift & pad
            self._opti.set_initial(self._ds, np.r_[self.ds_sol[1:], self.ds_sol[-1]])
            self._opti.set_initial(self._K,  np.r_[self.K_sol[1:], self.K_sol[-1]])
            self._opti.set_initial(self._X,  np.r_[self.X_sol[1:], self.X_sol[-1]])
            self._opti.set_initial(self._Y,  np.r_[self.Y_sol[1:], self.Y_sol[-1]])
            self._opti.set_initial(self._TH, np.r_[self.TH_sol[1:], self.TH_sol[-1]])

    def _cache_solution(self, sol: ca.OptiSol):
        self.X_sol = np.array(sol.value(self._X))
        self.Y_sol = np.array(sol.value(self._Y))
        self.TH_sol = np.array(sol.value(self._TH))
        self.ds_sol = np.array(sol.value(self._ds))
        self.K_sol = np.array(sol.value(self._K))

    # ================================================================
    #  Path-tracking helper functions --------------------------------
    # ================================================================
    @staticmethod
    def compute_progress(current_pos: np.ndarray,
                         segment_start: np.ndarray,
                         segment_end: np.ndarray) -> float:
        path_vec = segment_end - segment_start
        denom = np.dot(path_vec, path_vec)
        if denom == 0:
            return 0.0
        return np.dot(current_pos - segment_start, path_vec) / denom

    def generate_reference(
        self,
        current_pos: np.ndarray,
        current_segment: Tuple[np.ndarray, np.ndarray],
        v_desired: float,
        dt: float,
        N: int
    ) -> np.ndarray:
        start, end = current_segment
        path_vec = end - start
        L = np.linalg.norm(path_vec)
        ref = np.zeros((3, N + 1))

        if L < 1e-3:          # very short segment
            ref[:2, :] = np.tile(current_pos, (N + 1, 1)).T
            ref[2, :] = 0.0
            return ref

        theta_ref = np.arctan2(path_vec[1], path_vec[0])
        s0 = self.compute_progress(current_pos, start, end)
        delta_s = (v_desired * dt) / L

        for k in range(N + 1):
            s_k = min(s0 + k * delta_s, 1.0)
            ref[:2, k] = start + s_k * path_vec
            ref[2, k] = theta_ref
        return ref
    
    def build_global_ref(self, step_size: float) -> Tuple[list, list, list]:
        """Precompute x/y/theta for the entire waypoint route."""
        x_full, y_full, th_full = [], [], []
        # loop over every waypoint segment
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            goal  = self.waypoints[i+1]
            # generate N+1 points along this segment
            ref = self.generate_reference(
                current_pos=start,
                current_segment=(start, goal),
                v_desired=self.v_desired,
                dt=self.dt,
                N=int(np.ceil(np.linalg.norm(goal-start) / (self.v_desired*self.dt)))
            )
            x_full   += list(ref[0, :])
            y_full   += list(ref[1, :])
            th_full  += list(ref[2, :])
        return x_full, y_full, th_full


    # ================================================================
    #  Simple plot - MPC result --------------------------------------
    # ================================================================
    def plot_mpc_trajectory(self):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=150) 
        if not self.mpc_X_hist:
            raise RuntimeError("mpc_step() was never called.")
        traj = np.column_stack([self.mpc_X_hist, self.mpc_Y_hist])
        prog = np.linspace(0, 1, traj.shape[0])

        fig, ax = plt.subplots(figsize=(7, 7))  # more square and compact area
        ax.plot(traj[:, 0], traj[:, 1], c="royalblue",  linewidth=2.5, label="Samples")

        # --- Curvilinear (smooth) path -------------------------------
        try:
            Xc, Yc = build_smooth_path(traj, theta0=self.theta_init)
            ax.plot(Xc, Yc, color="royalblue", linewidth=1.8,
                    alpha=0.85, label="Smooth curve")
        except Exception as e:
            print("[curve‑warn]", e)
        

        self.Goal = self.waypoints[-1]
        
        ax.scatter(self.Goal[0], self.Goal[1], marker="*", color="green", s=180, label="Goal")
        ax.scatter(traj[0, 0], traj[0, 1],  marker="X", color="red",   s=220, label="Start")
        if self.waypoints is not None:
            ax.plot(self.waypoints[:, 0], self.waypoints[:, 1], "r--o",
                    label="Waypoints")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.set_title("Path Smoothing using Curvelinear Model Equations")
        ax.legend()


        return fig, ax
    # ================================================================
    #  Simple plot - ds / K profile ----------------------------------
def plot_control_signals(ds_hist, K_hist, title_prefix="Optimal"):
    import numpy as np, matplotlib.pyplot as plt

    steps = np.arange(len(ds_hist))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # --- ds -----------------------------------------------------------------
    ax1.stem(steps, ds_hist, basefmt=" ")
    ax1.axhline(0, color='r')
    ax1.set_ylabel("ds [m]")
    ax1.set_title(f"{title_prefix} ds Control Signal")

    # --- K ------------------------------------------------------------------
    ax2.stem(steps, K_hist, linefmt='orange', markerfmt='o', basefmt=" ")
    ax2.axhline(0, color='r')
    ax2.set_ylabel("Curvature K [1/m]")
    ax2.set_xlabel("Step Number")
    ax2.set_title(f"{title_prefix} K Control Signal")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
#  Fills sparse (x,y) points with constant-curvature arc segments
# -----------------------------------------------------------
def build_smooth_path(points, theta0: float = 0.0, n_per_seg: int = 80):
    """
    points  : (N,2) ndarray or list
    theta0  : initial heading of the first segment [rad]
    returns : Xc, Yc (merged continuous curve)
    """
    import numpy as np
    ds, phi, theta_ends, _ = cartesian_to_curvilinear(points, theta0)

    x_all, y_all = [], []
    x0, y0 = points[0]
    for i, (ds_i, phi_i) in enumerate(zip(ds, phi)):
        th_start = theta0 if i == 0 else theta_ends[i-1]
        xseg, yseg = sample_arc(x0, y0, th_start, ds_i, phi_i, n=n_per_seg)
        if x_all:                               # avoid duplicating first point
            xseg, yseg = xseg[1:], yseg[1:]
        x_all.append(xseg); y_all.append(yseg)
        x0, y0 = points[i+1]
    return np.concatenate(x_all), np.concatenate(y_all)


# ----------------------------------------------------------------------
#  Demo - multi-waypoint path tracking
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # wp = np.array([[0, 0],
    #                [100, 0],
    #                [100, 50],
    #                [0, 50]], dtype=float)
    
    wp = np.array([[0,0], [90,60], [160,0], [240,60]])
    goal = wp[-1]  
    theta_start = np.deg2rad(0) 
    opt = PathOptimizer(
        X_init=0, Y_init=0, theta_init=theta_start,
        waypoints=wp, target_type="trajectory",
        v_desired=5, dt=1, N=25, ds_max=5.0
    )
    opt.K_max=0.03
    opt.w_pos=100
    opt.w_theta=40
    opt.w_ds_change=1000
    opt.w_K_change=70000
    opt.w_ds_total=0
    opt.w_K_mag=0
    


    # --- TestMain approach: build reference once with constant Delta s ---
    ds_const = opt.v_desired * opt.dt              # Delta s = v_desired * dt
    opt.x_ref_full, opt.y_ref_full, opt.th_ref_full = \
            _uniform_reference_from_waypoints(opt.waypoints, ds_const)  # new
    opt.ref_index = 0

    x, y, th = 0, 0.0, 0
    ds_hist, K_hist = [], []  
    for _ in range(300):

        ds_cmd, K_cmd = opt.mpc_step(x, y, th)
        ds_hist.append(ds_cmd)       
        K_hist.append(K_cmd)          
        dth = K_cmd * ds_cmd
        th += dth
        if abs(dth) < 1e-4:
            x += np.cos(th) * ds_cmd
            y += np.sin(th) * ds_cmd
        else:
            R = 1.0 / K_cmd
            x += R * (np.sin(th) - np.sin(th - dth))
            y += -R * (np.cos(th) - np.cos(th - dth))

                # ---- EXIT IF TARGET REACHED
        if np.hypot(x - goal[0], y - goal[1]) < 0.5:
            print(f"[MPC] Zig-zag target reached.")
            break

    opt.plot_mpc_trajectory()
    plot_control_signals(ds_hist, K_hist)
    plt.show()

# ----------------------------------------------------------------------
#  Demo - "minimum rudder angle change" test
# ----------------------------------------------------------------------

# =============================================================================
#  Demo - "minimum rudder-change (Delta K) cost" comparison
# =============================================================================
# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # -------------------------------------------------------------------------
#     # 1) S-shaped route (ideal to observe rudder changes)
#     wp = np.array([
#         [0.0,  0.0],
#         [30.0, 0.0],
#         [60.0, 20.0],
#         [90.0, 0.0]
#     ])

#     # Initial state (tangent to first segment)
#     dx0, dy0 = wp[1] - wp[0]
#     theta0   = np.arctan2(dy0, dx0)
#     X0, Y0   = wp[0]

#     # -------------------------------------------------------------------------
#     # 2) Shared MPC parameters
#     v_des, dt, N = 2.0, 0.3, 25
#     ds_const = v_des * dt            # uniform reference spacing
#     K_max    = 0.8                   # [1/m]

#     base_kwargs = dict(
#         X_init      = X0,
#         Y_init      = Y0,
#         theta_init  = theta0,
#         waypoints   = wp,
#         v_desired   = v_des,
#         dt          = dt,
#         N           = N,
#         K_max       = K_max,
#         ds_max      = 2.0,           # small step -> flexibility in turns
#         w_pos       = 10.0,
#         w_theta     = 40.0,
#         w_K_mag     = 0.5            # mild penalty on |K| magnitude
#     )

#     # -------------------------------------------------------------------------
#     # 3) Two scenarios
#     #    A) no Delta K penalty - reference
#     #    B) high Delta K penalty - smoother rudder
#     opt_A = PathOptimizer(**base_kwargs, w_K_change=0.0)
#     opt_B = PathOptimizer(**base_kwargs, w_K_change=10.0)

#     # -------------------------------------------------------------------------
#     # 4) One-shot global reference (uniform Delta s)
#     for opt in (opt_A, opt_B):
#         opt.x_ref_full, opt.y_ref_full, opt.th_ref_full = \
#             _uniform_reference_from_waypoints(wp, ds_const)
#         opt.ref_index = 0

#     # -------------------------------------------------------------------------
#     # 5) Basic single-track simulation
#     def simulate(opt):
#         x, y, th = X0, Y0, theta0
#         K_hist = []
#         for _ in range(300):
#             ds_cmd, K_cmd = opt.mpc_step(x, y, th)
#             K_hist.append(K_cmd)

#             dth = K_cmd * ds_cmd
#             R   = float('inf') if abs(K_cmd) < 1e-9 else 1.0 / K_cmd
#             x  += R * (np.sin(th + dth) - np.sin(th))
#             y  += -R * (np.cos(th + dth) - np.cos(th))
#             th += dth

#             # stop if within 0.5 m of target
#             if np.hypot(x - wp[-1,0], y - wp[-1,1]) < 0.5:
#                 break
#         return np.asarray(K_hist)

#     K_A = simulate(opt_A)
#     K_B = simulate(opt_B)

#     # -------------------------------------------------------------------------
#     # 6) Delta K RMSE output
#     rmse = lambda K: np.sqrt(np.mean(np.diff(K)**2))
#     print(f"ΔK RMSE  (w_K_change = 0)  : {rmse(K_A):.4f}")
#     print(f"ΔK RMSE  (w_K_change = 10) : {rmse(K_B):.4f}")

#     # -------------------------------------------------------------------------
#     # 7) K(t) plot
#     plt.figure(figsize=(8,4))
#     plt.plot(K_A, label="K(t) - reference (w_K_change=0)")
#     plt.plot(K_B, label="K(t) - smooth (w_K_change=10)", alpha=0.8)
#     plt.title("Curvature history comparison")
#     plt.xlabel("MPC step"); plt.ylabel("K  [1/m]")
#     plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
