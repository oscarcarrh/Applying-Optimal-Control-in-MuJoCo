#!/usr/bin/env python3
"""Run MuJoCo quadruped simulation with a WebSocket bridge to the browser dashboard.

Usage:
    pip install websockets
    python examples/run_web.py                         # LQG, mini_cheetah, impulse
    python examples/run_web.py --controller mpc        # MPC controller
    python examples/run_web.py --controller all         # hot-switch from browser
    python examples/run_web.py --no-render              # headless (browser-only view)

Then open  web/index.html  in a browser (or http://localhost:8765 for the WS).

Arrow keys in the browser control the robot.  The MuJoCo viewer (if enabled)
shows the ground-truth physics while the dashboard shows telemetry + top-down.
"""

import sys, os, json, time, argparse, threading, asyncio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import websockets
    import websockets.asyncio.server as ws_server
except ImportError:
    print("Install websockets:  pip install websockets")
    sys.exit(1)

from gym_quadruped.quadruped_env import QuadrupedEnv
from src.dynamics import QuadrupedDynamics
from src.estimator_ekf import OrientationEKF
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.controller_mpc import MPCController

# ── constants (same as run_mujoco.py) ──────────────────────────────────
ROBOT_MASS = 9.0
ROBOT_INERTIA = np.diag([0.107, 0.098, 0.024])
ROBOT_HIP_HEIGHT = 0.225
ROBOT_FOOT_OFFSET = np.array([
    [0.19,  0.111, -0.225],
    [0.19, -0.111, -0.225],
    [-0.19,  0.111, -0.225],
    [-0.19, -0.111, -0.225],
])

# ── shared state (sim ↔ websocket) ────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.controller_name = "lqg"
        self.disturbance_type = "impulse"
        self.reset_requested = False
        self.quit = False
        # Outgoing (sim → browser)
        self.frame = {}

shared = SharedState()

# ── helpers (identical to run_mujoco.py) ──────────────────────────────
def get_state(env) -> np.ndarray:
    p = env.base_pos.copy()
    v = env.base_lin_vel(frame="world")
    rpy = env.base_ori_euler_xyz.copy()
    omega = env.base_ang_vel(frame="base")
    return np.concatenate([p, v, rpy, omega])

def grf_to_torques(env, grfs, contact):
    tau = np.zeros(env.mjModel.nu)
    try:
        jacobians = env.feet_jacobians(frame="world")
    except Exception:
        return tau
    for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
        if not contact[i]:
            continue
        J_full = jacobians[leg]
        leg_idx = env.legs_qvel_idx[leg]
        J_leg = J_full[:, leg_idx]
        tau_leg = -J_leg.T @ grfs[3*i:3*i+3]
        tau[env.legs_tau_idx[leg]] = tau_leg
    return tau

def get_contacts(env):
    try:
        cs, _ = env.feet_contact_state()
        return np.array([cs.FL, cs.FR, cs.RL, cs.RR], dtype=bool)
    except Exception:
        return np.ones(4, dtype=bool)

def get_feet_world(env):
    try:
        fp = env.feet_pos(frame="world")
        return np.array([fp.FL, fp.FR, fp.RL, fp.RR])
    except Exception:
        return None

# ── dynamics / controllers ────────────────────────────────────────────
def build_dynamics():
    dyn = QuadrupedDynamics(mass=ROBOT_MASS, inertia=ROBOT_INERTIA, dt=0.002)
    dyn.r_feet_body = ROBOT_FOOT_OFFSET.copy()
    return dyn

def build_cost():
    Q = np.diag([80,80,400, 8,8,40, 150,150,30, 1,1,4])
    R = np.eye(12) * 1e-4
    return Q, R, Q * 5

def build_ref(dyn, vx=0., vy=0., wz=0.):
    x = dyn.standing_state(height=ROBOT_HIP_HEIGHT)
    x[3:6] = [vx, vy, 0.]
    x[9:12] = [0., 0., wz]
    return x

def build_controller(name, dyn, Q, R, Qf, x_ref):
    Ad, Bd, gd = dyn.get_linear_system(x_ref)
    Ac, Bc = dyn.continuous_AB(x_ref)
    if name == "pmp":
        c = PontryaginController(A=Ac, B=Bc, Q_s=Q, R_u=R, Q_f=Qf,
                                  g_aff=dyn.gravity_vector()/dyn.dt,
                                  dt=dyn.dt, horizon=500)
        c.solve_discrete_sweep(x_ref.copy(), x_ref)
        return c
    if name == "lqg":
        c = LQGController(A_d=Ad, B_d=Bd, g_d=gd, Q=Q*dyn.dt, R=R*dyn.dt,
                           Q_proc=np.diag([1e-3]*3+[1e-2]*3+[5e-3]*3+[1e-2]*3),
                           R_meas=np.diag([5e-3]*3+[2e-2]*3+[1e-2]*3+[5e-2]*3))
        c.set_initial_estimate(x_ref)
        return c
    if name == "mpc":
        return MPCController(A_d=Ad, B_d=Bd, g_d=gd, Q=Q*dyn.dt, R=R*dyn.dt,
                              Q_f=Qf*dyn.dt, N=10, mu=0.6, fz_max=150.)
    raise ValueError(name)

# ── WebSocket server ──────────────────────────────────────────────────
WS_PORT = 8765

async def ws_handler(websocket):
    print(f"  [WS] client connected")
    try:
        async for msg in websocket:
            data = json.loads(msg)
            with shared.lock:
                if "vx" in data: shared.cmd_vx = float(data["vx"])
                if "vy" in data: shared.cmd_vy = float(data["vy"])
                if "wz" in data: shared.cmd_wz = float(data["wz"])
                if "controller" in data: shared.controller_name = data["controller"]
                if "disturbance" in data: shared.disturbance_type = data["disturbance"]
                if data.get("reset"): shared.reset_requested = True
    except websockets.exceptions.ConnectionClosed:
        pass
    print(f"  [WS] client disconnected")

async def ws_broadcast(stop_event):
    """Broadcast sim frames to all connected clients."""
    clients = set()

    async def register(websocket):
        clients.add(websocket)
        try:
            await ws_handler(websocket)
        finally:
            clients.discard(websocket)

    async with ws_server.serve(register, "0.0.0.0", WS_PORT):
        print(f"  [WS] server listening on ws://localhost:{WS_PORT}")
        while not stop_event.is_set():
            with shared.lock:
                frame = shared.frame.copy() if shared.frame else None
            if frame and clients:
                msg = json.dumps(frame)
                dead = set()
                for ws in clients:
                    try:
                        await ws.send(msg)
                    except Exception:
                        dead.add(ws)
                clients -= dead
            await asyncio.sleep(0.03)  # ~33 Hz broadcast

def start_ws_thread():
    stop = threading.Event()
    def run():
        asyncio.run(ws_broadcast(stop))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return stop, t

# ── Simulation loop ──────────────────────────────────────────────────
def run_sim(robot_name, render, duration, allow_switch):
    print(f"\n{'='*60}")
    print(f"  WebSocket Quadruped Control")
    print(f"  Robot:  {robot_name}   Render: {render}   Duration: {duration}s")
    print(f"  Dashboard: open  web/index.html  in your browser")
    print(f"{'='*60}\n")

    ws_stop, ws_thread = start_ws_thread()
    time.sleep(0.3)

    env = QuadrupedEnv(
        robot=robot_name, scene="flat", sim_dt=0.002,
        base_vel_command_type="human",
        state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
    )
    _ = env.reset(random=False)
    if render:
        env.render()

    dyn = build_dynamics()
    Q, R, Qf = build_cost()

    current_ctrl_name = "lqg"
    x_ref = build_ref(dyn)
    u_ref = dyn.standing_control()
    controller = build_controller(current_ctrl_name, dyn, Q, R, Qf, x_ref)
    ori_ekf = OrientationEKF(dt=env.mjModel.opt.timestep)

    sim_dt = env.mjModel.opt.timestep
    ctrl_dt = 0.01
    ctrl_steps = max(1, int(ctrl_dt / sim_dt))
    n_steps = int(duration / sim_dt)
    current_grfs = u_ref.copy()

    log_pos_err = []
    log_vel_err = []
    log_grf = []

    print(f"  Sim dt={sim_dt}s  ctrl={1/ctrl_dt:.0f}Hz  steps={n_steps}")
    print("  Waiting for browser connection...\n")

    try:
        for step in range(n_steps):
            if shared.quit:
                break
            t = step * sim_dt

            x = get_state(env)
            contact = get_contacts(env)
            r_feet = get_feet_world(env)

            # Read commands from browser
            with shared.lock:
                cmd_vx = shared.cmd_vx
                cmd_vy = shared.cmd_vy
                cmd_wz = shared.cmd_wz
                want_ctrl = shared.controller_name
                dist_type = shared.disturbance_type
                if shared.reset_requested:
                    shared.reset_requested = False
                    _ = env.reset(random=False)
                    if render: env.render()
                    current_grfs = u_ref.copy()
                    log_pos_err.clear(); log_vel_err.clear(); log_grf.clear()
                    continue

            # Hot-switch controller
            if allow_switch and want_ctrl != current_ctrl_name:
                try:
                    x_ref_tmp = build_ref(dyn, cmd_vx, cmd_vy, cmd_wz)
                    controller = build_controller(want_ctrl, dyn, Q, R, Qf, x_ref_tmp)
                    current_ctrl_name = want_ctrl
                    print(f"  [t={t:.1f}] Switched to {current_ctrl_name.upper()}")
                except Exception as e:
                    print(f"  Switch failed: {e}")

            x_ref = build_ref(dyn, cmd_vx, cmd_vy, cmd_wz)

            # Disturbance
            dist = np.zeros(6)
            if dist_type == "impulse" and 2.0 <= t < 2.15:
                dist = np.array([50., 25., 0., 0., 0., 5.])
            elif dist_type == "persistent" and t >= 2.0:
                dist = np.array([15., 8., 0., 0., 0., 2.])
            env.mjData.qfrc_applied[:6] = dist

            # EKF
            gyro = env.base_ang_vel(frame="base")
            acc_w = env.base_lin_acc(frame="world")
            R_WB = env.base_configuration[0:3, 0:3]
            acc_b = R_WB.T @ (acc_w - np.array([0., 0., -9.81]))
            ori_ekf.predict(gyro)
            ori_ekf.update_accel(acc_b)

            # Control
            if step % ctrl_steps == 0:
                try:
                    if current_ctrl_name == "lqg":
                        y = x + np.random.randn(12) * np.array(
                            [5e-3]*3 + [2e-2]*3 + [1e-2]*3 + [5e-2]*3)
                        current_grfs = controller.step(y, x_ref, u_ref)
                    else:
                        current_grfs = controller.compute_control(x=x, x_ref=x_ref, u_ref=u_ref)
                except Exception:
                    current_grfs = u_ref.copy()
                current_grfs = np.clip(current_grfs, -150., 150.)
                for i in range(4):
                    if not contact[i]:
                        current_grfs[3*i:3*i+3] = 0.

            tau = grf_to_torques(env, current_grfs, contact)
            _, _, terminated, _, _ = env.step(action=tau)
            if render:
                env.render()

            # Telemetry
            pos_err = float(np.linalg.norm(x[:3] - x_ref[:3]))
            vel_err = float(np.linalg.norm(x[3:6] - x_ref[3:6]))
            grf_n = float(np.linalg.norm(current_grfs))
            log_pos_err.append(pos_err)
            log_vel_err.append(vel_err)
            log_grf.append(grf_n)
            MAX_HIST = 200
            if len(log_pos_err) > MAX_HIST:
                log_pos_err.pop(0); log_vel_err.pop(0); log_grf.pop(0)

            # Broadcast at ~33 Hz
            if step % max(1, int(0.03 / sim_dt)) == 0:
                with shared.lock:
                    shared.frame = {
                        "t": round(t, 3),
                        "x": [round(float(v), 5) for v in x],
                        "u": [round(float(v), 2) for v in current_grfs],
                        "cmd": [round(cmd_vx, 3), round(cmd_vy, 3), round(cmd_wz, 3)],
                        "dist": [round(float(v), 1) for v in dist],
                        "contact": contact.tolist(),
                        "ctrl": current_ctrl_name,
                        "dist_type": dist_type,
                        "pos_err": round(pos_err, 5),
                        "vel_err": round(vel_err, 5),
                        "grf_norm": round(grf_n, 1),
                        "hist_pos": [round(v, 4) for v in log_pos_err[-100:]],
                        "hist_vel": [round(v, 4) for v in log_vel_err[-100:]],
                        "hist_grf": [round(v, 1) for v in log_grf[-100:]],
                    }

            if step % int(1.0 / sim_dt) == 0:
                print(f"  t={t:5.1f}s | h={x[2]:.3f} | vx={x[3]:+.3f} | "
                      f"cmd=({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f}) | "
                      f"ctrl={current_ctrl_name} | dist={dist_type}")

            if terminated:
                _ = env.reset(random=False)
                if render: env.render()

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        ws_stop.set()
        env.close()
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="all", choices=["pmp","lqg","mpc","all"])
    parser.add_argument("--robot-name", default="mini_cheetah")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--disturbance", default="impulse", choices=["impulse","persistent","none"])
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    allow_switch = args.controller == "all"
    with shared.lock:
        shared.controller_name = "lqg" if allow_switch else args.controller
        shared.disturbance_type = args.disturbance

    run_sim(
        robot_name=args.robot_name,
        render=not args.no_render,
        duration=args.duration,
        allow_switch=allow_switch,
    )
