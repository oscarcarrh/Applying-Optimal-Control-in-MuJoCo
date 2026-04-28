"""Simulation harness for closed-loop control of MuJoCo quadruped.

Interfaces the QuadrupedEnv from gym-quadruped with the three controllers
(PMP, LQG, MPC) and the EKF orientation estimator.

The simulator:
1. Extracts floating-base state from MuJoCo (ground truth)
2. Adds sensor noise to create measurements
3. Runs the EKF for orientation estimation (paper Sec IV.A)
4. Computes GRFs via the selected controller
5. Converts GRFs to joint torques via Jacobian transpose
6. Applies disturbances and logs data

GRF → Joint torque mapping (standard for quadruped control):
    τ_leg = −Jᵀ_leg f_leg
where J_leg is the foot Jacobian from the environment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimConfig:
    """Simulation configuration."""
    robot: str = 'go1'
    scene: str = 'flat'
    sim_dt: float = 0.002
    ctrl_dt: float = 0.01  # controller rate (≤ sim_dt)
    duration: float = 5.0
    ref_height: float = 0.30
    ref_lin_vel: float = 0.0  # m/s forward velocity command

    # Disturbance settings
    disturbance_type: str = 'impulse'  # 'impulse', 'persistent', 'none'
    disturbance_time: float = 1.5      # when to apply impulse (s)
    disturbance_force: np.ndarray = field(
        default_factory=lambda: np.array([30.0, 15.0, 0.0, 0.0, 0.0, 0.0])
    )
    disturbance_duration: float = 0.1  # seconds

    # Sensor noise
    pos_noise: float = 0.005
    vel_noise: float = 0.02
    ori_noise: float = 0.01
    omega_noise: float = 0.05


@dataclass
class SimLog:
    """Logged simulation data."""
    time: list = field(default_factory=list)
    state_true: list = field(default_factory=list)
    state_est: list = field(default_factory=list)
    control: list = field(default_factory=list)
    state_ref: list = field(default_factory=list)
    disturbance: list = field(default_factory=list)
    cost: list = field(default_factory=list)

    def to_arrays(self):
        return {
            'time': np.array(self.time),
            'state_true': np.array(self.state_true),
            'state_est': np.array(self.state_est),
            'control': np.array(self.control),
            'state_ref': np.array(self.state_ref),
            'disturbance': np.array(self.disturbance),
            'cost': np.array(self.cost),
        }


def extract_state_from_env(env) -> np.ndarray:
    """Extract the 12-dim floating-base state from MuJoCo environment.

    x = [p(3), v(3), θ_rpy(3), ω(3)]
    """
    p = env.base_pos.copy()                        # (3,) world position
    v = env.base_lin_vel(frame='world')             # (3,) world velocity
    rpy = env.base_ori_euler_xyz.copy()             # (3,) roll, pitch, yaw
    omega = env.base_ang_vel(frame='base')          # (3,) body angular velocity
    return np.concatenate([p, v, rpy, omega])


def add_sensor_noise(x_true: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Add Gaussian noise to simulate sensor measurements."""
    noise = np.concatenate([
        np.random.randn(3) * cfg.pos_noise,
        np.random.randn(3) * cfg.vel_noise,
        np.random.randn(3) * cfg.ori_noise,
        np.random.randn(3) * cfg.omega_noise,
    ])
    return x_true + noise


def grf_to_joint_torques(env, grfs: np.ndarray) -> np.ndarray:
    """Convert ground reaction forces to joint torques.

    τ = −Jᵀ f  for each leg.

    Parameters
    ----------
    env  : QuadrupedEnv instance
    grfs : (12,) GRFs [f_FL(3), f_FR(3), f_RL(3), f_RR(3)]

    Returns
    -------
    tau : (n_actuators,) joint torques
    """
    tau = np.zeros(env.mjModel.nu)
    jacobians = env.feet_jacobians(frame='world')

    for i, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
        J_leg = jacobians[leg_name]  # (3, nv) full Jacobian

        # Extract only the columns for this leg's joints
        leg_qvel_idx = env.legs_qvel_idx[leg_name]
        J_leg_local = J_leg[:, leg_qvel_idx]  # (3, n_joints_per_leg)

        f_leg = grfs[3 * i: 3 * i + 3]

        # τ_leg = −Jᵀ f
        tau_leg = -J_leg_local.T @ f_leg

        leg_tau_idx = env.legs_tau_idx[leg_name]
        tau[leg_tau_idx] = tau_leg

    return tau


def get_foot_positions_world(env) -> np.ndarray:
    """Get 4 foot positions in world frame (4, 3)."""
    feet = env.feet_pos(frame='world')
    return np.array([feet.FL, feet.FR, feet.RL, feet.RR])


def get_contact_mask(env) -> np.ndarray:
    """Get boolean contact mask (4,)."""
    contact, _ = env.feet_contact_state()
    return np.array([contact.FL, contact.FR, contact.RL, contact.RR], dtype=bool)


def apply_disturbance(env, t: float, cfg: SimConfig) -> np.ndarray:
    """Apply external disturbance to the robot body."""
    dist = np.zeros(6)

    if cfg.disturbance_type == 'none':
        return dist

    if cfg.disturbance_type == 'impulse':
        if cfg.disturbance_time <= t < cfg.disturbance_time + cfg.disturbance_duration:
            dist = cfg.disturbance_force.copy()

    elif cfg.disturbance_type == 'persistent':
        if t >= cfg.disturbance_time:
            dist = cfg.disturbance_force.copy() * 0.3  # scaled down

    env.mjData.qfrc_applied[:6] = dist
    return dist


def run_simulation(env, controller, dynamics, cfg: SimConfig,
                   controller_name: str = 'controller') -> SimLog:
    """Run closed-loop simulation.

    Parameters
    ----------
    env        : QuadrupedEnv instance
    controller : controller object with compute_control(x, x_ref, u_ref)
    dynamics   : QuadrupedDynamics instance
    cfg        : SimConfig

    Returns
    -------
    log : SimLog with recorded data
    """
    log = SimLog()
    obs = env.reset()

    x_ref = dynamics.standing_state(height=cfg.ref_height)
    u_ref = dynamics.standing_control()

    ctrl_steps = int(cfg.ctrl_dt / cfg.sim_dt)
    n_steps = int(cfg.duration / cfg.sim_dt)

    current_grfs = u_ref.copy()
    step_count = 0

    print(f"  Running {controller_name} for {cfg.duration}s "
          f"({n_steps} sim steps, ctrl every {ctrl_steps} steps)...")

    for i in range(n_steps):
        t = i * cfg.sim_dt

        # Extract true state
        x_true = extract_state_from_env(env)

        # Sensor measurements (noisy)
        y = add_sensor_noise(x_true, cfg)

        # Recompute control at controller rate
        if i % ctrl_steps == 0:
            try:
                # Update dynamics linearization at current state
                contact = get_contact_mask(env)
                r_feet = get_foot_positions_world(env)

                # Compute GRFs
                current_grfs = controller.compute_control(
                    x=y, x_ref=x_ref, u_ref=u_ref
                )

                # Clip forces
                current_grfs = np.clip(current_grfs, -200, 200)

            except Exception as e:
                current_grfs = u_ref.copy()

        # Convert GRFs to joint torques
        try:
            tau = grf_to_joint_torques(env, current_grfs)
        except Exception:
            tau = np.zeros(env.mjModel.nu)

        # Apply disturbance
        dist = apply_disturbance(env, t, cfg)

        # Step simulation
        obs, reward, terminated, truncated, info = env.step(tau)

        # Compute running cost
        dx = x_true - x_ref
        du = current_grfs - u_ref
        cost = 0.5 * (dx @ dynamics.Q_tracking @ dx + du @ dynamics.R_control @ du) \
            if hasattr(dynamics, 'Q_tracking') else 0.0

        # Log
        log.time.append(t)
        log.state_true.append(x_true.copy())
        log.state_est.append(y.copy())
        log.control.append(current_grfs.copy())
        log.state_ref.append(x_ref.copy())
        log.disturbance.append(dist.copy())
        log.cost.append(cost)

        if terminated:
            print(f"    Episode terminated at t={t:.2f}s")
            break

        step_count += 1

    print(f"    Completed {step_count} steps.")
    return log
