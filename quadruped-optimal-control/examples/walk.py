#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv
from src.gait_scheduler import TrotGaitScheduler, LEG_ORDER
from src.foot_trajectory import JointSpaceTrotPlanner
from src.trajectory_generator import WaypointTrajectory, build_trajectory

# ===== CONTROL ÓPTIMO =====
try:
    from run_mujoco import (
        build_dynamics,
        build_cost_matrices,
        build_controller,
        build_reference_state,
        get_state,
        get_contacts,
        grf_to_torques,
        ROBOT_HIP_HEIGHT,
    )
    HAS_OPT = True
except Exception as e:
    HAS_OPT = False
    OPT_ERROR = e


# UTILIDADES
def get_joint_states(env):
    q = env.mjData.qpos[-12:].reshape(4, 3)
    dq = env.mjData.qvel[-12:].reshape(4, 3)

    qd = {leg: q[i].copy() for i, leg in enumerate(LEG_ORDER)}
    dqd = {leg: dq[i].copy() for i, leg in enumerate(LEG_ORDER)}

    return qd, dqd


def pd_control(env, q_des, q, dq, kp, kd):
    tau = np.zeros(env.mjModel.nu)

    for leg in LEG_ORDER:
        tau_leg = kp[leg] * (q_des[leg] - q[leg]) - kd[leg] * dq[leg]
        tau[env.legs_tau_idx[leg]] = tau_leg

    return tau


def build_gain(kp_val, kd_val):
    return (
        {leg: kp_val.copy() for leg in LEG_ORDER},
        {leg: kd_val.copy() for leg in LEG_ORDER},
    )


# PI CONTROL 
class PositionPI:
    def __init__(self, kp=0.25, ki=0.02):
        self.kp = kp
        self.ki = ki
        self.ix = 0.0
        self.iy = 0.0

    def reset(self):
        self.ix = 0.0
        self.iy = 0.0

    def update(self, ex, ey, dt):
        self.ix = np.clip(self.ix + ex * dt, -0.5, 0.5)
        self.iy = np.clip(self.iy + ey * dt, -0.5, 0.5)

        vx = self.kp * ex + self.ki * self.ix
        vy = self.kp * ey + self.ki * self.iy

        return vx, vy


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# CONTROL ÓPTIMO
def compute_opt_tau(env, controller, u_ref, vx, vy, wz, dyn):
    try:
        x = get_state(env)
        contact = get_contacts(env)

        x_ref = build_reference_state(
            dyn,
            height=ROBOT_HIP_HEIGHT,
            vx=vx,
            vy=vy,
            wz=wz,
        )

        grf = controller.compute_control(x, x_ref, u_ref)
        grf = np.clip(np.array(grf).reshape(-1), -50, 50)

        for i in range(4):
            if not contact[i]:
                grf[3*i:3*i+3] = 0.0

        return grf_to_torques(env, grf, contact)

    except:
        return np.zeros(env.mjModel.nu)


# ===============================
# MAIN
# ===============================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller", default="pi",
                        choices=["pi", "mpc", "lqg", "pmp"])
    parser.add_argument("--traj", default="line",
                        choices=["line", "square", "zigzag", "circle"])
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--vref", type=float, default=0.04)

    args = parser.parse_args()

    env = QuadrupedEnv(robot="mini_cheetah", scene="flat", sim_dt=0.002)
    env.reset(random=False)
    env.render()

    dt = env.mjModel.opt.timestep
    steps = int(args.duration / dt)

    x0, y0 = env.base_pos[0], env.base_pos[1]

    traj = WaypointTrajectory(build_trajectory(args.traj), speed=args.vref)

    gait = TrotGaitScheduler(period=0.8, duty_factor=0.6)

    q_nom, _ = get_joint_states(env)
    planner = JointSpaceTrotPlanner(q_nom)

    kp, kd = build_gain(
        np.array([45.0, 60.0, 60.0]),
        np.array([2.5, 3.5, 3.5])
    )

    pi = PositionPI()

    opt_ctrl = None
    u_ref = None
    dyn = None

    if args.controller != "pi" and HAS_OPT:
        dyn = build_dynamics()
        Q, R, Qf = build_cost_matrices()
        x_ref0 = build_reference_state(dyn, height=ROBOT_HIP_HEIGHT, vx=0, vy=0, wz=0)
        u_ref = dyn.standing_control()
        opt_ctrl = build_controller(args.controller, dyn, Q, R, Qf, x_ref0)

    tau_opt = np.zeros(env.mjModel.nu)

    # ================= LOOP =================
    for i in range(steps):
        t = i * dt

        q, dq = get_joint_states(env)

        pos, vel, _, omega, _ = traj.sample(t)

        x = env.base_pos[0] - x0
        y = env.base_pos[1] - y0

        ex, ey = pos[0] - x, pos[1] - y
        dvx, dvy = pi.update(ex, ey, dt)

        #  FASES ESTABLES
        if t < 2.5:
            s = smoothstep(t / 2.5)

            q_init = np.array([0.0, -1.57, 0.0])
            q_stand = np.array([0.0, -1.0, 2.1])

            q_des = {
                leg: (1 - s) * q_init + s * q_stand
                for leg in LEG_ORDER
            }

            tau = pd_control(env, q_des, q, dq, kp, kd)

        elif t < 4.0:
            q_des = {leg: np.array([0.0, -1.0, 2.1]) for leg in LEG_ORDER}
            tau = pd_control(env, q_des, q, dq, kp, kd)

        else:
            vx = np.clip(vel[0] + dvx, -0.10, 0.10)
            vy = np.clip(vel[1] + dvy, -0.05, 0.05)
            wz = omega[2]

            q_des = planner.get_joint_targets(t, gait, vx, vy, wz)
            tau = pd_control(env, q_des, q, dq, kp, kd)

        # ===== MPC SOLO CUANDO YA ESTÁ PARADO =====
        if args.controller != "pi" and t > 4.0 and i % 5 == 0:
            tau_opt = compute_opt_tau(env, opt_ctrl, u_ref, vx, vy, wz, dyn)

        if args.controller != "pi" and t > 4.0:
            tau += args.alpha * tau_opt * 0.3

        tau = np.clip(tau, -16, 16)

        env.step(tau)

        if i % 10 == 0:
            env.render()

        if i % int(1/dt) == 0:
            z = env.base_pos[2]
            err = np.sqrt(ex**2 + ey**2)
            print(f"t={t:.1f} | err={err:.3f} | z={z:.3f}")

    env.close()


if __name__ == "__main__":
    main()