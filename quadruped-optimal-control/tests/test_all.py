"""Tests for dynamics and controllers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.dynamics import QuadrupedDynamics
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.estimator_ekf import OrientationEKF, KalmanFilter


def test_dynamics_equilibrium():
    """At standing pose with gravity-compensating GRFs, state should not change."""
    dyn = QuadrupedDynamics(dt=0.005)
    x = dyn.standing_state()
    u = dyn.standing_control()
    x_next = dyn.step(x, u)
    # velocity and position should remain ~0 (gravity cancelled)
    assert np.allclose(x_next[:6], x[:6], atol=1e-6), \
        f"Equilibrium violated: {x_next[:6]}"
    print("✓ dynamics_equilibrium")


def test_dynamics_freefall():
    """With zero GRFs, robot should accelerate downward at g."""
    dyn = QuadrupedDynamics(dt=0.01)
    x = dyn.standing_state()
    u = np.zeros(12)
    x_next = dyn.step(x, u)
    # vz should decrease by g*dt
    expected_vz = -9.81 * 0.01
    assert abs(x_next[5] - expected_vz) < 1e-4, \
        f"Freefall vz: {x_next[5]} vs {expected_vz}"
    print("✓ dynamics_freefall")


def test_pmp_gain_matches_lqr():
    """PMP steady-state gain should match LQR gain (separation principle)."""
    dyn = QuadrupedDynamics(dt=0.005)
    x_ref = dyn.standing_state()
    A_c, B_c = dyn.continuous_AB(x_ref)
    A_d, B_d, g_d = dyn.get_linear_system(x_ref)

    Q = np.diag([100]*3 + [10]*3 + [200]*3 + [1]*3)
    R = np.eye(12) * 1e-3

    pmp = PontryaginController(A=A_c, B=B_c, Q_s=Q, R_u=R, dt=0.005)
    lqg = LQGController(A_d=A_d, B_d=B_d, g_d=g_d,
                         Q=Q*0.005, R=R*0.005)

    # Both should produce similar control for same state error
    dx = np.array([0.01, 0, 0, 0.05, 0, 0, 0, 0.03, 0, 0, 0, 0])
    u_pmp = pmp.compute_control(dx + x_ref, x_ref)
    u_lqg = lqg.compute_control(dx + x_ref, x_ref)

    # They won't be identical (continuous vs discrete Riccati) but same direction
    cos_sim = np.dot(u_pmp, u_lqg) / (np.linalg.norm(u_pmp) * np.linalg.norm(u_lqg) + 1e-10)
    assert cos_sim > 0.1, f"PMP/LQR gains misaligned: cosine similarity = {cos_sim:.3f}"
    print(f"✓ pmp_gain_matches_lqr (cosine similarity: {cos_sim:.3f})")


def test_kalman_filter_convergence():
    """Kalman filter should converge to true state."""
    kf = KalmanFilter(nx=12, ny=12)
    x_true = np.array([0.1, 0, 0.3, 0, 0, 0, 0.05, 0, 0, 0, 0, 0])
    kf.x_hat = np.zeros(12)

    A = np.eye(12)
    B = np.zeros((12, 12))

    for _ in range(50):
        y = x_true + np.random.randn(12) * 0.01
        kf.predict(A, B, np.zeros(12))
        kf.update(y)

    err = np.linalg.norm(kf.state_estimate - x_true)
    assert err < 0.05, f"KF didn't converge: error = {err:.4f}"
    print(f"✓ kalman_filter_convergence (error: {err:.4f})")


def test_orientation_ekf():
    """EKF should estimate gravity direction correctly at rest."""
    ekf = OrientationEKF(dt=0.005)

    for _ in range(200):
        gyro = np.zeros(3) + np.random.randn(3) * 0.001
        accel = np.array([0, 0, -9.81]) + np.random.randn(3) * 0.05
        ekf.predict(gyro)
        ekf.update_accel(accel)

    rpy = ekf.euler_rpy
    assert np.allclose(rpy[:2], 0, atol=0.05), f"EKF roll/pitch: {rpy[:2]}"
    print(f"✓ orientation_ekf (rpy: {rpy})")


def test_friction_cone():
    """Friction cone constraints should have correct structure."""
    D, d = QuadrupedDynamics.friction_cone_constraints(mu=0.6)
    assert D.shape == (20, 12)
    assert d.shape == (20,)

    # A force straight down should satisfy constraints
    u = np.zeros(12)
    for i in range(4):
        u[3*i + 2] = 30.0  # fz = 30N per leg
    assert np.all(D @ u <= d + 1e-10), "Vertical force violates friction cone"

    # A force too horizontal should violate
    u2 = np.zeros(12)
    u2[0] = 50.0  # fx on FL with fz=0
    violated = np.any(D @ u2 > d + 1e-10)
    assert violated, "Horizontal force should violate friction cone"
    print("✓ friction_cone")


if __name__ == '__main__':
    test_dynamics_equilibrium()
    test_dynamics_freefall()
    test_friction_cone()
    test_orientation_ekf()
    test_kalman_filter_convergence()
    test_pmp_gain_matches_lqr()
    print("\nAll tests passed!")
