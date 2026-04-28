import numpy as np


def wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class WaypointTrajectory:
    def __init__(self, waypoints, speed=0.15, dt=0.01, start_delay=2.0):
        self.waypoints = np.array(waypoints, dtype=float)
        self.speed = float(speed)
        self.dt = float(dt)
        self.start_delay = float(start_delay)

        self.segment_times = []
        self.cumulative_times = [self.start_delay]

        for i in range(len(self.waypoints) - 1):
            p0 = self.waypoints[i, :2]
            p1 = self.waypoints[i + 1, :2]

            dist = np.linalg.norm(p1 - p0)

            # 🔒 protección numérica
            T = max(dist / max(self.speed, 1e-6), self.dt)

            self.segment_times.append(T)
            self.cumulative_times.append(self.cumulative_times[-1] + T)

        self.total_time = self.cumulative_times[-1]

    def sample(self, t, height=0.225):
        # ===== ESPERA INICIAL =====
        if t <= self.start_delay:
            wp = self.waypoints[0]
            return (
                np.array([wp[0], wp[1], height]),
                np.zeros(3),
                np.array([0.0, 0.0, wp[2]]),
                np.zeros(3),
                False
            )

        # ===== FINAL =====
        if t >= self.total_time:
            wp = self.waypoints[-1]
            return (
                np.array([wp[0], wp[1], height]),
                np.zeros(3),
                np.array([0.0, 0.0, wp[2]]),
                np.zeros(3),
                True
            )

        # ===== SEGMENTO =====
        seg = 0
        for i in range(len(self.segment_times)):
            if self.cumulative_times[i] <= t < self.cumulative_times[i + 1]:
                seg = i
                break

        t0 = self.cumulative_times[seg]
        T = self.segment_times[seg]

        alpha = np.clip((t - t0) / T, 0.0, 1.0)

        wp0 = self.waypoints[seg]
        wp1 = self.waypoints[seg + 1]

        # ===== POSICIÓN =====
        p0 = wp0[:2]
        p1 = wp1[:2]
        pos_xy = (1 - alpha) * p0 + alpha * p1

        # ===== VELOCIDAD =====
        delta = p1 - p0
        dist = np.linalg.norm(delta)

        if dist > 1e-9:
            vel_xy = self.speed * delta / dist
        else:
            vel_xy = np.zeros(2)

        # ===== ORIENTACIÓN =====
        yaw0 = wp0[2]
        yaw1 = wp1[2]

        dyaw = wrap_to_pi(yaw1 - yaw0)
        yaw = wrap_to_pi(yaw0 + alpha * dyaw)

        wz = dyaw / max(T, 1e-6)

        # ===== OUTPUT COMPLETO =====
        pos = np.array([pos_xy[0], pos_xy[1], height])
        vel = np.array([vel_xy[0], vel_xy[1], 0.0])
        euler = np.array([0.0, 0.0, yaw])
        omega = np.array([0.0, 0.0, wz])

        return pos, vel, euler, omega, False


# =====================================
# 🔥 TRAYECTORIAS PRO
# =====================================

def build_trajectory(name):
    name = name.lower()

    # ===== 1. LÍNEA =====
    if name == "line":
        waypoints = [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]

    # ===== 2. CUADRADO =====
    elif name == "square":
        waypoints = [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, 0.2, np.pi/2],
            [0.0, 0.2, np.pi],
            [0.0, 0.0, -np.pi/2],
        ]

    # ===== 3. ZIGZAG =====
    elif name == "zigzag":
        waypoints = [
            [0.0, 0.0, 0.0],
            [0.1, 0.05, 0.2],
            [0.2, -0.05, -0.2],
            [0.3, 0.05, 0.2],
            [0.4, 0.0, 0.0],
        ]

    # ===== 4. CÍRCULO (NUEVO 🔥) =====
    elif name == "circle":
        waypoints = []
        R = 0.15
        for theta in np.linspace(0, 2*np.pi, 12):
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            yaw = theta + np.pi/2
            waypoints.append([x, y, yaw])

    # ===== 5. FIGURA 8 (NUEVO 🔥) =====
    elif name == "figure8":
        waypoints = []
        for t in np.linspace(0, 2*np.pi, 20):
            x = 0.2 * np.sin(t)
            y = 0.1 * np.sin(2*t)
            yaw = np.arctan2(2*np.cos(2*t), np.cos(t))
            waypoints.append([x, y, yaw])

    # ===== 6. ESPIRAL (NUEVO 🔥) =====
    elif name == "spiral":
        waypoints = []
        for t in np.linspace(0, 4*np.pi, 25):
            r = 0.02 * t
            x = r * np.cos(t)
            y = r * np.sin(t)
            yaw = t
            waypoints.append([x, y, yaw])

    else:
        raise ValueError(f"Trayectoria desconocida: {name}")

    return np.array(waypoints, dtype=float)