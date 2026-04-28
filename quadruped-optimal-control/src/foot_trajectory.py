import numpy as np
from src.gait_scheduler import LEG_ORDER


class JointSpaceTrotPlanner:
    def __init__(self, q_nom):
        self.q_nom = {
            leg: np.array(q_nom[leg], dtype=np.float32).copy()
            for leg in LEG_ORDER
        }

        self.side_sign = {
            "FL": +1.0,
            "RL": +1.0,
            "FR": -1.0,
            "RR": -1.0,
        }

    # =========================
    # PERFIL SUAVE DE SWING
    # =========================
    def _smooth_swing(self, s):
        return np.sin(np.pi * s)

    def _knee_profile(self, s):
        # más natural (no tan brusco)
        if s < 0.2:
            return 0.5 * np.sin(np.pi * s / 0.2)
        return np.sin(np.pi * s)

    # =========================
    # WALK (MEJORADO)
    # =========================
    def _walk(self, t, gait, vx, vy, wz):
        q_des = {}

        vmag = np.sqrt(vx**2 + vy**2)
        walk_sign = 1.0 if vx >= 0 else -1.0

        # 🔥 amplitudes adaptativas (MUY importante)
        hip_amp = np.clip(0.15 + 2.0 * vmag, 0.15, 0.35)
        knee_amp = np.clip(0.18 + 2.2 * vmag, 0.18, 0.35)
        lift_amp = np.clip(0.25 + 1.5 * vmag, 0.25, 0.5)
        lat_amp = np.clip(0.15 * abs(vy), 0.0, 0.2)

        for leg in LEG_ORDER:
            q0 = self.q_nom[leg].copy()

            side = self.side_sign[leg]
            forward = 1.0 if "F" in leg else -1.0

            if gait.is_swing(leg, t):
                s = gait.swing_phase(leg, t)

                swing = self._smooth_swing(s)

                # ===== LIFT =====
                lift = lift_amp * swing

                # ===== HFE (avance) =====
                sweep = hip_amp * walk_sign * (1.0 - 2.0 * s)

                # ===== HAA (lateral) =====
                lat = lat_amp * side * np.sin(np.pi * s)

                # ===== APPLY =====
                q_haa = q0[0] + lat
                q_hfe = q0[1] + forward * sweep
                q_kfe = q0[2] + knee_amp * self._knee_profile(s) + lift

            else:
                s = gait.stance_phase(leg, t)

                # ===== EMPUJE =====
                push = hip_amp * walk_sign * (0.5 - s)

                # ===== LATERAL =====
                lat = -0.3 * vy * side

                # ===== APPLY =====
                q_haa = q0[0] + lat
                q_hfe = q0[1] + forward * push
                q_kfe = q0[2] + 0.1 * knee_amp * np.cos(np.pi * s)

            q_des[leg] = np.array([q_haa, q_hfe, q_kfe], dtype=np.float32)

        return q_des

    # =========================
    # TURN (MUCHO MÁS ESTABLE)
    # =========================
    def _turn(self, t, gait, wz):
        q_des = {}

        turn_sign = np.sign(wz + 1e-9)

        hip_amp = 0.25
        knee_amp = 0.3
        ab_amp = 0.18

        for leg in LEG_ORDER:
            q0 = self.q_nom[leg].copy()

            side = self.side_sign[leg]
            diag = 1.0 if leg in ["FL", "RR"] else -1.0

            if gait.is_swing(leg, t):
                s = gait.swing_phase(leg, t)

                sweep = -diag * hip_amp * (1 - 2*s)

                q_haa = q0[0] + side * ab_amp * np.sin(np.pi*s) * turn_sign
                q_hfe = q0[1] + sweep
                q_kfe = q0[2] + knee_amp * np.sin(np.pi*s)

            else:
                s = gait.stance_phase(leg, t)

                sweep = diag * hip_amp * (0.8 - 1.6*s)

                q_haa = q0[0]
                q_hfe = q0[1] + sweep
                q_kfe = q0[2] + 0.15 * np.sin(np.pi*s)

            q_des[leg] = np.array([q_haa, q_hfe, q_kfe], dtype=np.float32)

        return q_des

    # =========================
    # MAIN API
    # =========================
    def get_joint_targets(self, t, gait, vx_cmd=0.0, vy_cmd=0.0, wz_cmd=0.0):

        # 🔥 decisión inteligente
        if abs(wz_cmd) > 0.15:
            return self._turn(t, gait, wz_cmd)

        return self._walk(t, gait, vx_cmd, vy_cmd, wz_cmd)