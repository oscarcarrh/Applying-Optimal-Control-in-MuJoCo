import numpy as np

LEG_ORDER = ["FL", "FR", "RL", "RR"]


class TrotGaitScheduler:
    """
    Trot robusto y limpio.

    FL + RR en fase
    FR + RL en contrafase
    """

    def __init__(self, period=0.6, duty_factor=0.6):
        self.T = float(period)
        self.duty = float(duty_factor)

        # offsets en ciclo [0,1)
        self.phase_offsets = {
            "FL": 0.0,
            "RR": 0.0,
            "FR": 0.5,
            "RL": 0.5,
        }

    # =========================
    # FASE NORMALIZADA
    # =========================
    def phase(self, leg, t):
        return (t / self.T + self.phase_offsets[leg]) % 1.0

    # =========================
    # ESTADOS
    # =========================
    def is_stance(self, leg, t):
        return self.phase(leg, t) < self.duty

    def is_swing(self, leg, t):
        return not self.is_stance(leg, t)

    # =========================
    # FASES NORMALIZADAS
    # =========================
    def swing_phase(self, leg, t):
        p = self.phase(leg, t)

        if p <= self.duty:
            return 0.0

        return (p - self.duty) / max(1e-6, (1.0 - self.duty))

    def stance_phase(self, leg, t):
        p = self.phase(leg, t)

        if p >= self.duty:
            return 0.0

        return p / max(1e-6, self.duty)

    # =========================
    # CONTACTO GLOBAL
    # =========================
    def contact_mask(self, t):
        return np.array(
            [self.is_stance(leg, t) for leg in LEG_ORDER],
            dtype=bool
        )

    # =========================
    # UTILIDAD PRO (debug/control)
    # =========================
    def duty_cycle_time(self):
        return self.T * self.duty

    def swing_time(self):
        return self.T * (1.0 - self.duty)