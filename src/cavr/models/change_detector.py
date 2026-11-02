"""Predictive-surprise CUSUM change detector.

Surprise z_t = -log P_M(s2|s,a) is computed from the PRE-UPDATE predictive
distribution (p_before) so that the anomaly score can never be concealed by
the very observation it describes. The adaptive null mean is the predictive
entropy E_p[-log p] (expected surprise under the current model), so a
well-modeled stationary stochasticity does not accumulate. Alarms fire when
the cumulative G_t exceeds a calibrated threshold h; the threshold is
calibrated on no-change validation runs to satisfy a false-alarm budget.
"""
from __future__ import annotations

import numpy as np


class ChangeDetector:
    """Two-level CUSUM: a global statistic and per-(s,a) pair statistics.

    The pair-level CUSUM accumulates excess surprise for a specific
    state-action pair with a slow per-pair drift, so sparse but repeated
    contradictions of the SAME edge accumulate across a long horizon (a
    global CUSUM would wash them out with well-predicted steps in between).
    Alarm localization is free: the crossing pair is the suspect region."""

    def __init__(self, h: float = 3.0, drift: float = 0.01, cooldown: int = 15,
                 inc_cap: float = 2.0, pair_drift: float = 0.02,
                 pair_h: float | None = None):
        self.h = float(h)
        self.pair_h = float(pair_h if pair_h is not None else h)
        self.drift = float(drift)
        self.pair_drift = float(pair_drift)
        self.cooldown = int(cooldown)
        self.inc_cap = float(inc_cap)  # winsorization: bounds the increment tail
        self.G = 0.0
        self.pair_G: dict[tuple, float] = {}
        self.last_alarm_t: int = -10**9
        self.last_pair_alarm_t: dict[tuple, int] = {}
        self.n_alarms = 0
        self.alarm_ts: list[int] = []

    def score_pre_update(self, p_before: np.ndarray, s2_category: int) -> tuple[float, float]:
        """Returns (z, g_increment). Uses only the frozen pre-update
        distribution; the caller must score BEFORE model.update()."""
        p = max(float(p_before[s2_category]), 1e-8)
        z = -float(np.log(p))
        p_safe = np.clip(p_before, 1e-12, None)
        mu_null = float(-(p_safe * np.log(p_safe)).sum())
        g_inc = min(z - mu_null - self.drift, self.inc_cap)
        return z, g_inc

    def confirm(self, g_inc: float, t: int, key=None) -> bool:
        """Advance the CUSUM statistic(s); return True exactly when a
        reliable alarm fires. With key=(s,a), the pair-level statistic is
        used (with the global statistic updated as well)."""
        self.G = max(0.0, self.G + g_inc)
        if key is None:
            if self.G > self.h and (t - self.last_alarm_t) > self.cooldown:
                self.last_alarm_t = t
                self.n_alarms += 1
                self.alarm_ts.append(t)
                self.G = 0.0
                return True
            return False
        g = self.pair_G.get(key, 0.0) + g_inc - self.pair_drift
        g = max(0.0, g)
        if g > 0:
            self.pair_G[key] = g
        else:
            self.pair_G.pop(key, None)
        if g > self.pair_h and (t - self.last_pair_alarm_t.get(key, -10**9)) > self.cooldown:
            self.last_pair_alarm_t[key] = t
            self.last_alarm_t = t
            self.n_alarms += 1
            self.alarm_ts.append(t)
            self.pair_G.pop(key, None)
            return True
        return False

    def alarm_strength(self) -> float:
        """Public (agent-visible) alarm confidence in [0, 1.5]."""
        pair = max(self.pair_G.values()) if self.pair_G else 0.0
        return min(1.5, max(self.G, pair) / max(self.pair_h, 1e-9))

    def recent_alarm(self, t: int, window: int = 60) -> bool:
        return (t - self.last_alarm_t) < window

    def reset(self) -> None:
        self.G = 0.0

    def state_dict(self) -> dict:
        keys = list(self.pair_G.keys())
        return {"G": np.float64(self.G), "h": np.float64(self.h),
                "last_alarm_t": np.int64(self.last_alarm_t),
                "n_alarms": np.int64(self.n_alarms),
                "pair_keys_a": np.asarray([k[0] for k in keys], dtype=np.int64),
                "pair_keys_b": np.asarray([k[1] for k in keys], dtype=np.int64),
                "pair_vals": np.asarray([self.pair_G[k] for k in keys], dtype=np.float64)}

    def load_state_dict(self, d: dict) -> None:
        self.G = float(d["G"])
        self.h = float(d["h"])
        self.last_alarm_t = int(d["last_alarm_t"])
        self.n_alarms = int(d["n_alarms"])
        self.pair_G = {}
        for a, b, v in zip(np.asarray(d.get("pair_keys_a", [])),
                           np.asarray(d.get("pair_keys_b", [])),
                           np.asarray(d.get("pair_vals", []))):
            self.pair_G[(int(a), int(b))] = float(v)
