"""Base tabular agent: shared Q-learning machinery and per-step stats.

All agents share the same public observation contract: they receive only
(s, a, r, s2, terminal, t) — never the true map, schedule, or slip draws.
"""
from __future__ import annotations

import numpy as np


class BaseAgent:
    name = "base"

    def __init__(self, n_states: int, W: int, H: int, goal_idx: int,
                 cfg: dict, rng):
        self.n_states = n_states
        self.W = W
        self.H = H
        self.goal_idx = goal_idx
        self.cfg = cfg
        self.rng = rng
        self.alpha = float(cfg.get("alpha", 0.1))
        self.gamma = float(cfg.get("gamma", 0.97))
        self.eps = float(cfg.get("eps", 0.1))
        self.q = np.zeros((n_states, 4), dtype=np.float64)
        self.step_stats: dict = {}
        self.totals = {"backups": 0, "queue_ops": 0, "planning_ms": 0.0,
                       "probes": 0, "alarms": 0, "real_updates": 0}

    # ------------------------------------------------------------------ #
    def V(self) -> np.ndarray:
        v = self.q.max(axis=1)
        v[self.goal_idx] = 0.0
        return v

    def greedy(self, s: int) -> int:
        row = self.q[s]
        m = row.max()
        idx = np.flatnonzero(row == m)
        return int(idx[int(self.rng.integers(len(idx)))]) if len(idx) > 1 else int(idx[0])

    def act(self, s: int, t: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(4))
        return self.greedy(s)

    def direct_update(self, s: int, a: int, r: float, s2: int, terminal: bool) -> float:
        """Exactly one real TD update. Terminal states never bootstrap;
        time-limit truncation still bootstraps (terminal=False there)."""
        target = r + self.gamma * (0.0 if terminal else float(self.q[s2].max()))
        delta = target - float(self.q[s, a])
        self.q[s, a] += self.alpha * delta
        self._after_q_update(s)
        self.totals["real_updates"] += 1
        return delta

    def _after_q_update(self, s: int) -> None:
        """Hook for subclasses to maintain derived state (e.g., V cache)."""
        return None

    def observe(self, s: int, a: int, r: float, s2: int, terminal: bool, t: int) -> dict:
        delta = self.direct_update(s, a, r, s2, terminal)
        self.step_stats = {"backups_executed": 0, "budget_assigned": 0,
                           "queue_operations": 0, "wall_time_ms": 0.0,
                           "model_version": 0, "model_uncertainty": 0.0,
                           "predicted_change_probability": 0.0,
                           "budget_reason": "n/a",
                           "proposed_probe": -1, "actual_action": a,
                           "alarm": False, "surprise": 0.0, "direct_delta": delta}
        return self.step_stats

    # ------------------------------------------------------------------ #
    def state_arrays(self) -> dict:
        return {"q": self.q.copy()}

    def load_state_arrays(self, arrays: dict) -> None:
        self.q = np.array(arrays["q"], dtype=np.float64)
