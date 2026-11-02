"""Vanilla Dyna-Q with last-outcome model (baseline 2, canonical Sutton)."""
from __future__ import annotations

import numpy as np

from .base import BaseAgent


class DynaQAgent(BaseAgent):
    name = "dyna_q"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng):
        super().__init__(n_states, W, H, goal_idx, cfg, rng)
        self.n_plan = int(cfg.get("dyna_n", 20))
        self.model: dict[tuple, tuple] = {}   # (s,a) -> (r, s2, terminal)
        self._keys: list[tuple] = []

    def observe(self, s, a, r, s2, terminal, t):
        delta = self.direct_update(s, a, r, s2, terminal)
        key = (s, a)
        if key not in self.model:
            self._keys.append(key)
        self.model[key] = (r, s2, bool(terminal))
        executed = 0
        if self._keys:
            for _ in range(self.n_plan):
                ks, ka = self._keys[int(self.rng.integers(len(self._keys)))]
                rm, s2m, term = self.model[(ks, ka)]
                target = rm + self.gamma * (0.0 if term else float(self.q[s2m].max()))
                self.q[ks, ka] += self.alpha * (target - float(self.q[ks, ka]))
                executed += 1
        self.totals["backups"] += executed
        self.step_stats = {"backups_executed": executed, "budget_assigned": self.n_plan,
                           "queue_operations": 0, "wall_time_ms": 0.0,
                           "model_version": 0, "model_uncertainty": 0.0,
                           "predicted_change_probability": 0.0,
                           "budget_reason": "n/a",
                           "proposed_probe": -1, "actual_action": a,
                           "alarm": False, "surprise": 0.0, "direct_delta": delta}
        return self.step_stats
