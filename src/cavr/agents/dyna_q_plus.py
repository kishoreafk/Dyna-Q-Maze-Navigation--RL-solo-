"""Dyna-Q+ (baseline 3): planning-time age bonus kappa*sqrt(tau) and
self-transition initialization for never-tried actions. The bonus is applied
ONLY in simulated backups, never to logged real rewards (spec 5.3)."""
from __future__ import annotations

import numpy as np

from .base import BaseAgent


class DynaQPlusAgent(BaseAgent):
    name = "dyna_q_plus"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng):
        super().__init__(n_states, W, H, goal_idx, cfg, rng)
        self.n_plan = int(cfg.get("dyna_n", 20))
        self.kappa = float(cfg.get("kappa", 0.001))
        self.model: dict[tuple, tuple] = {}
        self._keys: list[tuple] = []
        self.last_visit: dict[tuple, int] = {}
        self.tried: dict[int, set] = {}

    def observe(self, s, a, r, s2, terminal, t):
        delta = self.direct_update(s, a, r, s2, terminal)
        key = (s, a)
        if key not in self.model:
            self._keys.append(key)
        self.model[key] = (r, s2, bool(terminal))
        self.last_visit[key] = t
        self.tried.setdefault(s, set()).add(a)

        executed = 0
        if self._keys:
            # synthetic entries: untried actions of visited states act as
            # self-transitions with zero reward (Sutton's Dyna-Q+)
            synth_pool = []
            for sp, tried in self.tried.items():
                for ap in range(4):
                    if ap not in tried:
                        synth_pool.append((sp, ap))
            for _ in range(self.n_plan):
                use_synth = synth_pool and self.rng.random() < 0.2
                if use_synth:
                    up, ua = synth_pool[int(self.rng.integers(len(synth_pool)))]
                    rm, s2m, term = 0.0, up, False
                    tau = t  # never tried: maximal age
                else:
                    up, ua = self._keys[int(self.rng.integers(len(self._keys)))]
                    rm, s2m, term = self.model[(up, ua)]
                    tau = t - self.last_visit.get((up, ua), 0)
                bonus = self.kappa * np.sqrt(max(0, tau))
                target = (rm + bonus) + self.gamma * (0.0 if term else float(self.q[s2m].max()))
                self.q[up, ua] += self.alpha * (target - float(self.q[up, ua]))
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
