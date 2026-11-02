"""Prioritized sweeping (baseline 4): heap over absolute Bellman residuals,
reverse predecessor propagation, capped at the same backup budget."""
from __future__ import annotations

import heapq

from .base import BaseAgent


class PrioritizedSweepingAgent(BaseAgent):
    name = "prioritized_sweeping"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng):
        super().__init__(n_states, W, H, goal_idx, cfg, rng)
        self.n_plan = int(cfg.get("ps_n", 20))
        self.theta = float(cfg.get("ps_theta", 1e-3))
        self.model: dict[tuple, tuple] = {}
        self.preds: dict[int, set] = {}
        self._contrib: dict[tuple, int] = {}
        self._heap: list = []
        self._tie = 0
        self.queue_ops = 0

    def _set_model(self, s, a, r, s2, terminal):
        key = (s, a)
        old_s2 = self._contrib.get(key)
        if old_s2 is not None and key in self.preds.get(old_s2, ()):
            self.preds[old_s2].discard(key)
        self.model[key] = (r, s2, terminal)
        self._contrib[key] = s2
        self.preds.setdefault(s2, set()).add(key)

    def _push(self, s, a, p):
        if p > self.theta:
            heapq.heappush(self._heap, (-p, self._tie, s, a))
            self._tie += 1
            self.queue_ops += 1

    def _target(self, s, a):
        r, s2, term = self.model[(s, a)]
        return r + self.gamma * (0.0 if term else float(self.q[s2].max()))

    def observe(self, s, a, r, s2, terminal, t):
        delta = self.direct_update(s, a, r, s2, terminal)
        self._set_model(s, a, r, s2, terminal)
        self._push(s, a, abs(delta))
        executed = 0
        for _ in range(self.n_plan):
            if not self._heap:
                break
            _negp, _tie, sp, ap = heapq.heappop(self._heap)
            self.queue_ops += 1
            if (sp, ap) not in self.model:
                continue
            delta_p = self._target(sp, ap) - float(self.q[sp, ap])
            self.q[sp, ap] += self.alpha * delta_p
            executed += 1
            # propagate to predecessors of sp
            for (pp, pap) in list(self.preds.get(sp, ())):
                if (pp, pap) not in self.model:
                    continue
                P = abs(self._target(pp, pap) - float(self.q[pp, pap]))
                if P > self.theta:
                    self._push(pp, pap, P)
        self.totals["backups"] += executed
        self.totals["queue_ops"] = self.queue_ops
        self.step_stats = {"backups_executed": executed, "budget_assigned": self.n_plan,
                           "queue_operations": self.queue_ops, "wall_time_ms": 0.0,
                           "model_version": 0, "model_uncertainty": 0.0,
                           "predicted_change_probability": 0.0,
                           "budget_reason": "n/a",
                           "proposed_probe": -1, "actual_action": a,
                           "alarm": False, "surprise": 0.0, "direct_delta": delta}
        return self.step_stats
