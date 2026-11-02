"""D* Lite baseline agent (incremental search on the observed partial map).

The agent maintains edge beliefs (shared EdgeBeliefModel with an optimistic
prior) and runs D* Lite on the believed map. Blocked edges are inferred from
movement outcomes via the same slip-marginalized Bayes update used by CAVR
(a self-loop at rho>0 is ambiguous evidence, never instant proof). A bounded
age-based retest of believed-blocked adjacent edges provides the same nonzero
re-discovery floor required of every method.
"""
from __future__ import annotations

import time

import numpy as np

from ..envs.maze_gen import edge_key, neighbor_cell
from ..models.edge_belief import EdgeBeliefModel
from ..planning.dstar_lite import DStarLite
from .base import BaseAgent


class DStarLiteAgent(BaseAgent):
    name = "dstar_lite"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng, rho_est: float = 0.1):
        super().__init__(n_states, W, H, goal_idx, cfg, rng)
        self.rho_est = rho_est
        self.edges = EdgeBeliefModel(H, W, b0=float(cfg.get("b0", 0.6)),
                                     rho=rho_est, lr=float(cfg.get("edge_lr", 0.4)))
        self.planner: DStarLite | None = None
        self.retest_interval = int(cfg.get("dstar_retest_interval", 50))
        self._last_retest = -10**9
        self._last_a = 0
        # consecutive-failure evidence for hard blocked-edge detection
        # (classic D* Lite terrain observation; threshold derived from the
        # calibrated slip estimate, never from hidden state)
        self._fail_streak: dict[tuple, int] = {}
        self.block_streak = 3 if rho_est <= 0.15 else 5

    def seed_map(self, open_set, closed_known=None) -> None:
        """Optional: start from fully observed terrain (classic D* Lite
        setting with known map)."""
        for r in range(self.H):
            for c in range(self.W):
                for a in range(4):
                    nb = neighbor_cell((r, c), a)
                    if nb is None:
                        continue
                    e = edge_key((r, c), nb)
                    self.edges.set_b(e, 0.95 if e in open_set else 0.05)
        self.planner = None  # force rebuild with new beliefs

    def act(self, s: int, t: int) -> int:
        if self.planner is None:
            self.planner = DStarLite(self.H, self.W, self.goal_idx, self.edges,
                                     s_start=s)
        self.planner.move_agent(s)
        a = self.planner.next_action()
        if a is None:
            a = int(self.rng.integers(4))
        # bounded retest floor for believed-blocked adjacent edges
        if (t - self._last_retest) >= self.retest_interval and self.rng.random() < 0.5:
            cell = (s // self.W, s % self.W)
            best_e, best_age, best_a = None, -1, None
            for ang in range(4):
                nb = neighbor_cell(cell, ang)
                if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                    continue
                e = edge_key(cell, nb)
                if self.edges.b(e) <= 0.5:
                    age = t - 0  # last test times are not tracked per edge here
                    if age > best_age:
                        best_e, best_age, best_a = e, age, ang
            if best_a is not None:
                a = best_a
                self._last_retest = t
                self.totals["probes"] += 1
        self._last_a = a
        return a

    def observe(self, s, a, r, s2, terminal, t):
        t0 = time.perf_counter()
        flips = []
        cell = (s // self.W, s % self.W)
        nb_cmd = neighbor_cell(cell, a)
        e_cmd = edge_key(cell, nb_cmd) if (nb_cmd is not None and
                                           0 <= nb_cmd[0] < self.H and 0 <= nb_cmd[1] < self.W) else None
        # map-flip detection for all incident edges (before/after one update)
        before = {}
        for ang in range(4):
            nb = neighbor_cell(cell, ang)
            if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                continue
            before[ang] = self.edges.b(edge_key(cell, nb)) > 0.5
        # ONE Bayes update with the actually commanded action (marginalizing slip)
        self.edges.update(s, a, s2, t)
        for ang, b0 in before.items():
            nb = neighbor_cell(cell, ang)
            if (self.edges.b(edge_key(cell, nb)) > 0.5) != b0:
                flips.append((s, nb[0] * self.W + nb[1]))
        # consecutive self-loops on the same commanded edge: strong evidence
        if e_cmd is not None and s2 == s:
            k = (s, a)
            self._fail_streak[k] = self._fail_streak.get(k, 0) + 1
            if self._fail_streak[k] >= self.block_streak and self.edges.b(e_cmd) > 0.5:
                self.edges.set_b(e_cmd, 0.05)
                flips.append((s, nb_cmd[0] * self.W + nb_cmd[1]))
                self._fail_streak[k] = 0
        elif e_cmd is not None:
            self._fail_streak[(s, a)] = 0
        # direct update keeps a comparable Q table for logging/metrics
        delta = self.direct_update(s, a, r, s2, terminal)
        if self.planner is not None:
            for (u, v) in flips:
                self.planner.notify_cost_change(u, v)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        self.totals["planning_ms"] += wall_ms
        self.step_stats = {"backups_executed": 0, "budget_assigned": 0,
                           "queue_operations": 0, "wall_time_ms": wall_ms,
                           "model_version": 0, "model_uncertainty": 0.0,
                           "predicted_change_probability": 0.0,
                           "budget_reason": "n/a",
                           "proposed_probe": -1, "actual_action": a,
                           "alarm": False, "surprise": 0.0, "direct_delta": delta}
        return self.step_stats
