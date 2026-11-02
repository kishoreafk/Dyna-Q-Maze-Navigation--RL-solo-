"""CAVR-Dyna: Counterfactual, Aleatoric-aware, Value-of-computation-driven
Repair for Dyna-style navigation.

Per real step (spec 6.3 loop):
  1. freeze pre-update prediction p_before; score surprise on it;
  2. exactly one direct TD update; append episodic memory;
  3. CUSUM alarm confirmation; on alarm: local forgetting (obsolete evidence
     discarded, latest retained exactly once) + edge-belief doubt bump;
  4. incorporate the new observation into the Dirichlet model EXACTLY once;
  5. sync predecessor graph; update edge beliefs (mixture likelihood);
  6. goal-counterfactual edge scoring seeds repair targets in the heap;
  7. budget controller picks B in {0,8,16,32,64}; expected stochastic model
     backups run from the versioned heap (stale epochs discarded);
  8. probing is a REAL action choice, kept distinct from backups.

Variants (ablations): detect_changes=False -> no detector/reset;
adaptive_budget=False -> fixed B=32.
"""
from __future__ import annotations

import time

import numpy as np

from ..envs.maze_gen import DELTAS, edge_key, neighbor_cell
from ..models.categorical_dirichlet import CategoricalDirichlet, category_of
from ..models.episodic_map import EpisodicMap
from ..models.change_detector import ChangeDetector
from ..models.edge_belief import EdgeBeliefModel
from ..planning.predecessors import PredecessorGraph, VersionedPriorityQueue
from ..planning.counterfactual import CounterfactualScorer
from ..planning.probe import ProbeSelector
from ..planning.budget import BudgetController, BudgetFeatures
from .stochastic_dyna import StochasticDynaBase


def direction_between(x: int, y: int, W: int) -> int:
    dx = (y // W) - (x // W)
    dy = (y % W) - (x % W)
    for a, (dr, dc) in enumerate(DELTAS):
        if (dr, dc) == (dx, dy):
            return a
    raise ValueError(f"cells not adjacent: {x}, {y}")


class CAVRDynaAgent(StochasticDynaBase):
    name = "cavr"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng, rho_est: float = 0.1,
                 detect_changes: bool = True, adaptive_budget: bool = True):
        super().__init__(n_states, W, H, goal_idx, cfg, rng, rho_est=rho_est,
                         use_detector=detect_changes, do_reset=detect_changes)
        self.detect_changes = detect_changes
        self.adaptive_budget = adaptive_budget
        self.cap = int(cfg.get("cap", 64))
        self.theta_useful = float(cfg.get("theta_useful", 0.01))
        self.theta_push = float(cfg.get("theta_push", 0.05))
        self.theta_probe = float(cfg.get("theta_probe", 0.12))
        self.probe_min_gap = int(cfg.get("probe_min_gap", 8))

        self.edges = EdgeBeliefModel(H, W, b0=float(cfg.get("b0", 0.6)),
                                     rho=rho_est, lr=float(cfg.get("edge_lr", 0.4)))
        self.preds = PredecessorGraph(n_states)
        self.heap = VersionedPriorityQueue()
        self.cf = CounterfactualScorer(self.gamma, H, W, rho_est)
        self.probe = ProbeSelector(H, W, theta=self.theta_probe,
                                   min_gap=self.probe_min_gap)
        self.budget = BudgetController(cap=self.cap,
                                       theta_high=float(cfg.get("budget_theta_high", 0.15)),
                                       theta_lo=float(cfg.get("budget_theta_lo", 0.10)),
                                       fixed_b=None if adaptive_budget else 32)
        self.p_change: dict[tuple, float] = {}
        self._pchange_t: dict[tuple, int] = {}
        self._reach: dict[int, float] = {}
        self._reach_t = -10**9
        self._cand: list = []
        self._cand_t = -10**9
        self._last_cf_t = -10**9
        self._last_probe_t = -10**9
        self._top_repair = 0.0
        self._epi_ema = 0.5
        self._pending_probe_edge = None

    # ------------------------------------------------------------------ #
    # edge change-probability mass (lazily decayed)
    # ------------------------------------------------------------------ #
    def _pc_get(self, e, t: int) -> float:
        e = tuple(e)
        if e in self.p_change and e in self._pchange_t:
            dt = max(0, t - self._pchange_t[e])
            if dt:
                self.p_change[e] *= float(np.exp(-0.005 * dt))
        self._pchange_t[e] = t
        return self.p_change.get(e, 0.0)

    def _pc_add(self, e, w: float, t: int) -> None:
        e = tuple(e)
        cur = self._pc_get(e, t)
        self.p_change[e] = min(1.0, cur + w)

    # ------------------------------------------------------------------ #
    def localize_cells(self, s: int, a: int, s2: int) -> set:
        cells = {s}
        cell = (s // self.W, s % self.W)
        for ang in range(4):
            nb = neighbor_cell(cell, ang)
            if nb is not None and 0 <= nb[0] < self.H and 0 <= nb[1] < self.W:
                cells.add(nb[0] * self.W + nb[1])
        return cells

    def _suspect_edges(self, s: int, a: int):
        cell = (s // self.W, s % self.W)
        out = []
        for ang in range(4):
            nb = neighbor_cell(cell, ang)
            if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                continue
            out.append((ang, edge_key(cell, nb)))
        return out

    def _bump_suspect_edges(self, cells, s: int, a: int, t: int) -> None:
        for (ang, e) in self._suspect_edges(s, a):
            w = 0.5 if ang == a else 0.5 / 3.0
            self._pc_add(e, w, t)
            # soften structural belief toward doubt (30% toward 0.5)
            be = self.edges.b(e)
            self.edges.set_b(e, be + 0.3 * (0.5 - be))
        for sp in cells:
            if sp == s:
                continue
            cell = (sp // self.W, sp % self.W)
            for ang in range(4):
                nb = neighbor_cell(cell, ang)
                if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                    continue
                self._pc_add(edge_key(cell, nb), 0.5 / 3.0, t)

    def forget_region(self, cells, cur_t: int) -> None:
        super().forget_region(cells, cur_t)
        self.model.epoch += 1  # heap entries from older epochs become stale
        for sp in cells:
            for ap in range(4):
                self.preds.sync_pair(sp, ap, self.model)

    # ------------------------------------------------------------------ #
    # repair target scoring (counterfactual value influence)
    # ------------------------------------------------------------------ #
    def _enqueue_repairs(self, s_focus: int, t: int) -> float:
        V = self._V
        reach = self._reach
        cells = self.localize_cells(s_focus, -1, s_focus)
        cand: list[tuple] = []
        seen = set()
        for c in list(cells):
            cell = (c // self.W, c % self.W)
            for ang in range(4):
                nb = neighbor_cell(cell, ang)
                if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                    continue
                e = edge_key(cell, nb)
                if e not in seen:
                    seen.add(e)
                    cand.append(e)
        # add highest change-probability edges
        for e in sorted(self.p_change, key=lambda k: -self._pc_get(k, t))[:8]:
            if e not in seen and self.p_change.get(e, 0.0) > 0.02:
                seen.add(e)
                cand.append(e)
        cand = cand[:24]
        best = 0.0
        for e in cand:
            x, y = e
            xi = x[0] * self.W + x[1]
            yi = y[0] * self.W + y[1]
            D = self.cf.edge_impact(xi, yi, V)
            if D <= 1e-6:
                continue
            be = self.edges.b(e)
            uncert = 4.0 * be * (1.0 - be)
            pc = self._pc_get(e, t)
            age = min(1.0, float(np.sqrt(max(0, t - self.probe.last_test.get(e, 0))) / 50.0))
            score = (pc + 0.2 * uncert) * D * (reach.get(xi, 0.0) + reach.get(yi, 0.0)) + 0.05 * age
            if score <= 1e-6:
                continue
            prio = min(1.0, score) + 0.05
            for (node, other) in ((xi, yi), (yi, xi)):
                try:
                    a_toward = direction_between(node, other, self.W)
                except ValueError:
                    continue
                if self.model.n_observations(node, a_toward) > 0:
                    self.heap.push(node, a_toward, prio, self.model.epoch)
            best = max(best, prio)
        self._top_repair = max(self._top_repair * 0.9, best)
        return best

    # ------------------------------------------------------------------ #
    def _run_backups(self, B: int) -> int:
        executed = 0
        ops0 = self.heap.n_pushes + self.heap.n_pops
        for _ in range(B):
            item = self.heap.pop(self.model.epoch, self.theta_useful)
            if item is None:
                break
            s0, a0, _pr = item
            if self.model.n_observations(s0, a0) <= 0:
                continue
            delta = self.expected_backup(s0, a0)
            executed += 1
            if abs(delta) > self.theta_useful:
                for (sp, ap) in self.preds.get(s0):
                    if self.model.n_observations(sp, ap) <= 0:
                        continue
                    resid = abs(self.expected_target(sp, ap) - float(self.q[sp, ap]))
                    if resid > 2 * self.theta_useful:
                        self.heap.push(sp, ap, resid, self.model.epoch)
        self.totals["backups"] += executed
        self.totals["queue_ops"] += (self.heap.n_pushes + self.heap.n_pops) - ops0
        return executed

    # ------------------------------------------------------------------ #
    def _refresh_candidates(self, s: int, t: int, alarm_recent: bool) -> None:
        if not alarm_recent and (t - self._cand_t) < 10 and self._cand:
            return
        seen = set()
        cand = []
        frontier = {s}
        for _ in range(2):  # radius-2 neighborhood
            nxt = set()
            for c in frontier:
                cell = (c // self.W, c % self.W)
                for ang in range(4):
                    nb = neighbor_cell(cell, ang)
                    if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                        continue
                    e = edge_key(cell, nb)
                    if e not in seen:
                        seen.add(e)
                        cand.append(e)
                    nxt.add(nb[0] * self.W + nb[1])
            frontier |= nxt
        for e in sorted(self.p_change, key=lambda k: -self._pc_get(k, t))[:8]:
            if e not in seen:
                seen.add(e)
                cand.append(e)
        self._cand = cand[:30]
        self._cand_t = t

    # ------------------------------------------------------------------ #
    def act(self, s: int, t: int) -> int:
        alarm_recent = self.detector.recent_alarm(t) if self.detector else False
        self._refresh_candidates(s, t, alarm_recent)
        probe_edge, probe_action = None, None
        e, act, score = self.probe.top_probe(s, self._cand, self.edges, self._V,
                                             self._reach, {k: self._pc_get(k, t) for k in self._cand},
                                             t)
        if e is not None and score >= self.theta_probe and (t - self._last_probe_t) >= self.probe_min_gap \
                and self.rng.random() < 0.9:
            probe_edge, probe_action = e, act
            self._last_probe_t = t
            self.probe.observe_edge_test(e, t)
            self.totals["probes"] += 1
        if probe_action is not None:
            a = int(probe_action)
        elif self.rng.random() < self.eps:
            a = int(self.rng.integers(4))
        elif self.rng.random() < 0.02:
            a = int(self.rng.integers(4))  # explicit frontier exploration floor
        else:
            a = self.greedy(s)
        self._pending_probe_edge = probe_edge
        if probe_edge is None:
            self._pending_probe_id = -1
        else:
            self._pending_probe_id = (
                probe_edge[0][0] * self.W + probe_edge[0][1],
                probe_edge[1][0] * self.W + probe_edge[1][1],
            )
        return a

    # ------------------------------------------------------------------ #
    def observe(self, s, a, r, s2, terminal, t):
        t0 = time.perf_counter()
        # 1. frozen pre-update prediction -> surprise (never post-hoc)
        p_before = self.model.predict(s, a).copy()
        k = category_of(s, s2, self.W)
        if self.detector is not None:
            z, g_inc = self.detector.score_pre_update(p_before, k)
        else:
            z, g_inc = 0.0, 0.0
        # 2. exactly one direct TD update
        delta = self.direct_update(s, a, r, s2, terminal)
        # 3. episodic memory
        self.memory.append(s, a, r, s2, t)
        self._add_pair(s, a)
        # 4. alarm confirmation + local forgetting (latest kept exactly once)
        alarm = False
        if self.detector is not None:
            alarm = bool(self.detector.confirm(g_inc, t, key=(s, a)))
            self.totals["alarms"] += int(alarm)
            if alarm and self.do_reset:
                cells = self.localize_cells(s, a, s2)
                self.forget_region(cells, cur_t=t)
                self._bump_suspect_edges(cells, s, a, t)
        # 5. incorporate the new observation exactly once
        self.model.update(s, a, s2, r)
        self.preds.sync_pair(s, a, self.model)
        # 6. structural edge belief update (mixture likelihood, symmetric)
        self.edges.update(s, a, s2, t)
        cell = (s // self.W, s % self.W)
        nb = neighbor_cell(cell, a)
        if nb is not None and 0 <= nb[0] < self.H and 0 <= nb[1] < self.W:
            self.probe.observe_edge_test(edge_key(cell, nb), t)
        # 7. seed heap from direct residual
        if abs(delta) > self.theta_push and self.model.n_observations(s, a) > 0:
            self.heap.push(s, a, abs(delta), self.model.epoch)
        # 8. goal-counterfactual repair enqueue (on alarm / periodic)
        top_repair = 0.0
        if alarm or (t - self._last_cf_t) >= 10:
            top_repair = self._enqueue_repairs(s2, t)
            self._last_cf_t = t
        # 9. budget + planning
        f = BudgetFeatures(
            alarm_strength=self.detector.alarm_strength() if self.detector else 0.0,
            recent_alarm=self.detector.recent_alarm(t) if self.detector else False,
            top_repair_priority=self._top_repair,
            heap_size=len(self.heap),
            mean_epistemic=self._epi_ema,
        )
        B, why = self.budget.choose(f)
        executed = self._run_backups(B)
        # 10. reachability refresh (throttled)
        if alarm or (t - self._reach_t) >= 20:
            self._reach = self.cf.reach_mass(s2, self.edges, self.q, self.goal_idx)
            self._reach_t = t
        # 11. stats
        epi = self.model.epistemic(s, a)
        self._epi_ema = 0.9 * self._epi_ema + 0.1 * epi
        wall_ms = (time.perf_counter() - t0) * 1000.0
        self.totals["planning_ms"] += wall_ms
        self.step_stats = {
            "backups_executed": executed, "budget_assigned": int(B), "budget_reason": why,
            "queue_operations": int(self.heap.n_pushes + self.heap.n_pops),
            "wall_time_ms": wall_ms,
            "model_version": int(self.model.epoch),
            "model_uncertainty": float(epi),
            "predicted_change_probability": float(self.detector.alarm_strength()) if self.detector else 0.0,
            "proposed_probe": getattr(self, "_pending_probe_id", -1),
            "actual_action": int(a),
            "alarm": alarm, "surprise": float(z), "direct_delta": float(delta),
        }
        return self.step_stats

    # ------------------------------------------------------------------ #
    def checkpoint_arrays(self) -> dict:
        arrays = self.state_arrays()
        arrays.update(self.model.state_dict())
        arrays.update(self.edges.state_dict())
        arrays.update(self.detector.state_dict() if self.detector else
                      {"G": np.float64(0.0), "h": np.float64(self.cfg.get("h", 3.0)),
                       "last_alarm_t": np.int64(-10**9), "n_alarms": np.int64(0)})
        arrays.update(self.memory.state_dict())
        arrays["pchange_vals"] = np.asarray(
            [self.p_change.get(tuple(k), 0.0) for k in self.edges.edges()], dtype=np.float64)
        return arrays

    def load_checkpoint_arrays(self, arrays: dict, meta: dict) -> None:
        self.load_state_arrays(arrays)
        self.model.load_state_dict(arrays)
        self.edges.load_state_dict(arrays)
        if self.detector is not None:
            self.detector.load_state_dict(arrays)
        self.memory.load_state_dict(arrays)
        # rebuild derived structures
        for s in range(self.n_states):
            for a in range(4):
                if self.model.n_observations(s, a) > 0:
                    self.preds.sync_pair(s, a, self.model)
                    self._add_pair(s, a)
        # p_change keys align with edges list order
        vals = np.asarray(arrays.get("pchange_vals", []))
        for e, v in zip(self.edges.edges(), vals):
            self.p_change[tuple(e)] = float(v)
        self._V = self.q.max(axis=1)
        self._V[self.goal_idx] = 0.0
        self._epi_ema = float(meta.get("epi_ema", 0.5))
