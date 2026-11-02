"""Real-action probe selection (distinct from synthetic Bellman backups).

Probes are REAL actions that test uncertain or age-stale edges. Two motives:
  1. epistemic: 4*b(1-b) uncertainty of an edge, weighted by reachability;
  2. retest (Dyna-Q+ style age): believed-blocked edges with potential value
     gain if actually open, growing with sqrt(age since last test).
A nonzero retest floor guarantees hidden shortcuts can be rediscovered
(Sparse-Imagination blind-spot argument). Travel uses only believed-open
edges (b > 0.5); if the target is adjacent, the probe IS the move through it.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..envs.maze_gen import neighbor_cell


class ProbeSelector:
    def __init__(self, H: int, W: int, theta: float = 0.12, min_gap: int = 8,
                 floor_prob: float = 0.02, travel_cost: float = 0.04,
                 retest_age_scale: float = 30.0):
        self.H = H
        self.W = W
        self.theta = float(theta)
        self.min_gap = int(min_gap)
        self.floor_prob = float(floor_prob)
        self.travel_cost = float(travel_cost)
        self.retest_age_scale = float(retest_age_scale)
        self.last_test: dict[tuple, int] = {}

    def observe_edge_test(self, e, t: int) -> None:
        self.last_test[tuple(sorted((tuple(e[0]), tuple(e[1]))))] = int(t)

    # ------------------------------------------------------------------ #
    def edge_score(self, e, edges, V: np.ndarray, reach: dict, p_change: dict,
                   t: int) -> float:
        x, y = e
        sx, sy = x[0] * self.W + x[1], y[0] * self.W + y[1]
        be = edges.b(e)
        uncert = 4.0 * be * (1.0 - be)
        reach_w = reach.get(sx, 0.0) + reach.get(sy, 0.0)
        uncert_score = uncert * (0.3 + min(2.0, 10.0 * reach_w))
        # retest potential: gain if the believed-blocked edge were open
        # (r_move - r_stay = 0.02; gamma factor folded into 0.97)
        gain_open = 0.02 + 0.97 * (float(V[sy]) - float(V[sx]))
        age = t - self.last_test.get(e, 0)
        retest_score = (1.0 - be) * max(0.0, gain_open) * min(1.5, np.sqrt(max(0, age)) / self.retest_age_scale)
        pc = p_change.get(e, 0.0)
        return max(uncert_score, retest_score) + 0.5 * pc

    def travel_penalty(self, s_now: int, e) -> float:
        x, y = e
        rx, cx = s_now // self.W, s_now % self.W
        dx = min(abs(rx - x[0]) + abs(cx - x[1]), abs(rx - y[0]) + abs(cx - y[1]))
        return self.travel_cost * dx

    def top_probe(self, s_now: int, candidates, edges, V: np.ndarray, reach: dict,
                  p_change: dict, t: int):
        """Returns (edge | None, action | None, score)."""
        best = (None, None, -1e9)
        for e in candidates:
            sc = self.edge_score(e, edges, V, reach, p_change, t) - self.travel_penalty(s_now, e)
            if sc > best[2]:
                act = self.first_step_toward(s_now, e, edges)
                if act is not None:
                    best = (e, act, sc)
        return best

    # ------------------------------------------------------------------ #
    def first_step_toward(self, s_now: int, e, edges):
        """First action of the shortest believed-open path from s_now to the
        nearest endpoint of e. If already at an endpoint, the action THROUGH
        the edge is returned (that move is the test)."""
        x, y = e
        targets = {x[0] * self.W + x[1], y[0] * self.W + y[1]}
        start = s_now
        if start in targets:
            # move along the edge itself
            for a in range(4):
                nb = neighbor_cell((start // self.W, start % self.W), a)
                if nb is None:
                    continue
                n_idx = nb[0] * self.W + nb[1]
                if n_idx in targets and edges.b_open(start, a) > 0.0:
                    # even believed-blocked: attempting IS the probe
                    return a
            return None
        # BFS over believed-open edges
        prev: dict[int, tuple[int, int]] = {start: (-1, -1)}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in targets:
                break
            r, c = cur // self.W, cur % self.W
            for a in range(4):
                nb = neighbor_cell((r, c), a)
                if nb is None:
                    continue
                n_idx = nb[0] * self.W + nb[1]
                if n_idx in prev:
                    continue
                if edges.b_open(cur, a) > 0.5:  # only believed-open edges
                    prev[n_idx] = (cur, a)
                    q.append(n_idx)
        if cur not in targets:
            return None
        # walk back to start
        node = cur
        while prev[node][0] != start and prev[node][0] != -1:
            node = prev[node][0]
        return prev[node][1] if prev[node][0] == start else None
