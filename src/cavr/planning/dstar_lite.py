"""D* Lite incremental search (Koenig & Likhachev 2002) on the AGENT's
believed map — goal-rooted formulation ("D* Lite", optimized first version).

g values are cost-to-goal; the priority keys use the heuristic between the
AGENT's current position and each vertex, h(s_cur, u), with the standard km
correction as the agent moves. Unknown edges use an explicitly defined
optimistic prior (b0 = 0.6 -> believed open); believed-blocked edges cost INF.
Cost queries always read the CURRENT edge beliefs, so belief flips propagate
through update_vertex on the affected endpoints. A full-rebuild fallback
guards against queue inconsistencies.
"""
from __future__ import annotations

import heapq

import numpy as np

INF = 1e6


def manhattan(a: int, b: int, W: int) -> int:
    return abs(a // W - b // W) + abs(a % W - b % W)


class DStarLite:
    def __init__(self, H: int, W: int, goal: int, edges, s_start: int = 0):
        self.H = H
        self.W = W
        self.n = H * W
        self.goal = int(goal)
        self.edges = edges
        self.g = np.full(self.n, INF)
        self.rhs = np.full(self.n, INF)
        self.km = 0.0
        self.s_cur = int(s_start)
        self.s_last = int(s_start)
        self._heap: list = []
        self._key: dict[int, tuple] = {}
        self._tie = 0
        self.rhs[self.goal] = 0.0
        self.insert(self.goal)

    # ------------------------------------------------------------------ #
    def h(self, u: int) -> float:
        """Heuristic from the AGENT's current position to vertex u."""
        return float(manhattan(self.s_cur, u, self.W))

    def calc_key(self, u: int) -> tuple:
        k2 = min(self.g[u], self.rhs[u])
        return (k2 + self.h(u) + self.km, k2)

    def insert(self, u: int) -> None:
        k = self.calc_key(u)
        self._key[u] = k
        heapq.heappush(self._heap, (k[0], k[1], self._tie, u))
        self._tie += 1

    def remove(self, u: int) -> None:
        self._key.pop(u, None)  # lazy deletion

    def _top(self):
        while self._heap:
            k1, k2, _tie, u = self._heap[0]
            if u not in self._key or self._key[u] != (k1, k2):
                heapq.heappop(self._heap)
                continue
            return (k1, k2), u
        return None, None

    def succ(self, s: int):
        r, c = s // self.W, s % self.W
        for a, (dr, dc) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W:
                yield a, nr * self.W + nc

    def cost(self, s: int, a: int, nb: int) -> float:
        return 1.0 if self.edges.b_open(s, a) > 0.5 else INF

    def update_vertex(self, u: int) -> None:
        if u != self.goal:
            best = INF
            for a, nb in self.succ(u):
                c = self.cost(u, a, nb)
                if c + self.g[nb] < best:
                    best = c + self.g[nb]
            self.rhs[u] = best
        if u in self._key:
            self.remove(u)
        if self.g[u] != self.rhs[u]:
            self.insert(u)

    # ------------------------------------------------------------------ #
    def compute_shortest_path(self, max_iter: int = 20000) -> bool:
        """Returns True when consistent, False when the iteration guard hit
        (caller should fall back to a rebuild)."""
        it = 0
        while True:
            top_key, u = self._top()
            start_key = self.calc_key(self.s_cur)
            if top_key is None:
                return True
            if not (top_key < start_key or self.rhs[self.s_cur] != self.g[self.s_cur]):
                return True
            it += 1
            if it > max_iter:
                return False
            k_new = self.calc_key(u)
            self.remove(u)  # popped: out of the queue (prevents re-processing)
            if top_key < k_new:
                self.insert(u)  # key changed: reinsert with updated key
            elif self.g[u] > self.rhs[u]:
                # overconsistent: lower g and propagate to neighbors
                self.g[u] = self.rhs[u]
                for _a, nb in self.succ(u):
                    self.update_vertex(nb)
            else:
                # underconsistent: invalidate and recompute through neighbors
                self.g[u] = INF
                for _a, nb in self.succ(u):
                    self.update_vertex(nb)
                self.update_vertex(u)

    def rebuild(self) -> None:
        self.g[:] = INF
        self.rhs[:] = INF
        self.rhs[self.goal] = 0.0
        self._heap = []
        self._key = {}
        self.insert(self.goal)
        self.compute_shortest_path(max_iter=200000)

    # ------------------------------------------------------------------ #
    def move_agent(self, s_new: int) -> None:
        if self.s_last != s_new:
            self.km += manhattan(self.s_last, s_new, self.W)
            self.s_last = s_new
        self.s_cur = s_new

    def notify_cost_change(self, u: int, v: int) -> None:
        self.update_vertex(u)
        self.update_vertex(v)

    def next_action(self) -> int | None:
        if not self.compute_shortest_path():
            self.rebuild()
        s = self.s_cur
        best_a, best_val = None, INF
        for a, nb in self.succ(s):
            val = self.cost(s, a, nb) + self.g[nb]
            if val < best_val:
                best_a, best_val = a, val
        if best_val >= INF:
            return None
        return best_a
