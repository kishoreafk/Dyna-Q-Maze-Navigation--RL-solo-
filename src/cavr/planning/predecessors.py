"""Predecessor graph and model-epoch-versioned priority queue.

Heap entries are tagged with the model epoch at push time; after local
forgetting bumps the epoch, stale entries are discarded on pop so outdated
priorities are never executed (spec test_priority_version).
"""
from __future__ import annotations

import heapq
from collections import defaultdict


class PredecessorGraph:
    """s2 -> set of (sp, ap) pairs whose learned model puts mass on reaching
    s2 from sp. Self-loops are excluded to avoid self-reinforcing updates."""

    def __init__(self, n_states: int):
        self.n_states = n_states
        self.preds: dict[int, set] = defaultdict(set)
        self._contrib: dict[tuple, set] = {}

    def sync_pair(self, s: int, a: int, model) -> None:
        """Recompute the predecessor contribution of (s,a) from the current
        model (call after any model change affecting (s,a))."""
        old = self._contrib.pop((s, a), None)
        if old:
            for s2 in old:
                self.preds[s2].discard((s, a))
        new = set()
        p = model.predict(s, a)
        for k in range(p.shape[0]):
            if p[k] < 0.05:
                continue
            s2 = int(model.outcomes[s, a, k])
            if s2 != s:
                new.add(s2)
        self._contrib[(s, a)] = new
        for s2 in new:
            self.preds[s2].add((s, a))

    def get(self, s2: int):
        return self.preds.get(s2, ())


class VersionedPriorityQueue:
    def __init__(self):
        self._heap: list = []
        self._tie = 0
        self.n_pushes = 0
        self.n_pops = 0
        self.n_stale = 0

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, s: int, a: int, priority: float, epoch: int) -> None:
        heapq.heappush(self._heap, (-float(priority), self._tie, int(s), int(a), int(epoch)))
        self._tie += 1
        self.n_pushes += 1

    def top_priority(self) -> float:
        return -self._heap[0][0] if self._heap else 0.0

    def pop(self, cur_epoch: int, min_priority: float):
        """Pop the highest-priority non-stale entry with priority >=
        min_priority. Returns None when the useful work is exhausted."""
        while self._heap:
            negp, _tie, s, a, ep = heapq.heappop(self._heap)
            self.n_pops += 1
            pr = -negp
            if ep != cur_epoch:
                self.n_stale += 1
                continue
            if pr < min_priority:
                # everything below this priority is not worth executing
                return None
            return (s, a, pr)
        return None
