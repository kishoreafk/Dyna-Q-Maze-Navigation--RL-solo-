"""Episodic memory of observed real transitions (ESWM-informed principle:
independent one-step memories that can be updated and cleared locally).

Never merged across maze IDs (a fresh map is created per maze in the runner).
"""
from __future__ import annotations

from collections import defaultdict, deque


class EpisodicMap:
    def __init__(self, capacity_per_sa: int = 4):
        self.capacity = int(capacity_per_sa)
        self._rec: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=self.capacity))

    def append(self, s: int, a: int, r: float, s2: int, t: int) -> None:
        self._rec[(s, a)].append((s2, float(r), int(t)))

    def latest(self, s: int, a: int):
        d = self._rec.get((s, a))
        return d[-1] if d else None

    def clear_region(self, cells, keep_latest: bool = True) -> int:
        """Forget episodic records for (s,a) with s in cells. When
        keep_latest, retain only the most recent record (the latest
        contradictory observation must survive local reset)."""
        n = 0
        for (s, a), d in list(self._rec.items()):
            if s in cells:
                n += len(d)
                if keep_latest and d:
                    last = d[-1]
                    d.clear()
                    d.append(last)
                    n -= 1
                else:
                    d.clear()
        return n

    def visited_pairs(self):
        return list(self._rec.keys())

    def state_dict(self) -> dict:
        return {
            "records": {f"{s},{a}": list(v) for (s, a), v in self._rec.items()}
        }

    def load_state_dict(self, d: dict) -> None:
        self._rec = defaultdict(lambda: deque(maxlen=self.capacity))
        for key, recs in d.get("records", {}).items():
            s, a = key.split(",")
            for (s2, r, t) in recs:
                self.append(int(s), int(a), float(r), int(s2), int(t))
