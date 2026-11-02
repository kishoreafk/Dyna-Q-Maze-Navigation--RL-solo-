"""Change scenarios: deterministic, predeclared edge toggles.

The scenario planner uses the TRUE map to choose which edge to toggle; the
resulting schedule (absolute step + edges) is private to the environment and
the evaluator. Agents never see it.
"""
from __future__ import annotations

from typing import Optional

from .maze_gen import (
    bfs_dist,
    connected,
    edge_key,
    neighbor_cell,
    shortest_path,
)

Change = dict  # {"t": int, "edges": [((r1,c1),(r2,c2), open_bool)], "kind": str}


def find_blocking_edge(open_edges, H, W, start, goal, rng) -> Optional[Change]:
    """Pick an on-route open edge whose closure keeps the goal reachable and
    increases the shortest path length."""
    path = shortest_path(open_edges, H, W, start, goal)
    if len(path) < 2:
        return None
    base_len = len(path) - 1
    candidates = []
    seen = set()
    # Consider edges of the primary shortest path plus a few near-optimal ones.
    for i in range(len(path) - 1):
        e = edge_key(path[i], path[i + 1])
        if e in seen:
            continue
        seen.add(e)
        reduced = open_edges - {e}
        if not connected(reduced, H, W, start, goal):
            continue
        new_len = len(shortest_path(reduced, H, W, start, goal)) - 1
        if new_len > base_len:
            candidates.append((new_len - base_len, e))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top = [e for d, e in candidates if d == candidates[0][0]]
    e = top[int(rng.integers(len(top)))]
    return {"t": None, "edges": [(e[0], e[1], False)], "kind": "blocking"}


def find_shortcut_edge(open_edges, H, W, start, goal, rng) -> Optional[Change]:
    """Pick a closed internal edge whose opening reduces the shortest path
    (largest reduction first; falls back to any positive reduction)."""
    base_len = len(shortest_path(open_edges, H, W, start, goal)) - 1
    closed = []
    for r in range(H):
        for c in range(W):
            for a in (1, 2):
                nb = neighbor_cell((r, c), a)
                if nb is None or not (0 <= nb[0] < H and 0 <= nb[1] < W):
                    continue
                e = edge_key((r, c), nb)
                if e not in open_edges:
                    closed.append(e)
    candidates = []
    for e in closed:
        aug = open_edges | {e}
        new_len = len(shortest_path(aug, H, W, start, goal)) - 1
        if base_len - new_len >= 1:
            candidates.append((base_len - new_len, e))
    if not candidates:
        # Fallback: no edge strictly shortens the start-goal path (fully
        # braided maze). Open the closed edge that most improves the goal
        # distance from any cell — a frontier-opening with the same
        # discovery semantics (previously blocked edge becomes open).
        if not closed:
            return None
        best = None
        for e in closed:
            aug = open_edges | {e}
            gain = 0
            for (r, c) in [(r, c) for r in range(H) for c in range(W)]:
                d0 = len(shortest_path(open_edges, H, W, (r, c), goal))
                d1 = len(shortest_path(aug, H, W, (r, c), goal))
                gain += max(0, d0 - d1)
            if best is None or gain > best[0]:
                best = (gain, e)
        e = best[1]
        return {"t": None, "edges": [(e[0], e[1], True)], "kind": "shortcut"}
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top = [e for d, e in candidates if d == candidates[0][0]]
    e = top[int(rng.integers(len(top)))]
    return {"t": None, "edges": [(e[0], e[1], True)], "kind": "shortcut"}


def build_schedule(maze_open, H, W, start, goal, scenario: str, t_change: int,
                   second_change_frac: float, rng) -> list[Change]:
    """Build the deterministic change schedule (absolute times).

    - static: no changes
    - blocking: close an on-route edge at t_change
    - shortcut: open a previously blocked edge at t_change
    - oscillation: blocking at t_change, then re-open (restore) at second time
    """
    schedule: list[Change] = []
    if scenario == "static":
        return schedule
    if scenario == "blocking":
        ch = find_blocking_edge(maze_open, H, W, start, goal, rng)
        if ch is None:
            raise ValueError("no valid blocking edge for this maze")
        ch["t"] = int(t_change)
        schedule.append(ch)
        return schedule
    if scenario == "shortcut":
        ch = find_shortcut_edge(maze_open, H, W, start, goal, rng)
        if ch is None:
            raise ValueError("no valid shortcut edge for this maze")
        ch["t"] = int(t_change)
        schedule.append(ch)
        return schedule
    if scenario == "oscillation":
        ch = find_blocking_edge(maze_open, H, W, start, goal, rng)
        if ch is None:
            raise ValueError("no valid blocking edge for this maze")
        ch["t"] = int(t_change)
        (a, b, _) = ch["edges"][0]
        reopen = {"t": int(t_change + (1.0 - second_change_frac) * t_change),
                  "edges": [(a, b, True)], "kind": "reopen"}
        schedule.extend([ch, reopen])
        return schedule
    raise ValueError(f"unknown scenario {scenario}")
