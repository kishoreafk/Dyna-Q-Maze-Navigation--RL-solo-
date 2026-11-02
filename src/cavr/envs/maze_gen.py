"""Procedural maze generation with saved seeds.

Mazes are generated locally (randomized depth-first spanning tree + braiding
to create loops). No external dataset files are used or bundled.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Actions: N=0, E=1, S=2, W=3
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPPOSITE = (2, 3, 0, 1)


def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Canonical undirected edge id from two cells."""
    return (a, b) if a <= b else (b, a)


def in_bounds(r: int, c: int, H: int, W: int) -> bool:
    return 0 <= r < H and 0 <= c < W


def neighbor_cell(cell: tuple[int, int], a: int) -> tuple[int, int] | None:
    r, c = cell
    dr, dc = DELTAS[a]
    return (r + dr, c + dc)


def generate_maze(H: int, W: int, seed: int, braid: float = 0.15) -> dict:
    """Generate a braided perfect maze.

    Returns a maze dict {H, W, seed, braid, open: [edge keys sorted]} where an
    open edge means the two adjacent cells are connected (no wall).
    """
    rng = np.random.default_rng(seed)
    visited = np.zeros((H, W), dtype=bool)
    open_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    # Iterative recursive-backtracker spanning tree.
    start = (0, 0)
    visited[start] = True
    stack = [start]
    while stack:
        cell = stack[-1]
        options = []
        for a in range(4):
            nb = neighbor_cell(cell, a)
            if nb is not None and in_bounds(*nb, H, W) and not visited[nb]:
                options.append(nb)
        if not options:
            stack.pop()
            continue
        nxt = options[int(rng.integers(len(options)))]
        open_edges.add(edge_key(cell, nxt))
        visited[nxt] = True
        stack.append(nxt)

    # Braiding: open additional internal walls to create loops/shortcuts.
    for r in range(H):
        for c in range(W):
            for a in (1, 2):  # E and S suffice for undirected edges
                nb = neighbor_cell((r, c), a)
                if nb is None or not in_bounds(*nb, H, W):
                    continue
                e = edge_key((r, c), nb)
                if e not in open_edges and rng.random() < braid:
                    open_edges.add(e)

    return {
        "H": H,
        "W": W,
        "seed": int(seed),
        "braid": float(braid),
        "open": sorted([list(map(tuple, e)) for e in open_edges]),
    }


def maze_open_set(maze: dict) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    return {edge_key(tuple(e[0]), tuple(e[1])) for e in maze["open"]}


def bfs_dist(open_edges: set, H: int, W: int, src: tuple[int, int]) -> dict:
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt_frontier = []
        for cell in frontier:
            for a in range(4):
                nb = neighbor_cell(cell, a)
                if nb is None or not in_bounds(*nb, H, W):
                    continue
                if edge_key(cell, nb) in open_edges and nb not in dist:
                    dist[nb] = dist[cell] + 1
                    nxt_frontier.append(nb)
        frontier = nxt_frontier
    return dist


def shortest_path(open_edges: set, H: int, W: int, src: tuple[int, int], dst: tuple[int, int]) -> list:
    """BFS shortest path as list of cells (empty list if unreachable)."""
    if src == dst:
        return [src]
    prev = {src: None}
    frontier = [src]
    while frontier:
        nxt_frontier = []
        for cell in frontier:
            for a in range(4):
                nb = neighbor_cell(cell, a)
                if nb is None or not in_bounds(*nb, H, W):
                    continue
                if edge_key(cell, nb) in open_edges and nb not in prev:
                    prev[nb] = cell
                    if nb == dst:
                        path = [nb]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    nxt_frontier.append(nb)
        frontier = nxt_frontier
    return []


def connected(open_edges: set, H: int, W: int, src, dst) -> bool:
    return dst in bfs_dist(open_edges, H, W, src)


def maze_sha256(maze: dict) -> str:
    import hashlib
    import json
    canon = json.dumps(
        {"H": maze["H"], "W": maze["W"], "seed": maze["seed"], "braid": maze["braid"],
         "open": [[list(e[0]), list(e[1])] for e in sorted(map(lambda x: edge_key(tuple(x[0]), tuple(x[1])), maze["open"]))]},
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
