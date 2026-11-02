"""Evaluator-only oracles: BFS shortest path and value iteration on the TRUE
MDP. These are used EXCLUSIVELY for evaluation metrics, never by agents
(spec section 4: oracle access only for evaluating regret/optimality)."""
from __future__ import annotations

import numpy as np

from ..envs.maze_gen import DELTAS, edge_key, neighbor_cell, shortest_path

STEP_R = -0.01
COLL_R = -0.03  # step + collision
GOAL_R = 1.0


def bfs_shortest_len(open_edges, H, W, src, dst) -> int:
    p = shortest_path(open_edges, H, W, src, dst)
    return max(0, len(p) - 1)


def value_iteration(open_edges, H: int, W: int, rho: float, gamma: float,
                    goal: tuple[int, int], iters: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Optimal V and greedy policy for the TRUE stochastic MDP (evaluator
    only). States indexed r*W+c; terminal goal has V=0."""
    n = H * W
    goal_idx = goal[0] * W + goal[1]
    V = np.zeros(n)
    pol = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        delta = 0.0
        for s in range(n):
            if s == goal_idx:
                continue
            r_, c_ = s // W, s % W
            best, best_a = -1e18, 0
            for a in range(4):
                val = 0.0
                for u in range(4):
                    pu = (1.0 - rho) if u == a else rho / 3.0
                    if pu == 0.0:
                        continue
                    dr, dc = DELTAS[u]
                    nr, nc = r_ + dr, c_ + dc
                    inb = 0 <= nr < H and 0 <= nc < W
                    nb = (nr * W + nc) if inb else s
                    blocked = (not inb) or (edge_key((r_, c_), (nr, nc)) not in open_edges)
                    s2 = s if blocked else nb
                    if s2 == goal_idx:
                        rew = GOAL_R
                    elif s2 == s:
                        rew = COLL_R
                    else:
                        rew = STEP_R
                    val += pu * (rew + gamma * V[s2])
                if val > best:
                    best, best_a = val, a
            delta = max(delta, abs(best - V[s]))
            V[s] = best
            pol[s] = best_a
        if delta < 1e-9:
            break
    return V, pol


def optimal_policy_rollout_len(open_edges, H, W, rho, gamma, goal, start,
                               rng, max_steps: int = 500) -> int:
    """Diagnostic: steps for the optimal policy to reach the goal (one
    rollout with the true slip). Evaluator-only."""
    V, pol = value_iteration(open_edges, H, W, rho, gamma, goal)
    goal_idx = goal[0] * W + goal[1]
    s = start[0] * W + start[1]
    for t in range(1, max_steps + 1):
        if s == goal_idx:
            return t - 1
        a = int(pol[s])
        u = a if rng.random() >= rho else [x for x in range(4) if x != a][int(rng.integers(3))]
        r_, c_ = s // W, s % W
        dr, dc = DELTAS[u]
        nr, nc = r_ + dr, c_ + dc
        inb = 0 <= nr < H and 0 <= nc < W
        blocked = (not inb) or (edge_key((r_, c_), (nr, nc)) not in open_edges)
        s = s if blocked else (nr * W + nc)
    return max_steps
