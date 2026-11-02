"""Grid environment with strictly private map state.

Observation contract (spec section 4/9): the agent receives only
(state coordinates, goal coordinates, action, reward, next coordinates,
terminated/truncated). The true edge set, change schedule, true slip rate and
sampled slipped direction are PRIVATE. `enable_leak_guard()` replaces the
private attribute names with raising properties; environment internals read
state through `self._state`, so the environment keeps working while any
external (agent) access to the guarded names fails loudly. This is used by
test_no_leakage.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .maze_gen import DELTAS, edge_key, in_bounds, maze_open_set
from .change_scenarios import build_schedule

STEP_REWARD = -0.01
COLLISION_REWARD = -0.02
GOAL_REWARD = 1.0


class LeakageError(RuntimeError):
    pass


class GridEnv:
    """Public-observation grid maze with scheduled private topology changes."""

    def __init__(self, maze: dict, goal: tuple[int, int], scenario: str,
                 rho: float, max_steps: int, t_change: int,
                 second_change_frac: float = 0.75, env_seed: int = 0,
                 start: tuple[int, int] = (0, 0)):
        self.H = int(maze["H"])
        self.W = int(maze["W"])
        self.goal = tuple(goal)
        self.max_steps = int(max_steps)
        self.rho = float(rho)  # agents may use a calibrated estimate, not the true one
        self._state = {
            "open": maze_open_set(maze),
            "schedule": build_schedule(
                maze_open_set(maze), self.H, self.W, tuple(start), tuple(goal),
                scenario, t_change, second_change_frac,
                np.random.default_rng(env_seed + 7777),
            ),
            "rho": float(rho),
            "rng": np.random.default_rng(env_seed),
            "events": [],
        }
        self._start_cell = tuple(start)
        self._s = self._start_cell
        self._t = 0            # global real-step counter (survives episode resets)
        self._ep_steps = 0
        self._guard = False

    # ------------------------------------------------------------------ #
    # public API (safe for everyone)
    # ------------------------------------------------------------------ #
    @property
    def n_states(self) -> int:
        return self.H * self.W

    def state_idx(self, cell) -> int:
        return cell[0] * self.W + cell[1]

    def reset(self, start: Optional[tuple[int, int]] = None) -> tuple[int, int]:
        self._s = tuple(start) if start is not None else self._start_cell
        self._ep_steps = 0
        return self._s

    def step(self, action: int):
        st = self._state
        self._t += 1
        self._ep_steps += 1
        self._apply_due_changes()
        a = int(action)
        # slip: executed direction u ~ {a: 1-rho; each other: rho/3}
        if st["rng"].random() >= st["rho"]:
            u = a
        else:
            others = [x for x in range(4) if x != a]
            u = others[int(st["rng"].integers(3))]
        nb = self._neighbor(self._s, u)
        collided = nb is None or edge_key(self._s, nb) not in st["open"]
        s2 = self._s if collided else nb
        terminated = s2 == self.goal
        if terminated:
            r = GOAL_REWARD
        else:
            r = STEP_REWARD + (COLLISION_REWARD if collided else 0.0)
        truncated = (not terminated) and self._ep_steps >= self.max_steps
        self._s = s2
        return s2, float(r), bool(terminated), bool(truncated), {}

    # ------------------------------------------------------------------ #
    # leak protection / evaluator access
    # ------------------------------------------------------------------ #
    def enable_leak_guard(self) -> None:
        """Turn the private attribute names into raising properties. The
        environment keeps working (internals use self._state); any external
        access to the hidden state now fails loudly."""
        if self._guard:
            return
        self._guard = True

        def _raise(_self):
            raise LeakageError("access to private environment attribute blocked")

        for name in ("_open", "_schedule", "_rho", "_rng", "_change_events"):
            setattr(GridEnv, name, property(_raise))

    def evaluator_state(self) -> dict:
        """Evaluator-only view. NEVER call from agent code."""
        st = self._state
        return {
            "t": self._t,
            "open_edges": sorted([list(map(list, e)) for e in st["open"]]),
            "schedule": [
                {"t": ch["t"], "kind": ch["kind"],
                 "edges": [[list(e[0]), list(e[1]), bool(e[2])] for e in ch["edges"]]}
                for ch in st["schedule"]
            ],
            "events": list(st["events"]),
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _neighbor(self, cell, a: int):
        dr, dc = DELTAS[a]
        nr, nc = cell[0] + dr, cell[1] + dc
        if in_bounds(nr, nc, self.H, self.W):
            return (nr, nc)
        return None

    def _apply_due_changes(self) -> None:
        st = self._state
        for ch in st["schedule"]:
            if ch["t"] == self._t:
                for (a, b, to_open) in ch["edges"]:
                    e = edge_key(tuple(a), tuple(b))
                    if to_open:
                        st["open"].add(e)
                    else:
                        st["open"].discard(e)
                st["events"].append({
                    "t": self._t, "kind": ch["kind"],
                    "edges": [list(map(list, e[:2])) for e in ch["edges"]],
                })
