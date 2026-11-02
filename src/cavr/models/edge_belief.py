"""Edge-factorized structural belief model.

Maintains b_e = P(undirected edge e is open) as log-odds, shared across both
travel directions (symmetry by construction: one canonical key per edge).
Predictions marginalize the latent executed direction under action slip rho,
so a single self-loop at rho>0 is weak, ambiguous evidence and never collapses
the belief (spec 6.2/6.3). Beliefs decay lazily toward the prior.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.maze_gen import DELTAS, edge_key, neighbor_cell

LOGIT_MIN, LOGIT_MAX = -3.5, 3.5  # b in ~[0.03, 0.97]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


class EdgeBeliefModel:
    def __init__(self, H: int, W: int, b0: float = 0.6, rho: float = 0.0,
                 lr: float = 0.4, decay_per_step: float = 0.002,
                 inner_steps: int = 3):
        self.H = H
        self.W = W
        self.b0 = float(b0)
        self.rho = float(rho)  # calibrated slip estimate (not the true hidden one)
        self.lr = float(lr)
        self.decay = float(decay_per_step)
        self.inner_steps = int(inner_steps)
        self._logit0 = _logit(b0)
        self._logit: dict[tuple, float] = {}
        self._b_cache: dict[tuple, float] = {}   # b = sigmoid(logit), kept in sync
        self._last_t: dict[tuple, int] = {}

    # ------------------------------------------------------------------ #
    def b(self, e) -> float:
        e = edge_key(tuple(e[0]), tuple(e[1]))
        v = self._b_cache.get(e)
        return v if v is not None else float(1.0 / (1.0 + np.exp(-self._logit0)))

    def _set_logit(self, e, logit: float) -> None:
        self._logit[e] = logit
        self._b_cache[e] = float(1.0 / (1.0 + np.exp(-logit)))

    def b_open(self, s: int, a: int) -> float:
        cell = (s // self.W, s % self.W)
        nb = neighbor_cell(cell, a)
        if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
            return 0.0
        return self.b(edge_key(cell, nb))

    def map_open(self, s: int, a: int) -> bool:
        return self.b_open(s, a) > 0.5

    def set_b(self, e, value: float) -> None:
        e = edge_key(tuple(e[0]), tuple(e[1]))
        self._set_logit(e, float(np.clip(_logit(value), LOGIT_MIN, LOGIT_MAX)))

    def edges(self):
        return sorted(self._logit.keys())

    # ------------------------------------------------------------------ #
    def _decay_edge(self, e, t: int) -> None:
        if e in self._logit and e in self._last_t:
            dt = max(0, t - self._last_t[e])
            if dt > 0:
                f = float(np.exp(-self.decay * dt))
                self._set_logit(e, self._logit0 + (self._logit[e] - self._logit0) * f)
        self._last_t[e] = t

    def predict_mixture(self, s: int, a: int, rho: Optional[float] = None) -> dict[int, float]:
        """P(s'|s,a) marginalizing the latent executed direction u under slip.

        Returns a dict {next_state_idx: prob} that always sums to 1.
        """
        rho = self.rho if rho is None else rho
        cell = (s // self.W, s % self.W)
        out: dict[int, float] = {s: 0.0}
        for u in range(4):
            pu = (1.0 - rho) if u == a else rho / 3.0
            if pu == 0.0:
                continue
            nb = neighbor_cell(cell, u)
            if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                out[s] += pu  # out of bounds -> stay
                continue
            be = self.b(edge_key(cell, nb))
            out[s] += pu * (1.0 - be)
            out[nb[0] * self.W + nb[1]] = out.get(nb[0] * self.W + nb[1], 0.0) + pu * be
        return out

    def update(self, s: int, a: int, s2: int, t: int) -> None:
        """Gradient ascent on log-odds of the involved edges given the observed
        next state, marginalizing over the latent executed direction. A small
        number of inner steps approximates a stronger Bayes update so that a
        blocked edge belief crosses the MAP boundary in ~2 contradictory
        observations while stationary slip still leaves beliefs diffuse."""
        rho = self.rho
        cell = (s // self.W, s % self.W)
        for _inner in range(self.inner_steps):
            obs = {s2: 1.0}
            # mixture likelihood
            lik = 0.0
            contrib = []  # (edge, pu, move_state)
            for u in range(4):
                pu = (1.0 - rho) if u == a else rho / 3.0
                if pu == 0.0:
                    continue
                nb = neighbor_cell(cell, u)
                if nb is None or not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                    lik += pu * (1.0 if s2 == s else 0.0)
                    continue
                e = edge_key(cell, nb)
                move_state = nb[0] * self.W + nb[1]
                lik += pu * ((1.0 - self.b(e)) if s2 == s else (self.b(e) if s2 == move_state else 0.0))
                contrib.append((e, pu, move_state))
            lik = max(lik, 1e-8)
            # gradient ascent on log-likelihood w.r.t. each edge logit
            acc: dict[tuple, float] = {}
            for (e, pu, move_state) in contrib:
                be = self.b(e)
                if s2 == s:
                    dlik_dbe = -pu
                elif s2 == move_state:
                    dlik_dbe = pu
                else:
                    dlik_dbe = 0.0
                if dlik_dbe == 0.0:
                    continue
                dloglik = dlik_dbe / lik
                dlogit = dloglik * be * (1.0 - be)
                acc[e] = acc.get(e, 0.0) + dlogit
            if not acc:
                break
            for e, g in acc.items():
                self._decay_edge(e, t)
                self._set_logit(e, float(np.clip(self._logit.get(e, self._logit0) + self.lr * g,
                                                 LOGIT_MIN, LOGIT_MAX)))

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        keys = sorted(self._logit.keys())
        return {
            "edge_keys": [(k[0][0] * self.W + k[0][1], k[1][0] * self.W + k[1][1]) for k in keys],
            "edge_logits": np.asarray([self._logit[k] for k in keys], dtype=np.float64),
            "b0": np.float64(self.b0),
            "rho_est": np.float64(self.rho),
        }

    def load_state_dict(self, d: dict) -> None:
        self._logit = {}
        self._b_cache = {}
        self._last_t = {}
        for (a_idx, b_idx), lg in zip(d["edge_keys"], np.asarray(d["edge_logits"])):
            a = (int(a_idx) // self.W, int(a_idx) % self.W)
            b = (int(b_idx) // self.W, int(b_idx) % self.W)
            self._set_logit(edge_key(a, b), float(lg))
        self.b0 = float(d.get("b0", self.b0))
        self.rho = float(d.get("rho_est", self.rho))
