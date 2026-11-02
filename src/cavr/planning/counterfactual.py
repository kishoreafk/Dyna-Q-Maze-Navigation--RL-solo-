"""Goal-sensitive counterfactual edge scoring and reachability weighting.

For an edge e between cells x,y with commanded direction a toward e, the
one-edge counterfactual value impact has a closed form under the mixture
model (all other beliefs held fixed):

    D_e = | P(u) * [ (r_move - r_stay) + gamma * (V(y) - V(x)) ] |

where P(u) is the probability that the executed direction is the edge
direction. This is a HEURISTIC planning-influence estimate, not an exact
value-of-information (spec caution 6.2).

Reachability d_pi is a short probability-mass rollout (H<=12) of the current
epsilon-greedy policy from the agent's state, using only learned/edge-model
predictions (no true map).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

R_MOVE = -0.01
R_STAY = -0.03


class CounterfactualScorer:
    def __init__(self, gamma: float, H: int, W: int, rho_est: float,
                 reach_H: int = 12, reach_disc: float = 0.9, reach_eps: float = 0.1):
        self.gamma = gamma
        self.H = H
        self.W = W
        self.rho_est = rho_est
        self.reach_H = reach_H
        self.reach_disc = reach_disc
        self.reach_eps = reach_eps

    def edge_impact(self, x: int, y: int, V: np.ndarray) -> float:
        """|value impact if edge x-y flipped open/closed| with other beliefs
        fixed; V[goal] must be 0."""
        gain = (R_MOVE - R_STAY) + self.gamma * (float(V[y]) - float(V[x]))
        # commanded toward the edge: P(u)=1-rho; slipped into it: rho/3 — take max
        p_u = max(1.0 - self.rho_est, self.rho_est / 3.0)
        return abs(p_u * gain)

    def reach_mass(self, s0: int, edges, q: np.ndarray, goal: int) -> dict[int, float]:
        """Discounted visitation mass from s0 under the current policy."""
        mass = {s0: 1.0}
        out: dict[int, float] = {}
        disc = 1.0
        for _ in range(self.reach_H):
            nxt: dict[int, float] = defaultdict(float)
            disc *= self.reach_disc
            for s, m in mass.items():
                if m * disc < 1e-4:
                    continue
                a_star = int(np.argmax(q[s]))
                for a in range(4):
                    pa = self.reach_eps / 4.0 + (1.0 - self.reach_eps) * (1.0 if a == a_star else 0.0)
                    if pa == 0.0:
                        continue
                    for s2, p in edges.predict_mixture(s, a).items():
                        if s2 == goal:
                            continue
                        nxt[s2] += m * pa * p
            mass = dict(nxt)
            for s, m in mass.items():
                out[s] = out.get(s, 0.0) + m
            if not mass:
                break
        return out
