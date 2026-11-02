"""Categorical Dirichlet transition model over local next-state categories.

Categories K=5: {N=0, E=1, S=2, W=3, self=4}. The posterior mean is the
predicted categorical distribution P_M(s'|s,a); posterior variance provides
an EPISTEMIC uncertainty estimate that is kept strictly separate from the
predictive entropy (aleatoric mixture). The model is used for expected
(stochastic) Bellman backups and for pre-update surprise scoring.
"""
from __future__ import annotations

import numpy as np

K = 5
SELF = 4
# Absolute normalizer for epistemic uncertainty: var_sum for a posterior with
# uniform predictive over 5 categories and a0=3 equals 0.8/(a0+1); we anchor
# the scale so a completely unobserved (s,a) has epistemic ~= 1.0.
EPIS_NORM = 0.2


def category_of(s_idx: int, s2_idx: int, W: int) -> int:
    """Map an observed transition to a category. Invalid transitions raise."""
    if s2_idx == s_idx:
        return SELF
    d = s2_idx - s_idx
    if d == -W:
        return 0
    if d == +1:
        return 1
    if d == +W:
        return 2
    if d == -1:
        return 3
    raise ValueError(f"non-local transition {s_idx}->{s2_idx}")


def outcome_table(n_states: int, W: int) -> np.ndarray:
    """outcomes[s, a, k] -> resulting state index for category k.

    k<4 maps to the neighbor cell in that direction (or the cell itself when
    out of bounds / treated as blocked); k=4 maps to the cell itself.
    """
    H = n_states // W
    tab = np.zeros((n_states, 4, K), dtype=np.int64)
    for s in range(n_states):
        r, c = divmod(s, W)
        for a in range(4):
            for k in range(4):
                dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[k]
                nr, nc = r + dr, c + dc
                tab[s, a, k] = (nr * W + nc) if (0 <= nr < H and 0 <= nc < W) else s
            tab[s, a, SELF] = s
    return tab


class CategoricalDirichlet:
    def __init__(self, n_states: int, W: int, beta: float = 0.5):
        self.n_states = n_states
        self.W = W
        self.beta = float(beta)
        self.counts = np.full((n_states, 4, K), beta, dtype=np.float64)
        self.r_sum = np.zeros((n_states, 4, K), dtype=np.float64)
        self.r_cnt = np.zeros((n_states, 4, K), dtype=np.float64)
        self.epoch = 0  # change-epoch: bumped on local forgetting (heap staleness)
        self.outcomes = outcome_table(n_states, W)

    # ------------------------------------------------------------------ #
    def predict(self, s: int, a: int) -> np.ndarray:
        """Posterior mean predictive distribution over K categories (sums to 1)."""
        return self.counts[s, a] / self.counts[s, a].sum()

    def n_observations(self, s: int, a: int) -> float:
        return float(self.counts[s, a].sum() - K * self.beta)

    def epistemic(self, s: int, a: int) -> float:
        """Posterior variance based lack-of-knowledge in [0, 1].

        Var_k = a_k (a0 - a_k) / (a0^2 (a0 + 1)); sum over k vanishes as data
        accumulates even when the predictive stays multi-modal (this is what
        separates epistemic uncertainty from predictive entropy).
        """
        a = self.counts[s, a]
        a0 = a.sum()
        var_sum = float(((a * (a0 - a)) / (a0 * a0 * (a0 + 1.0))).sum())
        return min(1.0, var_sum / EPIS_NORM)

    def entropy(self, s: int, a: int) -> float:
        """Entropy of the predictive mean (aleatoric mixture, NOT epistemic)."""
        p = self.predict(s, a)
        p = np.clip(p, 1e-12, None)
        return float(-(p * np.log(p)).sum())

    # ------------------------------------------------------------------ #
    def update(self, s: int, a: int, s2: int, reward: float) -> None:
        k = category_of(s, s2, self.W)
        self.counts[s, a, k] += 1.0
        self.r_sum[s, a, k] += reward
        self.r_cnt[s, a, k] += 1.0

    def reset_sa(self, s: int, a: int) -> None:
        """Reset a single (s,a) to the prior."""
        self.counts[s, a] = self.beta
        self.r_sum[s, a] = 0.0
        self.r_cnt[s, a] = 0.0

    def reobserve(self, s: int, a: int, s2: int, reward: float) -> None:
        """Re-incorporate a kept observation after a reset (no double count)."""
        self.update(s, a, s2, reward)

    # ------------------------------------------------------------------ #
    def reward_mean(self, s: int, a: int) -> np.ndarray:
        """Mean observed reward per category with fixed priors for unseen
        outcomes: self-loop -0.03 (step+collision), movement -0.01."""
        r = np.where(self.r_cnt[s, a] > 0, self.r_sum[s, a] / np.maximum(self.r_cnt[s, a], 1.0), 0.0)
        unseen = self.r_cnt[s, a] == 0
        r[unseen] = -0.03
        # movement categories default -0.01 (only SELF keeps collision default)
        r[:SELF][unseen[:SELF]] = -0.01
        return r

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {
            "counts": self.counts.copy(),
            "r_sum": self.r_sum.copy(),
            "r_cnt": self.r_cnt.copy(),
            "epoch": np.int64(self.epoch),
        }

    def load_state_dict(self, d: dict) -> None:
        self.counts = np.array(d["counts"], dtype=np.float64)
        self.r_sum = np.array(d["r_sum"], dtype=np.float64)
        self.r_cnt = np.array(d["r_cnt"], dtype=np.float64)
        self.epoch = int(d["epoch"])
