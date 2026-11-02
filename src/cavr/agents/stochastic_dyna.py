"""Stochastic Dyna family with Dirichlet categorical transition models.

- DirichletDynaAgent: expected (full-distribution) Bellman backups, uniform
  replay over visited pairs, NO change detector (baseline 5).
- LocalForgettingAgent: same model + predictive-surprise detector; on reliable
  alarms, obsolete local evidence is discarded while the most recent real
  transition is retained exactly once (baseline 6).
"""
from __future__ import annotations

import numpy as np

from ..models.categorical_dirichlet import CategoricalDirichlet, category_of
from ..models.episodic_map import EpisodicMap
from ..models.change_detector import ChangeDetector
from .base import BaseAgent


class StochasticDynaBase(BaseAgent):
    name = "stochastic_dyna"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng, rho_est: float = 0.1,
                 use_detector: bool = False, do_reset: bool = False):
        super().__init__(n_states, W, H, goal_idx, cfg, rng)
        self.alpha_model = float(cfg.get("alpha_model", 0.15))
        self.beta = float(cfg.get("beta", 0.5))
        self.model = CategoricalDirichlet(n_states, W, beta=self.beta)
        self.memory = EpisodicMap(capacity_per_sa=4)
        self.detector = ChangeDetector(h=float(cfg.get("h", 3.0)),
                                       drift=float(cfg.get("drift", 0.01)),
                                       cooldown=int(cfg.get("cooldown", 15))
                                       ) if use_detector else None
        self.do_reset = do_reset
        self.rho_est = rho_est
        self._pairs: list[tuple] = []
        self._pair_set: set = set()
        self.n_replay = int(cfg.get("dyna_n", 20))
        # incremental value cache (V[goal] kept at 0)
        self._V = np.zeros(n_states, dtype=np.float64)

    def _after_q_update(self, s: int) -> None:
        v = float(self.q[s].max())
        self._V[s] = 0.0 if s == self.goal_idx else v

    # ------------------------------------------------------------------ #
    def _add_pair(self, s, a):
        key = (s, a)
        if key not in self._pair_set:
            self._pair_set.add(key)
            self._pairs.append(key)

    def expected_target(self, s0: int, a0: int) -> float:
        """Expected model backup target integrating the FULL predictive
        distribution (not the last outcome)."""
        p = self.model.predict(s0, a0)
        rm = self.model.reward_mean(s0, a0)
        s2s = self.model.outcomes[s0, a0]
        return float((p * (rm + self.gamma * self._V[s2s])).sum())

    def expected_backup(self, s0: int, a0: int) -> float:
        targ = self.expected_target(s0, a0)
        delta = targ - float(self.q[s0, a0])
        w_conf = 1.0 - self.model.epistemic(s0, a0)
        self.q[s0, a0] += self.alpha_model * w_conf * delta
        self._after_q_update(s0)
        return delta

    def localize_cells(self, s: int, a: int, s2: int) -> set:
        return {s}

    def forget_region(self, cells, cur_t: int) -> None:
        """Discard old local evidence; re-add the latest PREVIOUS record per
        pair (records with t == cur_t are skipped — the new transition is
        incorporated exactly once right after this call)."""
        for sp in cells:
            for ap in range(4):
                self.model.reset_sa(sp, ap)
        for sp in cells:
            for ap in range(4):
                rec = self.memory.latest(sp, ap)
                if rec is not None and rec[2] != cur_t:
                    self.model.reobserve(sp, ap, rec[0], rec[1])

    def _replay(self, n: int) -> int:
        executed = 0
        if self._pairs:
            for _ in range(n):
                s0, a0 = self._pairs[int(self.rng.integers(len(self._pairs)))]
                self.expected_backup(s0, a0)
                executed += 1
        self.totals["backups"] += executed
        return executed

    # ------------------------------------------------------------------ #
    def observe(self, s, a, r, s2, terminal, t):
        # 1. frozen PRE-update prediction -> surprise
        p_before = self.model.predict(s, a).copy()
        k = category_of(s, s2, self.W)
        if self.detector is not None:
            z, g_inc = self.detector.score_pre_update(p_before, k)
        else:
            z, g_inc = 0.0, 0.0
        # 2. exactly one direct TD update
        delta = self.direct_update(s, a, r, s2, terminal)
        # 3. episodic memory
        self.memory.append(s, a, r, s2, t)
        self._add_pair(s, a)
        # 4. alarm confirmation
        alarm = False
        if self.detector is not None:
            alarm = bool(self.detector.confirm(g_inc, t, key=(s, a)))
            self.totals["alarms"] += int(alarm)
            if alarm and self.do_reset:
                self.forget_region(self.localize_cells(s, a, s2), cur_t=t)
        # 5. incorporate the new observation exactly once
        self.model.update(s, a, s2, r)
        # 6. replay
        executed = self._replay(self.n_replay)
        self.step_stats = {
            "backups_executed": executed, "budget_assigned": self.n_replay,
            "queue_operations": 0, "wall_time_ms": 0.0,
            "model_version": self.model.epoch,
            "model_uncertainty": float(self.model.epistemic(s, a)),
            "predicted_change_probability": float(self.detector.alarm_strength()) if self.detector else 0.0,
            "budget_reason": "n/a",
            "proposed_probe": -1, "actual_action": a,
            "alarm": alarm, "surprise": float(z), "direct_delta": delta}
        return self.step_stats


class DirichletDynaAgent(StochasticDynaBase):
    name = "dirichlet_dyna"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng, rho_est=0.1):
        super().__init__(n_states, W, H, goal_idx, cfg, rng,
                         use_detector=False, do_reset=False)


class LocalForgettingAgent(StochasticDynaBase):
    name = "local_forgetting"

    def __init__(self, n_states, W, H, goal_idx, cfg, rng, rho_est=0.1):
        super().__init__(n_states, W, H, goal_idx, cfg, rng, rho_est=rho_est,
                         use_detector=True, do_reset=True)
        self.n_replay = int(cfg.get("lf_n", 32))
