"""Value-of-computation style budget controller (heuristic, rule-based).

Chooses the number of expected Bellman backups B in {0, 8, 16, 32, 64} from
PUBLIC features only (alarm strength, repair priorities, model uncertainty,
queue state). Hard cap 64. The rules and thresholds are documented in
reports/decisions.md; a learned gating MLP is explicitly out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetFeatures:
    alarm_strength: float = 0.0
    recent_alarm: bool = False
    top_repair_priority: float = 0.0
    heap_size: int = 0
    mean_epistemic: float = 0.5


class BudgetController:
    LEVELS = (0, 8, 16, 32, 64)

    def __init__(self, cap: int = 64, theta_high: float = 0.15,
                 theta_lo: float = 0.10, fixed_b: int | None = None):
        self.cap = int(cap)
        self.theta_high = float(theta_high)
        self.theta_lo = float(theta_lo)
        self.fixed_b = fixed_b  # ablation: fixed budget

    def choose(self, f: BudgetFeatures) -> tuple[int, str]:
        if self.fixed_b is not None:
            return min(self.fixed_b, self.cap), "fixed"
        if f.recent_alarm:
            B, why = 64, "post_alarm_repair"
        elif f.alarm_strength >= 0.5:
            B, why = 32, "elevated_alarm"
        elif f.top_repair_priority >= self.theta_high:
            B, why = 16, "high_value_repair_targets"
        elif f.heap_size == 0 and f.mean_epistemic < self.theta_lo:
            B, why = 0, "converged_local_model"
        else:
            B, why = 8, "maintenance"
        return min(B, self.cap), why
