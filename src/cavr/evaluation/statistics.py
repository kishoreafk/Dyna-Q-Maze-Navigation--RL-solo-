"""Paired bootstrap confidence intervals across map/seed units."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 10000,
                        seed: int = 0, ci: float = 0.95) -> dict:
    """95% paired bootstrap CI for the mean of paired differences."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(diff))}


def median_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    meds = np.median(values[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {"median": float(np.median(values)), "lo": float(lo), "hi": float(hi), "n": int(len(values))}
