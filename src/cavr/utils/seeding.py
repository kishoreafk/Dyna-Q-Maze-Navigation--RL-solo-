"""Deterministic seeding helpers."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def run_seed(base_seed: int, *parts) -> int:
    """Derive a stable 31-bit seed from a base seed and arbitrary parts."""
    key = "|".join(str(p) for p in parts)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
