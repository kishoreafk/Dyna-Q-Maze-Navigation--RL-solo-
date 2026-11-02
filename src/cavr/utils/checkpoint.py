"""Checkpoint I/O for tabular agents ("model weights" = Q table, Dirichlet
counts, edge-belief logits, detector state). Saved as .npz + embedded JSON
metadata."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def save_checkpoint(path: str, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(meta)
    # Split edge dict payloads (variable length) into flat arrays.
    for key in ("edge_keys", "pchange_keys"):
        if key in meta:
            pairs = meta.pop(key)  # list of (s1, s2)
            if pairs:
                meta[key + "_a"] = np.asarray([p[0] for p in pairs], dtype=np.int64)
                meta[key + "_b"] = np.asarray([p[1] for p in pairs], dtype=np.int64)
            else:
                meta[key + "_a"] = np.zeros((0,), dtype=np.int64)
                meta[key + "_b"] = np.zeros((0,), dtype=np.int64)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta, default=float).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(path, **payload)
    return path


def load_checkpoint(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    meta_bytes = data["__meta__"].tobytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    arrays = {k: v for k, v in data.items() if k != "__meta__"}
    for key in ("edge_keys", "pchange_keys"):
        if key + "_a" in meta:
            a = np.asarray(meta.pop(key + "_a"))
            b = np.asarray(meta.pop(key + "_b"))
            meta[key] = list(zip(a.tolist(), b.tolist()))
    return arrays, meta
