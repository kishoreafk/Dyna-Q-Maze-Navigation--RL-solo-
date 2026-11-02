#!/usr/bin/env python3
"""Create the maze manifest (train/val/test splits by maze seed, SHA256)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cavr.envs.maze_gen import generate_maze, maze_sha256  # noqa: E402


def make_manifest(H: int, W: int, braid: float, maze_seed_start: int,
                  n_train: int, n_val: int, n_test: int, out_path: str) -> dict:
    mazes = []
    for i, split in enumerate(["train"] * n_train + ["val"] * n_val + ["test"] * n_test):
        seed = maze_seed_start + i
        maze = generate_maze(H, W, seed, braid)
        mazes.append({
            "seed": seed,
            "split": split,
            "H": H, "W": W, "braid": braid,
            "sha256": maze_sha256(maze),
        })
    manifest = {
        "generator": "procedural: randomized DFS spanning tree + braiding (no external dataset files)",
        "splits_by_maze_identity": True,
        "mazes": mazes,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--out", default="data/manifests/mazes_manifest.json")
    args = ap.parse_args()
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    env, splits = cfg["env"], cfg["splits"]
    manifest = make_manifest(env["H"], env["W"], env.get("braid", 0.15),
                             splits["maze_seed_start"], splits["n_train"],
                             splits["n_val"], splits["n_test"], args.out)
    print(f"manifest written: {args.out} ({len(manifest['mazes'])} mazes, "
          f"splits: train={splits['n_train']}, val={splits['n_val']}, test={splits['n_test']})")


if __name__ == "__main__":
    main()
