#!/usr/bin/env python3
"""Run the pilot experiment suite: calibration -> canonical Sutton test ->
evaluation grid (resume-safe).

Usage:
  python scripts/run_pilot.py --config configs/pilot.yaml
  python scripts/run_pilot.py --config configs/pilot.yaml --phase grid --scenarios blocking
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cavr.agents import make_agent  # noqa: E402
from cavr.envs.grid_env import GridEnv  # noqa: E402
from cavr.envs.maze_gen import generate_maze  # noqa: E402
from cavr.utils.logging import append_csv, ensure_dir  # noqa: E402
from cavr.utils.seeding import run_seed, make_rng  # noqa: E402
from cavr.evaluation import (RunnerConfig, run_calibration, run_grid,  # noqa: E402
                             load_maze)


def phase_manifest(cfg: dict, manifest_path: str) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_manifest import make_manifest
    env, splits = cfg["env"], cfg["splits"]
    man = make_manifest(env["H"], env["W"], env.get("braid", 0.15),
                        splits["maze_seed_start"], splits["n_train"],
                        splits["n_val"], splits["n_test"], manifest_path)
    seeds = [m["seed"] for m in man["mazes"] if m["split"] == "val"]
    print(f"[manifest] {len(man['mazes'])} mazes -> {manifest_path}; val seeds: {seeds}")
    return None


def phase_calibration(cfg: dict, out_dir: str) -> dict:
    path = os.path.join(out_dir, "calibration_summary.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    env, cal, splits = cfg["env"], cfg["calibration"], cfg["splits"]
    val_seeds = [splits["maze_seed_start"] + splits["n_train"] + i
                 for i in range(splits["n_val"])]
    mazes = [load_maze(env["H"], env["W"], s, env.get("braid", 0.15)) for s in val_seeds]
    cal_cfg = {"goal": tuple(env["goal"]), "max_steps": env["max_steps"],
               "agent_cfg": cfg["agent"]}
    print("[calibration] CUSUM threshold search (no-change, rho=0.3)...")
    cusum_records = []
    for h in cal["h_grid"]:
        from cavr.evaluation.calibration import _false_alarms_for_h
        far = _false_alarms_for_h(cal_cfg, mazes, h, steps=cal["val_steps"])
        cusum_records.append({"h": h, "false_per_1000": far})
        print(f"  h={h}: FAR={far:.2f}/1000")
    ok = [r for r in cusum_records if r["false_per_1000"] <= cal["target_far_per_1000"] * 0.6]
    chosen_h = ok[0]["h"] if ok else cusum_records[-1]["h"]
    print(f"[calibration] chosen h={chosen_h} (target <={cal['target_far_per_1000']}/1000)")
    print("[calibration] Dyna-Q+ kappa sweep (validation blocking)...")
    from cavr.evaluation.calibration import calibrate_kappa
    kap = calibrate_kappa(cal_cfg, mazes[:2], kappa_grid=cal["kappa_grid"],
                          steps=cal["val_steps"] + 300,
                          out_path=os.path.join(out_dir, "calibration_kappa.json"))
    print(f"[calibration] chosen kappa={kap['chosen_kappa']}")
    record = {"cusum_rows": cusum_records, "chosen_h": chosen_h,
              "kappa": kap, "val_seeds": val_seeds}
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    with open(os.path.join(out_dir, "calibration_cusum.json"), "w") as fh:
        json.dump({"h_grid": cusum_records, "chosen_h": chosen_h}, fh, indent=2)
    return record


def phase_canonical(cfg: dict, out_dir: str) -> None:
    """Stage 0: Sutton-style 6x9 blocking->shortcut demonstration with
    Dyna-Q backup-cap sweep (qualitative reproduction, rho=0)."""
    can = cfg["canonical"]
    path = os.path.join(out_dir, "canonical.csv")
    if os.path.exists(path + ".done"):
        print("[canonical] already complete")
        return
    # A fresh clone ships canonical.csv but not the .done marker (markers are
    # runtime state and are gitignored).  Reusing it avoids appending duplicate
    # rows; --fresh removes the file when a true re-run is required.
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path + ".done", "w") as fh:
            fh.write("done")
        print("[canonical] existing canonical.csv found; reusing it "
              "(use --fresh to regenerate)")
        return
    ensure_dir(out_dir)
    print("[canonical] 6x9 blocking->shortcut, Dyna n sweep...")
    header = ["dyna_n", "seed", "ep", "end_step", "steps", "success"]
    for n in can["dyna_ns"]:
        for sd in range(can["seeds"]):
            seed = run_seed(cfg["seed_base"], "canonical", n, sd)
            maze = generate_maze(can["H"], can["W"], run_seed(seed, "maze"), 0.10)
            env = GridEnv(maze, (can["H"] - 1, can["W"] - 1), "blocking", 0.0,
                          can["steps"], can["block_t"], env_seed=run_seed(seed, "env"))
            # add a shortcut opening later (Sutton's two-change blocking task)
            from cavr.envs.change_scenarios import find_shortcut_edge
            from cavr.envs.maze_gen import maze_open_set
            sc = find_shortcut_edge(maze_open_set(maze), can["H"], can["W"],
                                    (0, 0), (can["H"] - 1, can["W"] - 1),
                                    np.random.default_rng(run_seed(seed, "sc")))
            if sc is not None:
                env._state["schedule"].append({"t": can["shortcut_t"],
                                               "edges": sc["edges"], "kind": "shortcut"})
            agent_cfg = {**cfg["agent"], "dyna_n": n}
            agent = make_agent("dyna_q" if n > 0 else "q_learning",
                               env.n_states, env.W, env.H,
                               env.state_idx((can["H"] - 1, can["W"] - 1)),
                               agent_cfg, make_rng(run_seed(seed, "agent")))
            s = env.state_idx(env.reset())
            ep = 0
            steps_cur = 0
            for t in range(can["steps"]):
                a = agent.act(s, t)
                s2c, r, term, trunc, _ = env.step(a)
                s2 = env.state_idx(s2c)
                agent.observe(s, a, r, s2, term, t)
                steps_cur += 1
                if term or trunc:
                    append_csv(path, header, [n, sd, ep, t, steps_cur, int(term)])
                    ep += 1
                    steps_cur = 0
                    s = env.state_idx(env.reset())
                else:
                    s = s2
    with open(path + ".done", "w") as fh:
        fh.write("done")
    print("[canonical] complete")


def phase_grid(cfg: dict, out_dir: str, calibrated: dict,
               scenarios=None, methods=None) -> None:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["calibrated"] = {"h": calibrated["chosen_h"],
                         "kappa": calibrated["kappa"]["chosen_kappa"]}
    rcfg = RunnerConfig.from_yaml_dict(cfg)
    run_grid(rcfg, out_dir, scenarios=scenarios, methods=methods)


def clear_outputs(out_dir: str, phase: str) -> None:
    """Remove generated outputs for a true re-run (`--fresh`).

    Committed reference artifacts live in artifacts/pilot; this helper is only
    reached when the caller explicitly asks for a fresh run (usually into a
    separate --out directory)."""
    stale_files: list[str] = []
    stale_dirs: list[str] = []
    if phase in ("all", "calibration"):
        stale_files += ["calibration_summary.json", "calibration_cusum.json",
                        "calibration_kappa.json"]
    if phase in ("all", "canonical"):
        stale_files += ["canonical.csv", "canonical.csv.done"]
    if phase in ("all", "grid"):
        stale_files += ["metrics.csv", "episodes.csv"]
        stale_dirs += ["logs", "checkpoints"]
    for name in stale_files:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            os.remove(p)
            print(f"[fresh] removed {p}")
    for name in stale_dirs:
        p = os.path.join(out_dir, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"[fresh] removed {p}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--phase", default="all",
                    choices=["all", "manifest", "calibration", "canonical", "grid"])
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest-out", default=os.path.join("data", "manifests",
                                                           "mazes_manifest.json"),
                    help="where to write the maze manifest")
    ap.add_argument("--fresh", action="store_true",
                    help="delete generated outputs for this phase before running")
    args = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    out_dir = args.out or os.path.join("artifacts", cfg.get("run_name", "pilot"))
    if args.fresh:
        clear_outputs(out_dir, args.phase)
    ensure_dir(out_dir)

    if args.phase in ("all", "manifest"):
        phase_manifest(cfg, args.manifest_out)
    if args.phase in ("all", "calibration"):
        calibrated = phase_calibration(cfg, out_dir)
    else:
        p = os.path.join(out_dir, "calibration_summary.json")
        calibrated = json.load(open(p)) if os.path.exists(p) else {"chosen_h": 3.0, "kappa": {"chosen_kappa": 0.001}}
    if args.phase in ("all", "canonical"):
        phase_canonical(cfg, out_dir)
    if args.phase in ("all", "grid"):
        phase_grid(cfg, out_dir, calibrated, args.scenarios, args.methods)


if __name__ == "__main__":
    main()
