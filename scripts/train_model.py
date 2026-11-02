#!/usr/bin/env python3
"""Train CAVR-Dyna on the training mazes and save the model weights.

Training curriculum per maze (single continuous stream, tabular "training"):
  phase 1  static            (steps 0        .. T1)
  phase 2  blocking change   (at T1)
  phase 3  shortcut opening  (at T2)
Each phase is a held-in experience for the weights; verification (scripts/
verify_model.py) later runs NEW unseen change schedules on the same mazes.

Weights saved: artifacts/checkpoints/cavr_trained_m{seed}.npz and a copy at
artifacts/checkpoints/cavr_trained.npz. Weights = Q table + Dirichlet
counts + edge-belief logits + detector state + episodic memory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cavr.agents import make_agent  # noqa: E402
from cavr.envs.grid_env import GridEnv  # noqa: E402
from cavr.envs.maze_gen import generate_maze, maze_open_set  # noqa: E402
from cavr.envs.change_scenarios import find_blocking_edge, find_shortcut_edge  # noqa: E402
from cavr.utils.checkpoint import save_checkpoint  # noqa: E402
from cavr.utils.logging import JsonlWriter  # noqa: E402
from cavr.utils.seeding import run_seed, make_rng  # noqa: E402
from cavr.evaluation.metrics import success_rate  # noqa: E402


def train_on_maze(maze, goal, cfg: dict, agent_cfg: dict, seed: int,
                  steps: int, t_block: int, t_shortcut: int,
                  log_path: str | None) -> tuple[dict, list, list]:
    """Single-maze training stream with two scheduled changes. The change
    edge selection uses the true map (environment construction only)."""
    rng_env = np.random.default_rng(run_seed(seed, "env"))
    blocking = find_blocking_edge(maze_open_set(maze), maze["H"], maze["W"],
                                  (0, 0), tuple(goal), rng_env)
    shortcut = find_shortcut_edge(maze_open_set(maze), maze["H"], maze["W"],
                                  (0, 0), tuple(goal), rng_env)
    schedule = []
    if blocking is not None:
        schedule.append({"t": t_block, "edges": blocking["edges"], "kind": "blocking"})
    if shortcut is not None:
        schedule.append({"t": t_shortcut, "edges": shortcut["edges"], "kind": "shortcut"})

    env = GridEnv(maze, goal, "static", cfg["rho_true"], cfg["max_steps"],
                  t_block, env_seed=run_seed(seed, "envstep"))
    env._state["schedule"] = schedule  # replace with our two-change curriculum

    agent = make_agent("cavr", env.n_states, env.W, env.H, env.state_idx(goal),
                       agent_cfg, make_rng(run_seed(seed, "agent")),
                       rho_est=cfg["rho_est"])
    writer = JsonlWriter(log_path, gz=True) if log_path else None

    starts_rng = make_rng(run_seed(seed, "starts"))
    candidates = [(r, c) for r in range(maze["H"]) for c in range(maze["W"])
                  if (r, c) != tuple(goal)]
    perm = starts_rng.permutation(len(candidates))
    start_cells = [(0, 0)] + [candidates[int(i)] for i in perm[:30]]

    ep_records = []
    cur = {"start_step": 0, "ret": 0.0, "success": 0, "steps": 0}
    n_ep = 0
    phase_marks = {"static": (0, t_block), "post_block": (t_block, t_shortcut),
                   "post_shortcut": (t_shortcut, steps)}
    s_cell = env.reset(start=start_cells[0])
    s = env.state_idx(s_cell)
    for t in range(steps):
        a = agent.act(s, t)
        s2c, r, term, trunc, _ = env.step(a)
        s2 = env.state_idx(s2c)
        agent.observe(s, a, r, s2, term, t)
        if writer is not None:
            writer.write({"step": t, "maze_id": int(maze["seed"]), "seed": int(seed),
                          "action": int(a), "reward": round(float(r), 4),
                          "success": int(term), "alarm": bool(agent.step_stats.get("alarm", False)),
                          "budget_assigned": int(agent.step_stats.get("budget_assigned", 0)),
                          "backups_executed": int(agent.step_stats.get("backups_executed", 0)),
                          "wall_time_ms": round(float(agent.step_stats.get("wall_time_ms", 0.0)), 3),
                          "model_version": int(agent.step_stats.get("model_version", 0)),
                          "model_uncertainty": round(float(agent.step_stats.get("model_uncertainty", 0.0)), 4),
                          "predicted_change_probability": round(float(agent.step_stats.get("predicted_change_probability", 0.0)), 4),
                          "proposed_probe": agent.step_stats.get("proposed_probe", -1),
                          "actual_action": int(agent.step_stats.get("actual_action", a)),
                          "queue_operations": int(agent.step_stats.get("queue_operations", 0))})
        cur["ret"] += r
        cur["steps"] += 1
        if term or trunc:
            cur["success"] = int(term)
            cur["end_step"] = t
            ep_records.append(cur)
            n_ep += 1
            s_cell = env.reset(start=start_cells[(n_ep) % len(start_cells)])
            s = env.state_idx(s_cell)
            cur = {"start_step": t + 1, "ret": 0.0, "success": 0, "steps": 0}
        else:
            s = s2
    if writer is not None:
        writer.close()

    phase_metrics = {}
    for name, (t0, t1) in phase_marks.items():
        eps = [e for e in ep_records if e["start_step"] >= t0 and e["end_step"] < t1]
        phase_metrics[name] = {
            "n_episodes": len(eps),
            "success_rate": success_rate(eps) if eps else float("nan"),
        }
    events = env.evaluator_state()["events"]
    return agent, phase_metrics, [{"t": e["t"], "kind": e["kind"]} for e in events]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps per maze (default: config.train.steps or 4500)")
    ap.add_argument("--out", default="artifacts",
                    help="output directory for checkpoints/logs/training_report.json")
    ap.add_argument("--calibration", default=None,
                    help="calibration_summary.json from run_pilot.py (default: auto-detect)")
    args = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    env_cfg, splits = cfg["env"], cfg["splits"]
    calib = cfg.get("calibrated", {})
    calib_path = args.calibration
    if not calib and not calib_path:
        run_name = cfg.get("run_name", "pilot")
        candidates = [os.path.join("artifacts", run_name, "calibration_summary.json"),
                      os.path.join("artifacts", "pilot", "calibration_summary.json")]
        calib_path = next((p for p in candidates if os.path.exists(p)), None)
    if not calib and calib_path:
        if os.path.exists(calib_path):
            with open(calib_path) as fh:
                c = json.load(fh)
            calib = {"h": c["chosen_h"], "kappa": c["kappa"]["chosen_kappa"]}
            print(f"[train] calibration: {calib_path} -> h={calib['h']}, kappa={calib['kappa']}")
        else:
            print(f"[train] warning: calibration file not found: {calib_path}; using defaults")
    agent_cfg = {**cfg["agent"], **calib}

    steps = args.steps or int(cfg.get("train", {}).get("steps", 4500))
    t_block = int(steps * 0.4)
    t_shortcut = int(steps * 0.75)
    train_seeds = [splits["maze_seed_start"] + i for i in range(splits["n_train"])]
    rho_true = float(cfg.get("train", {}).get("rho_true", 0.1))

    out_dir = args.out
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    report = {"train_mazes": [], "config": {"steps": steps, "t_block": t_block,
                                            "t_shortcut": t_shortcut,
                                            "rho_true": rho_true,
                                            "h": agent_cfg.get("h"),
                                            "kappa": agent_cfg.get("kappa")}}
    last_ckpt = None
    for maze_seed in train_seeds:
        t0 = time.perf_counter()
        maze = generate_maze(env_cfg["H"], env_cfg["W"], maze_seed,
                             env_cfg.get("braid", 0.15))
        goal = tuple(env_cfg["goal"])
        log_path = os.path.join(out_dir, "logs",
                                f"train_cavr_m{maze_seed}_agent.jsonl.gz")
        agent, phase_metrics, events = train_on_maze(
            maze, goal, {"rho_true": rho_true, "max_steps": env_cfg["max_steps"],
                         "rho_est": agent_cfg.get("rho_est", 0.1)},
            agent_cfg, seed=run_seed(cfg["seed_base"], "train", maze_seed),
            steps=steps, t_block=t_block, t_shortcut=t_shortcut, log_path=log_path)
        ckpt = os.path.join(out_dir, "checkpoints", f"cavr_trained_m{maze_seed}.npz")
        arrays = agent.checkpoint_arrays()
        meta = {"method": "cavr", "maze_seed": int(maze_seed), "scenario": "train_curriculum",
                "rho_true": rho_true, "rho_est": float(agent.rho_est),
                "H": env_cfg["H"], "W": env_cfg["W"], "goal": list(goal),
                "h": float(agent.detector.h), "kappa": float(agent_cfg.get("kappa", 0.001)),
                "t_final": int(steps), "epi_ema": float(agent._epi_ema),
                "agent_cfg": agent_cfg,
                "schedule": events,
                "phase_metrics": phase_metrics}
        if "records" in arrays:
            meta["memory_records"] = arrays.pop("records")
        save_checkpoint(ckpt, arrays, meta)
        last_ckpt = ckpt
        report["train_mazes"].append({"maze_seed": int(maze_seed),
                                      "checkpoint": ckpt,
                                      "phases": phase_metrics})
        print(f"[train] maze {maze_seed}: {phase_metrics} ({time.perf_counter()-t0:.1f}s)")
    if last_ckpt:
        shutil.copyfile(last_ckpt, os.path.join(out_dir, "checkpoints", "cavr_trained.npz"))
        report["primary_checkpoint"] = os.path.join(out_dir, "checkpoints", "cavr_trained.npz")
    with open(os.path.join(out_dir, "training_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[train] weights saved: {report.get('primary_checkpoint', 'none')}")


if __name__ == "__main__":
    main()
