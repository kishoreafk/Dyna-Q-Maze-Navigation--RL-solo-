"""Experiment runner: single runs and the full evaluation grid.

Run keys are (maze_seed, scenario, rho, episode_seed, method); results are
appended to metrics CSV and each completed run writes a done-marker so the
grid can be resumed after interruption (spec section 8: checkpoints every
completed evaluation group).
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from ..agents import make_agent, ALL_METHODS
from ..envs.grid_env import GridEnv
from ..envs.maze_gen import generate_maze, maze_open_set, maze_sha256
from ..utils.seeding import run_seed, make_rng
from ..utils.logging import JsonlWriter, append_csv, ensure_dir
from .metrics import (adaptation_delay, alarm_stats, auc_success,
                      first_success_delay, post_change_regret, success_rate)
from .oracles import bfs_shortest_len


@dataclass
class RunnerConfig:
    H: int = 9
    W: int = 9
    braid: float = 0.15
    max_steps: int = 200
    goal: tuple = (8, 8)
    start: tuple = (0, 0)
    steps: int = 2500
    t_change_frac: float = 0.5
    second_change_frac: float = 0.75
    seed_base: int = 20261102
    methods: tuple = tuple(ALL_METHODS)
    scenario_rhos: dict = None
    n_mazes: int = 4
    maze_seed_start: int = 1000
    episode_seeds: int = 2
    agent_cfg: dict = None
    save_checkpoints: bool = True
    step_log_subset: bool = True

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "RunnerConfig":
        env = d["env"]
        grid = d["grid"]
        agent_cfg = dict(d.get("agent", {}))
        if d.get("calibrated", {}):
            agent_cfg.update(d["calibrated"])
        return cls(
            H=int(env["H"]), W=int(env["W"]), braid=float(env.get("braid", 0.15)),
            max_steps=int(env.get("max_steps", 200)),
            goal=tuple(env.get("goal", (8, 8))), start=tuple(env.get("start", (0, 0))),
            steps=int(grid.get("steps", 2500)),
            t_change_frac=float(grid.get("t_change_frac", 0.5)),
            second_change_frac=float(grid.get("second_change_frac", 0.75)),
            seed_base=int(d.get("seed_base", 20261102)),
            methods=tuple(grid.get("methods", ALL_METHODS)),
            scenario_rhos={k: list(v["rhos"]) for k, v in grid["scenarios"].items()},
            n_mazes=int(d["splits"].get("n_val", 4)),
            maze_seed_start=int(d["splits"].get("maze_seed_start", 1000)),
            episode_seeds=int(grid.get("episode_seeds", 2)),
            agent_cfg=agent_cfg,
            save_checkpoints=bool(grid.get("save_checkpoints", True)),
            step_log_subset=bool(d.get("logging", {}).get("step_log_subset", True)),
        )


def load_maze(H, W, seed, braid) -> dict:
    return generate_maze(H, W, seed, braid)


def run_single(maze: dict, maze_id: int, scenario: str, rho: float,
               episode_seed: int, method: str, cfg: RunnerConfig,
               out_dir: str, log_steps: bool = False) -> dict:
    t_run0 = time.perf_counter()
    seed = run_seed(cfg.seed_base, maze["seed"], scenario, rho, episode_seed, method)
    env_seed = run_seed(seed, "env")
    agent_rng = make_rng(run_seed(seed, "agent"))
    starts_rng = make_rng(run_seed(seed, "starts"))

    t_change = int(cfg.steps * cfg.t_change_frac)
    env = GridEnv(maze, cfg.goal, scenario, rho, cfg.max_steps, t_change,
                  second_change_frac=cfg.second_change_frac, env_seed=env_seed,
                  start=cfg.start)
    goal_idx = env.state_idx(cfg.goal)
    agent = make_agent(method, env.n_states, env.W, env.H, goal_idx,
                       cfg.agent_cfg, agent_rng, rho_est=rho)

    ensure_dir(out_dir)
    aw = ew = None
    if log_steps:
        aw = JsonlWriter(os.path.join(out_dir, "logs", f"{method}_{maze_id}_{scenario}_rho{rho}_s{episode_seed}_agent.jsonl.gz"), gz=True)
        ew = JsonlWriter(os.path.join(out_dir, "logs", f"{method}_{maze_id}_{scenario}_rho{rho}_s{episode_seed}_eval.jsonl.gz"), gz=True)

    start_cells = [tuple(cfg.start)]
    candidates = [(r, c) for r in range(env.H) for c in range(env.W)
                  if (r, c) != tuple(cfg.goal)]
    idx = starts_rng.permutation(len(candidates))[:30]
    start_cells += [candidates[int(i)] for i in idx]

    ep_records = []
    alarm_ts: list[int] = []
    backups_total = 0
    planning_ms = 0.0
    queue_ops = 0

    s_cell = env.reset(start=start_cells[0])
    s = env.state_idx(s_cell)
    cur = {"start_step": 0, "ret": 0.0, "success": 0, "steps": 0}
    n_ep = 0

    for t in range(cfg.steps):
        a = agent.act(s, t)
        s2_cell, r, term, trunc, _ = env.step(a)
        s2 = env.state_idx(s2_cell)
        stats = agent.observe(s, a, r, s2, term, t)
        backups_total += int(stats.get("backups_executed", 0))
        planning_ms += float(stats.get("wall_time_ms", 0.0))
        queue_ops += int(stats.get("queue_operations", 0))
        if stats.get("alarm"):
            alarm_ts.append(t)

        cur["ret"] += r
        cur["steps"] += 1
        if ew is not None:
            ev = env.evaluator_state()
            ew.write({"step": t, "n_change_events": len(ev["events"])})
        if aw is not None:
            aw.write({
                "step": t, "maze_id": int(maze["seed"]), "seed": int(episode_seed),
                "action": int(a), "reward": round(float(r), 4),
                "success": int(term),
                "model_uncertainty": round(float(stats.get("model_uncertainty", 0.0)), 4),
                "predicted_change_probability": round(float(stats.get("predicted_change_probability", 0.0)), 4),
                "budget_assigned": int(stats.get("budget_assigned", 0)),
                "backups_executed": int(stats.get("backups_executed", 0)),
                "queue_operations": int(stats.get("queue_operations", 0)),
                "wall_time_ms": round(float(stats.get("wall_time_ms", 0.0)), 3),
                "model_version": int(stats.get("model_version", 0)),
                "proposed_probe": stats.get("proposed_probe", -1),
                "actual_action": int(stats.get("actual_action", a)),
            })
        if term or trunc:
            cur["success"] = int(term)
            cur["end_step"] = t
            ep_records.append(cur)
            n_ep += 1
            nxt = start_cells[(n_ep) % len(start_cells)]
            s_cell = env.reset(start=nxt)
            s = env.state_idx(s_cell)
            cur = {"start_step": t + 1, "ret": 0.0, "success": 0, "steps": 0}
        else:
            s = s2

    if aw is not None:
        aw.close()
    if ew is not None:
        ew.close()

    events = env.evaluator_state()["events"]
    change_ts = [e["t"] for e in events]

    total_steps = cfg.steps
    delay, cens = adaptation_delay(ep_records, t_change)
    fsd, fsd_cens = first_success_delay(ep_records, t_change)
    al_stats = alarm_stats(alarm_ts, change_ts, total_steps)

    summary = {
        "method": method,
        "maze_seed": int(maze["seed"]),
        "maze_id": int(maze_id),
        "scenario": scenario,
        "rho": float(rho),
        "episode_seed": int(episode_seed),
        "seed": int(seed),
        "steps": total_steps,
        "t_change": t_change,
        "n_episodes": len(ep_records),
        "success_rate": success_rate(ep_records),
        "success_pre": success_rate(ep_records, 0, t_change),
        "success_post": success_rate(ep_records, t_change),
        "adaptation_delay": delay,
        "adaptation_censored": int(cens),
        "first_success_delay": fsd,
        "auc_success": auc_success(ep_records, total_steps),
        "post_change_regret": post_change_regret(ep_records, t_change),
        "mean_return_per_ep": float(np.mean([e["ret"] for e in ep_records])) if ep_records else 0.0,
        "false_per_1000": al_stats["false_per_1000"],
        "n_alarms": al_stats["n_alarms"],
        "n_false_alarms": al_stats["n_false"],
        "detection_delay": al_stats["detection_delay"],
        "changes_detected": al_stats["n_changes_detected"],
        "changes_total": al_stats["n_changes_total"],
        "backups_total": backups_total,
        "backups_per_step": backups_total / total_steps,
        "probes_total": int(agent.totals.get("probes", 0)),
        "planning_ms": planning_ms,
        "queue_ops": queue_ops,
        "run_wall_s": time.perf_counter() - t_run0,
        "maze_sha256": maze_sha256(maze)[:16],
    }

    if cfg.save_checkpoints and method.startswith("cavr"):
        from ..utils.checkpoint import save_checkpoint
        ckpt_path = os.path.join(out_dir, "checkpoints",
                                 f"{method}_m{maze['seed']}_{scenario}_rho{rho}_s{episode_seed}.npz")
        arrays = agent.checkpoint_arrays()
        meta = {
            "method": method, "maze_seed": int(maze["seed"]), "scenario": scenario,
            "rho_true": float(rho), "rho_est": float(agent.rho_est),
            "H": env.H, "W": env.W, "goal": list(cfg.goal),
            "h": float(agent.detector.h) if agent.detector else float(cfg.agent_cfg.get("h", 3.0)),
            "kappa": float(cfg.agent_cfg.get("kappa", 0.001)),
            "t_final": int(cfg.steps), "epi_ema": float(agent._epi_ema),
            "agent_cfg": {k: v for k, v in cfg.agent_cfg.items()},
        }
        if "records" in arrays:
            meta["memory_records"] = arrays.pop("records")
        save_checkpoint(ckpt_path, arrays, meta)
        summary["checkpoint"] = ckpt_path

    # episode-level CSV (one row per episode)
    ep_header = ["method", "maze_seed", "scenario", "rho", "episode_seed",
                 "ep", "start_step", "end_step", "steps", "success", "ret"]
    for i, e in enumerate(ep_records):
        append_csv(os.path.join(out_dir, "episodes.csv"), ep_header,
                   [summary["method"], summary["maze_seed"], summary["scenario"],
                    summary["rho"], summary["episode_seed"], i,
                    e["start_step"], e["end_step"], e["steps"], e["success"],
                    round(e["ret"], 4)])
    return summary


def run_key(maze_seed, scenario, rho, episode_seed, method) -> str:
    return f"{method}_m{maze_seed}_{scenario}_rho{rho}_s{episode_seed}"


def run_grid(cfg: RunnerConfig, out_dir: str, scenarios=None, methods=None,
             verbose: bool = True) -> list[dict]:
    ensure_dir(out_dir)
    done_dir = os.path.join(out_dir, "logs", "done")
    ensure_dir(done_dir)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    # Resume safety: a fresh clone ships metrics.csv but not the gitignored
    # done markers.  Treat keys already present in metrics.csv as complete so
    # re-running the documented command never appends duplicate rows.
    existing_keys: set[tuple] = set()
    if os.path.exists(metrics_path):
        with open(metrics_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    existing_keys.add((row["method"], int(row["maze_seed"]),
                                       row["scenario"], round(float(row["rho"]), 6),
                                       int(row["episode_seed"])))
                except (KeyError, ValueError):
                    continue
    header = ["method", "maze_seed", "maze_id", "scenario", "rho", "episode_seed",
              "steps", "t_change", "n_episodes", "success_rate", "success_pre",
              "success_post", "adaptation_delay", "adaptation_censored",
              "first_success_delay", "auc_success", "post_change_regret",
              "mean_return_per_ep", "false_per_1000", "n_alarms", "n_false_alarms",
              "detection_delay", "changes_detected", "changes_total",
              "backups_total", "backups_per_step", "probes_total", "planning_ms",
              "queue_ops", "run_wall_s"]
    scenarios = scenarios if scenarios is not None else list(cfg.scenario_rhos.keys())
    methods = methods if methods is not None else list(cfg.methods)
    summaries = []
    n_done = 0
    n_skipped = 0
    for scenario in scenarios:
        rhos = cfg.scenario_rhos[scenario]
        for rho in rhos:
            for method in methods:
                for mi in range(cfg.n_mazes):
                    maze_seed = cfg.maze_seed_start + mi
                    maze = load_maze(cfg.H, cfg.W, maze_seed, cfg.braid)
                    for es in range(cfg.episode_seeds):
                        key = run_key(maze_seed, scenario, rho, es, method)
                        done_path = os.path.join(done_dir, key + ".done")
                        key_parts = (method, maze_seed, scenario,
                                     round(float(rho), 6), es)
                        if os.path.exists(done_path) or key_parts in existing_keys:
                            if not os.path.exists(done_path):
                                with open(done_path, "w") as fh:
                                    fh.write(json.dumps({"key": key,
                                                         "skipped_existing": True}))
                            n_done += 1
                            n_skipped += 1
                            continue
                        log_steps = cfg.step_log_subset and es == 0
                        s = run_single(maze, mi, scenario, rho, es, method, cfg,
                                       out_dir, log_steps=log_steps)
                        append_csv(metrics_path, header,
                                   [s[h] for h in header])
                        with open(done_path, "w") as fh:
                            fh.write(json.dumps({"key": key, "run_wall_s": s["run_wall_s"]}))
                        summaries.append(s)
                        existing_keys.add(key_parts)
                        n_done += 1
                        if verbose:
                            print(f"[{n_done:4d}] {key} succ={s['success_rate']:.2f} "
                                  f"delay={s['adaptation_delay']:.0f}{'(c)' if s['adaptation_censored'] else ''} "
                                  f"far1000={s['false_per_1000']:.2f} "
                                  f"bups={s['backups_per_step']:.1f} {s['run_wall_s']:.1f}s",
                                  flush=True)
    if verbose:
        print(f"grid complete: {n_done} runs accounted for "
              f"({n_skipped} already present; {len(summaries)} executed)", flush=True)
    return summaries
