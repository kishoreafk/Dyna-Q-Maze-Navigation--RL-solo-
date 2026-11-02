"""Threshold calibration on no-change validation mazes (spec 6.2 / G1).

CUSUM threshold h: grid search on stationary slip=0.3 validation runs with
the detector observing a Dirichlet stochastic Dyna agent (no reset), choosing
the smallest h meeting the false-alarm budget (<=5 alarms/1000 transitions
with safety margin).

Dyna-Q+ kappa: small sweep on validation blocking mazes; the value with the
best post-change recovery is kept.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..agents import make_agent
from ..envs.grid_env import GridEnv
from ..utils.seeding import run_seed, make_rng
from .runner import load_maze


def _false_alarms_for_h(cfg, mazes, h: float, rho: float = 0.3,
                        steps: int = 1200) -> float:
    total = 0
    false = 0
    for maze in mazes:
        seed = run_seed(999, maze["seed"], "calib", h)
        env = GridEnv(maze, cfg["goal"], "static", rho, cfg["max_steps"], 0,
                      env_seed=run_seed(seed, "env"))
        agent = make_agent("local_forgetting", env.n_states, env.W, env.H,
                           env.state_idx(cfg["goal"]),
                           {**cfg["agent_cfg"], "h": h},
                           make_rng(run_seed(seed, "agent")), rho_est=rho)
        # observe-only calibration: disable the local reset
        agent.do_reset = False
        s = env.state_idx(env.reset())
        for t in range(steps):
            a = agent.act(s, t)
            s2c, r, term, trunc, _ = env.step(a)
            s2 = env.state_idx(s2c)
            stats = agent.observe(s, a, r, s2, term, t)
            if stats.get("alarm"):
                false += 1
            if term or trunc:
                s = env.state_idx(env.reset())
            else:
                s = s2
        total += steps
    return 1000.0 * false / max(1, total)


def calibrate_cusum(cfg: dict, mazes: list[dict], h_grid=None,
                    target_per_1000: float = 5.0, out_path: str = None) -> dict:
    h_grid = h_grid or [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    target = target_per_1000 * 0.6  # safety margin
    rows = []
    for h in h_grid:
        far = _false_alarms_for_h(cfg, mazes, h)
        rows.append({"h": h, "false_per_1000": far})
    ok = [r for r in rows if r["false_per_1000"] <= target]
    chosen = ok[0]["h"] if ok else rows[-1]["h"]
    record = {"h_grid": rows, "chosen_h": chosen,
              "target_per_1000": target_per_1000,
              "met_budget": bool(ok)}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)
    return record


def calibrate_kappa(cfg: dict, mazes: list[dict], kappa_grid=None,
                    steps: int = 1500, out_path: str = None) -> dict:
    kappa_grid = kappa_grid or [0.0003, 0.001, 0.003]
    t_change = steps // 2
    rows = []
    for kappa in kappa_grid:
        delays = []
        for maze in mazes:
            seed = run_seed(998, maze["seed"], "kappa", kappa)
            env = GridEnv(maze, cfg["goal"], "blocking", 0.0, cfg["max_steps"],
                          t_change, env_seed=run_seed(seed, "env"))
            agent = make_agent("dyna_q_plus", env.n_states, env.W, env.H,
                               env.state_idx(cfg["goal"]),
                               {**cfg["agent_cfg"], "kappa": kappa},
                               make_rng(run_seed(seed, "agent")))
            s = env.state_idx(env.reset())
            ep_end = []
            succ = []
            n_ep = 0
            cur_steps = 0
            for t in range(steps):
                a = agent.act(s, t)
                s2c, r, term, trunc, _ = env.step(a)
                s2 = env.state_idx(s2c)
                agent.observe(s, a, r, s2, term, t)
                cur_steps += 1
                if term or trunc:
                    ep_end.append(t)
                    succ.append(int(term))
                    n_ep += 1
                    s = env.state_idx(env.reset())
                    cur_steps = 0
                else:
                    s = s2
            # first successful episode ending after t_change
            d = steps - t_change
            for e_end, sc in zip(ep_end, succ):
                if e_end >= t_change and sc:
                    d = e_end - t_change
                    break
            delays.append(d)
        rows.append({"kappa": kappa, "mean_delay": float(np.mean(delays))})
    best = min(rows, key=lambda r: r["mean_delay"])
    record = {"kappa_grid": rows, "chosen_kappa": best["kappa"]}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)
    return record


def run_calibration(cfg: dict, maze_seeds: list[int], H: int, W: int,
                    braid: float, out_dir: str) -> dict:
    mazes = [load_maze(H, W, s, braid) for s in maze_seeds]
    cusum = calibrate_cusum(cfg, mazes, out_path=os.path.join(out_dir, "calibration_cusum.json"))
    kappa = calibrate_kappa(cfg, mazes, out_path=os.path.join(out_dir, "calibration_kappa.json"))
    return {"cusum": cusum, "kappa": kappa}
