#!/usr/bin/env python3
"""Generate all figures and summary tables from the experiment outputs.

Figures (artifacts/figures/):
  fig_canonical.png        - Sutton-style blocking->shortcut, Dyna backup-cap sweep
  fig_learning_curves.png  - rolling success per method (blocking, rho=0.3)
  fig_adaptation_bar.png   - adaptation delay by method with bootstrap CIs
  fig_pareto.png           - compute (planning ms/step) vs post-change performance
  fig_detector.png         - CUSUM alarm distribution: FAR vs detection (rho sweep)
  fig_belief_heatmap.png   - edge-belief map before/after the change (trained weights)
Tables (artifacts/tables/): per scenario/rho method summary CSV + markdown.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cavr.utils.logging import markdown_table  # noqa: E402
from cavr.utils.seeding import run_seed  # noqa: E402


def _rolling_curve(episodes_csv: pd.DataFrame, t_change: float, window: int = 8,
                   steps: int = 2500) -> tuple[np.ndarray, np.ndarray]:
    """Time axis + mean rolling success across runs for one method."""
    curves = []
    for (_mz, _es), grp in episodes_csv.groupby(["maze_seed", "episode_seed"]):
        grp = grp.sort_values("end_step")
        succ = grp["success"].to_numpy(dtype=float)
        ends = grp["end_step"].to_numpy(dtype=float)
        roll = np.array([succ[max(0, i - window + 1):i + 1].mean()
                         for i in range(len(succ))])
        curves.append((ends, roll))
    if not curves:
        return np.array([0.0]), np.array([0.0])
    grid = np.linspace(0, steps, 200)
    mean = np.zeros_like(grid)
    for grid_i, x in enumerate(grid):
        vals = []
        for ends, roll in curves:
            k = np.searchsorted(ends, x, side="right")
            if 0 < k <= len(roll):
                vals.append(roll[k - 1])
            elif k == 0:
                vals.append(0.0)
        mean[grid_i] = np.mean(vals) if vals else 0.0
    return grid, mean


def fig_canonical(art_dir: str, fig_dir: str) -> None:
    path = os.path.join(art_dir, "canonical.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for n, grp in df.groupby("dyna_n"):
        eps = sorted(grp["ep"].unique())
        cum = [grp[grp["ep"] == e]["steps"].max() for e in eps]
        base = grp[grp["ep"] == 0]["end_step"].max() if 0 in set(grp["ep"]) else 0
        steps_at_ep = [grp[grp["ep"] == e]["end_step"].max() for e in eps]
        ax.plot(steps_at_ep, eps, marker="o", ms=3, label=f"Dyna n={n}" if n > 0 else "Q-learning (n=0)")
    ax.axvline(1000, color="crimson", lw=1, ls="--")
    ax.axvline(2000, color="seagreen", lw=1, ls="--")
    ax.text(1000, 2, " path blocked", color="crimson", fontsize=8, va="bottom")
    ax.text(2000, 2, " shortcut opens", color="seagreen", fontsize=8, va="bottom")
    ax.set_xlabel("real environment steps")
    ax.set_ylabel("completed episodes")
    ax.set_title("Canonical 6x9 blocking/shortcut task (Sutton-style), rho=0")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(fig_dir, "fig_canonical.png"), dpi=150)
    plt.close(fig)


def fig_learning_curves(art_dir: str, fig_dir: str, scenario="blocking", rho=0.3,
                        fname: str = "fig_learning_curves.png") -> None:
    ep_path = os.path.join(art_dir, "episodes.csv")
    met_path = os.path.join(art_dir, "metrics.csv")
    if not (os.path.exists(ep_path) and os.path.exists(met_path)):
        return
    eps = pd.read_csv(ep_path)
    met = pd.read_csv(met_path)
    sel_met = met[(met["scenario"] == scenario) & (np.isclose(met["rho"], rho))]
    if sel_met.empty:
        return
    steps = int(sel_met["steps"].iloc[0])
    t_change = int(sel_met["t_change"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    order = ["q_learning", "dyna_q", "dyna_q_plus", "prioritized_sweeping",
             "dirichlet_dyna", "local_forgetting", "dstar_lite",
             "cavr_fixedB", "cavr_nodetector", "cavr"]
    for m in order:
        sel = eps[(eps["method"] == m) & (eps["scenario"] == scenario)
                  & (np.isclose(eps["rho"], rho))]
        if sel.empty:
            continue
        grid, mean = _rolling_curve(sel, t_change, steps=steps)
        ax.plot(grid, mean, lw=1.4, label=m)
    ax.axvline(t_change, color="k", lw=1, ls=":")
    ax.text(t_change + 10, 0.02, "change", fontsize=8)
    ax.set_xlabel("real step")
    ax.set_ylabel(f"rolling success (window 8 episodes)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{scenario}, rho={rho}: rolling success")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.savefig(os.path.join(fig_dir, fname), dpi=150)
    plt.close(fig)


def fig_adaptation_bar(art_dir: str, fig_dir: str) -> None:
    met_path = os.path.join(art_dir, "metrics.csv")
    if not os.path.exists(met_path):
        return
    met = pd.read_csv(met_path)
    met = met[(met["scenario"] != "static")]
    if met.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for ax, scenario in zip(axes, ["blocking", "shortcut"]):
        sub = met[met["scenario"] == scenario]
        sub = sub[np.isclose(sub["rho"], 0.3)] if len(sub[sub["rho"] == 0.3]) else sub
        methods = sorted(sub["method"].unique(), key=lambda m: (m != "cavr", m))
        meds, los, his = [], [], []
        for m in methods:
            v = sub[sub["method"] == m]["adaptation_delay"].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                meds.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            rng = np.random.default_rng(0)
            boots = np.median(rng.choice(v, size=(5000, len(v))), axis=1)
            meds.append(np.median(v)); los.append(np.percentile(boots, 2.5)); his.append(np.percentile(boots, 97.5))
        y = np.arange(len(methods))
        colors = ["#c8102e" if m == "cavr" else "#7f8c9a" for m in methods]
        ax.barh(y, meds, xerr=[np.maximum(0, np.array(meds) - np.array(los)),
                               np.maximum(0, np.array(his) - np.array(meds))],
                color=colors, alpha=0.85, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("median adaptation delay (steps, 95% boot CI)")
        ax.set_title(scenario)
    fig.savefig(os.path.join(fig_dir, "fig_adaptation_bar.png"), dpi=150)
    plt.close(fig)


def fig_pareto(art_dir: str, fig_dir: str) -> None:
    met_path = os.path.join(art_dir, "metrics.csv")
    if not os.path.exists(met_path):
        return
    met = pd.read_csv(met_path)
    sub = met[(met["scenario"] == "blocking") & (np.isclose(met["rho"], 0.3))]
    if sub.empty:
        sub = met[met["scenario"] != "static"]
    if sub.empty:
        return
    agg = sub.groupby("method").agg(
        ms=("planning_ms", "mean"),
        succ=("success_post", "mean"),
        delay=("adaptation_delay", "median"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.scatter(agg["ms"], agg["delay"], s=60)
    for _, row in agg.iterrows():
        ax.annotate(row["method"], (row["ms"], row["delay"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("mean planning wall-clock per run (ms, log)")
    ax.set_ylabel("median post-change adaptation delay (steps)")
    ax.set_title("Compute vs adaptation (blocking, rho=0.3)")
    fig.savefig(os.path.join(fig_dir, "fig_pareto.png"), dpi=150)
    plt.close(fig)


def fig_detector(art_dir: str, fig_dir: str) -> None:
    met_path = os.path.join(art_dir, "metrics.csv")
    if not os.path.exists(met_path):
        return
    met = pd.read_csv(met_path)
    det = met[met["method"].isin(["cavr", "local_forgetting"])]
    if det.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    for m, grp in det.groupby("method"):
        g = grp.groupby("rho").agg(far=("false_per_1000", "mean")).reset_index()
        axes[0].plot(g["rho"], g["far"], marker="o", label=m)
        g2 = grp[grp["scenario"] != "static"].groupby("rho").agg(
            dd=("detection_delay", "mean")).reset_index()
        axes[1].plot(g2["rho"], g2["dd"], marker="o", label=m)
    axes[0].axhline(5, color="crimson", ls="--", lw=1)
    axes[0].text(0.02, 5.3, "G1 budget: 5/1000", color="crimson", fontsize=8)
    axes[0].set_xlabel("true slip rho"); axes[0].set_ylabel("false alarms / 1000 steps")
    axes[0].set_title("False alarms (static runs)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("true slip rho"); axes[1].set_ylabel("mean detection delay (steps)")
    axes[1].set_title("Detection delay (changed runs)")
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(fig_dir, "fig_detector.png"), dpi=150)
    plt.close(fig)


def fig_belief_heatmap(fig_dir: str, checkpoints_dir: str) -> None:
    ckpts = sorted(glob.glob(os.path.join(checkpoints_dir,
                                          "cavr_trained_m*.npz")))
    if not ckpts:
        return
    from cavr.utils.checkpoint import load_checkpoint
    from cavr.envs.maze_gen import generate_maze
    arrays, meta = load_checkpoint(ckpts[0])
    H, W = int(meta["H"]), int(meta["W"])
    maze = generate_maze(H, W, int(meta["maze_seed"]), 0.15)
    from cavr.envs.maze_gen import maze_open_set
    true_open = maze_open_set(maze)
    from cavr.agents.cavr_dyna import CAVRDynaAgent
    agent = CAVRDynaAgent(H * W, W, H, meta["goal"][0] * W + meta["goal"][1],
                          meta["agent_cfg"], np.random.default_rng(0),
                          rho_est=float(meta["rho_est"]))
    agent.load_checkpoint_arrays(arrays, meta)
    B = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            if c + 1 < W:
                e = ((r, c), (r, c + 1))
                val = agent.edges.b(e)
                B[r, c] = val
                B[r, c + 1] = val
            if r + 1 < H:
                e = ((r, c), (r + 1, c))
                val = agent.edges.b(e)
                B[r, c] = max(B[r, c], val)
                B[r + 1, c] = val
    T = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            for (rr, cc) in [(r, c + 1), (r + 1, c)]:
                if rr < H and cc < W:
                    val = 1.0 if ((r, c), (rr, cc)) in true_open else 0.05
                    T[r, c] = max(T[r, c], val)
                    T[rr, cc] = max(T[rr, cc], val)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(T, cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("True open edges (evaluator view)")
    im1 = axes[1].imshow(B, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Learned edge-open beliefs b_e (trained weights)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(f"maze seed {meta['maze_seed']}: learned topology beliefs")
    fig.savefig(os.path.join(fig_dir, "fig_belief_heatmap.png"), dpi=150)
    plt.close(fig)


def tables(art_dir: str, tab_dir: str) -> None:
    met_path = os.path.join(art_dir, "metrics.csv")
    if not os.path.exists(met_path):
        return
    met = pd.read_csv(met_path)
    rows_md = []
    for (scenario, rho), sub in met.groupby(["scenario", "rho"]):
        out_rows = []
        for m, g in sub.groupby("method"):
            out_rows.append({
                "method": m,
                "success": g["success_rate"].mean(),
                "succ_post": g["success_post"].mean(),
                "delay_med": g["adaptation_delay"].median(),
                "delay_censored%": 100 * g["adaptation_censored"].mean(),
                "far/1000": g["false_per_1000"].mean(),
                "detect_delay": g["detection_delay"].mean(),
                "AULC": g["auc_success"].mean(),
                "regret": g["post_change_regret"].mean(),
                "backups/step": g["backups_per_step"].mean(),
                "plan_ms": g["planning_ms"].mean(),
            })
        df = pd.DataFrame(out_rows).sort_values("method")
        csv_path = os.path.join(tab_dir, f"summary_{scenario}_rho{rho}.csv")
        df.to_csv(csv_path, index=False)
        rows_md.append(f"### {scenario}, rho={rho}\n")
        rows_md.append(markdown_table(list(df.columns),
                                      df.itertuples(index=False, name=None)))
        rows_md.append("")
    with open(os.path.join(tab_dir, "all_tables.md"), "w") as fh:
        fh.write("\n".join(rows_md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts/pilot",
                    help="directory with metrics.csv/episodes.csv/canonical.csv")
    ap.add_argument("--fig-dir", default="artifacts/figures")
    ap.add_argument("--tab-dir", default="artifacts/tables")
    ap.add_argument("--checkpoints-dir", default="artifacts/checkpoints")
    args = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    fig_dir = args.fig_dir
    tab_dir = args.tab_dir
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    fig_canonical(args.art, fig_dir)
    fig_learning_curves(args.art, fig_dir)
    fig_learning_curves(args.art, fig_dir, scenario="shortcut", rho=0.0,
                        fname="fig_learning_curves_shortcut_rho0.png")
    fig_adaptation_bar(args.art, fig_dir)
    fig_pareto(args.art, fig_dir)
    fig_detector(args.art, fig_dir)
    fig_belief_heatmap(fig_dir, args.checkpoints_dir)
    tables(args.art, tab_dir)
    print("figures ->", fig_dir)
    print("tables  ->", tab_dir)


if __name__ == "__main__":
    main()
