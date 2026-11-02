#!/usr/bin/env python3
"""One-command reproduction of the CAVR-Dyna pipeline.

Runs, in order:
  1. unit tests (unless --skip-tests)
  2. maze manifest + calibration + canonical task + evaluation grid
  3. CAVR training + held-out verification (unless --skip-train/-verify)
  4. figures + summary tables
  5. reports/results.md (pilot config by default)

All outputs go under --out, so the committed reference artifacts are never
modified.  The pilot's committed metrics/figures/tables stay in artifacts/ and
can be previewed without running anything (notebook: "view" mode).

Usage:
  python scripts/reproduce.py --config configs/pilot.yaml --out artifacts/runs/pilot
  python scripts/reproduce.py --config configs/smoke.yaml --out artifacts/runs/smoke --skip-report
  python scripts/reproduce.py --config configs/final.yaml --out artifacts/runs/final --skip-report
  python scripts/reproduce.py --config configs/pilot.yaml --out artifacts/runs/pilot --force
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"  [done in {time.perf_counter() - t0:.1f}s]", flush=True)


def clean_outputs(out_dir: str) -> None:
    """Remove only the generated subtree, never the committed artifacts."""
    for name in ("pilot", "manifests", "checkpoints", "verification",
                 "figures", "tables", "logs"):
        p = os.path.join(out_dir, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"[fresh] removed {p}/")
    for name in ("training_report.json", "results.md"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            os.remove(p)
            print(f"[fresh] removed {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--out", default=None,
                    help="output root (default: artifacts/runs/<run_name>)")
    ap.add_argument("--train-steps", type=int, default=None,
                    help="override config.train.steps")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="skip training (verification still runs if checkpoints exist)")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="generate results.md even when run_name != pilot")
    ap.add_argument("--force", action="store_true",
                    help="delete generated outputs under --out before running")
    args = ap.parse_args()

    os.chdir(ROOT)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    run_name = str(cfg.get("run_name", "run"))
    out_dir = os.path.abspath(args.out or os.path.join("artifacts", "runs", run_name))

    protected = {
        os.path.abspath(ROOT),
        os.path.abspath(os.path.join(ROOT, "artifacts")),
        os.path.abspath(os.path.join(ROOT, "artifacts", "pilot")),
        os.path.abspath(os.path.join(ROOT, "artifacts", "checkpoints")),
        os.path.abspath(os.path.join(ROOT, "artifacts", "figures")),
        os.path.abspath(os.path.join(ROOT, "artifacts", "tables")),
        os.path.abspath(os.path.join(ROOT, "artifacts", "verification")),
        os.path.abspath(os.path.join(ROOT, "reports")),
    }
    if args.force and out_dir in protected:
        raise SystemExit(f"Refusing --force on committed path: {out_dir}. "
                         "Use a separate --out directory (e.g. artifacts/runs/pilot).")

    if args.force:
        clean_outputs(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pilot_dir = os.path.join(out_dir, "pilot")
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    ver_dir = os.path.join(out_dir, "verification")
    report_path = os.path.join(out_dir, "results.md")
    manifest_path = os.path.join(out_dir, "manifests", "mazes_manifest.json")
    py = sys.executable

    print(f"[reproduce] config: {args.config} (run_name={run_name})")
    print(f"[reproduce] output: {out_dir}")

    # ---------------------------------------------------------------- tests
    if not args.skip_tests:
        run([py, "-m", "pytest", "-q"])

    # --------------------------------------------------------- experiment run
    cmd = [py, "scripts/run_pilot.py",
           "--config", args.config,
           "--out", pilot_dir,
           "--manifest-out", manifest_path,
           "--phase", "all"]
    if args.force:
        cmd.append("--fresh")
    run(cmd)

    # -------------------------------------------------------------- training
    if not args.skip_train:
        cmd = [py, "scripts/train_model.py",
               "--config", args.config,
               "--out", out_dir,
               "--calibration", os.path.join(pilot_dir, "calibration_summary.json")]
        if args.train_steps is not None:
            cmd += ["--steps", str(args.train_steps)]
        run(cmd)

    # ------------------------------------------------------------ verification
    if not args.skip_verify:
        checkpoints = glob.glob(os.path.join(ckpt_dir, "cavr_trained_m*.npz"))
        if checkpoints:
            run([py, "scripts/verify_model.py",
                 "--config", args.config,
                 "--checkpoints-dir", ckpt_dir,
                 "--out", ver_dir])
        else:
            print("[reproduce] no training checkpoints found; skipping verification")

    # -------------------------------------------------------- figures + tables
    run([py, "scripts/plot_results.py",
         "--art", pilot_dir,
         "--fig-dir", fig_dir,
         "--tab-dir", tab_dir,
         "--checkpoints-dir", ckpt_dir])

    # ---------------------------------------------------------------- report
    make_report = (not args.skip_report) and (args.report or run_name == "pilot")
    if make_report:
        if not os.path.exists(os.path.join(ver_dir, "verification_report.json")):
            print("[reproduce] verification report missing; skipping results.md")
        else:
            run([py, "scripts/make_results_report.py",
                 "--config", args.config,
                 "--art", pilot_dir,
                 "--verification", ver_dir,
                 "--out", report_path])
    elif not args.skip_report:
        print(f"[reproduce] results.md narrative is pilot-specific; "
              f"skipping for run_name={run_name} (pass --report to force)")

    print("\n[reproduce] complete. Outputs:")
    for p in (pilot_dir, fig_dir, tab_dir, ckpt_dir, ver_dir, report_path):
        if os.path.exists(p):
            print("  ", p)


if __name__ == "__main__":
    main()
