# CAVR-Dyna — change-aware stochastic model-based RL for dynamic mazes

Research implementation of **CAVR-Dyna** (Counterfactual, Aleatoric-aware,
Value-of-computation-driven Repair for Dyna-style navigation), following the
2026 research blueprint and master implementation prompt provided as the
design authority. The method is a HYPOTHESIS evaluated empirically; all
results in `reports/results.md` come from actually executed runs.

**What it is:** a Dyna-style tabular agent that (i) learns a categorical
Dirichlet transition model and a factored edge-belief map under action slip,
(ii) detects topology changes with a calibrated per-(s,a) CUSUM on the
FROZEN pre-update prediction, (iii) repairs by local forgetting + goal-
counterfactual prioritized expected backups through an epoch-versioned
priority heap, (iv) arbitrates a real-action probe schedule against a
value-of-computation-style backup budget in {0,8,16,32,64}, and (v) keeps a
nonzero exploration floor so hidden shortcuts can be rediscovered.

**Baselines included:** Q-learning, Dyna-Q (last-outcome), Dyna-Q+ (age
bonus + synthetic untried actions), prioritized sweeping, stochastic
Dirichlet Dyna, local-forgetting Dyna, and D* Lite on the believed map.

**Headline pilot findings** (8 map/seed pairs, feasibility scale):
- CAVR cuts median post-change adaptation delay by ~47-52% vs the strongest
  Dyna baseline (`local_forgetting`) at rho=0.3, paired bootstrap CI
  excluding zero — but with a credible steady-state success cost
  (`reports/results.md` section 2, G2 verdict: PARTIAL).
- D* Lite is stronger still in several regimes and is reported as such
  (spec: publish it even if it wins).
- False-alarm budget met (<= 2.7/1000 stationary steps at rho=0.3).
- Trained weights verified end-to-end on held-out data (unseen change
  schedules): V1-V4 all PASS (`artifacts/verification/`).

## Install (local)

```bash
cd cavr-dyna
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
# or: pip install -e .
```

Python >= 3.10. No GPU needed (CPU-only, tabular).

## Run on Google Colab

The notebook **`notebooks/cavr_dyna_colab.ipynb`** clones this repository,
installs the pinned dependencies, runs the test suite, previews the committed
pilot artifacts, and can reproduce the smoke / pilot / final pipeline.

1. The notebook is preconfigured for
   `https://github.com/kishoreafk/Dyna-Q-Maze-Navigation--RL-solo-.git`.
   If you fork or rename the repository, update `REPO_URL` in the notebook's
   first code cell.
2. Open it directly in Colab:

   ```text
   https://colab.research.google.com/github/kishoreafk/Dyna-Q-Maze-Navigation--RL-solo-/blob/main/notebooks/cavr_dyna_colab.ipynb
   ```

3. Choose **Runtime → Run all**. The default mode only previews the committed
   results, so it is fast and safe. No GPU is needed; a CPU runtime is enough.

| Mode (cell 1) | What runs | Approx. CPU time |
| --- | --- | --- |
| View committed results (default) | preview + tests only | < 1 min |
| Smoke test | full pipeline on `configs/smoke.yaml` (7x7, 4 methods, 300 steps) | ~2-4 min |
| Reproduce pilot | `configs/pilot.yaml` (640 runs) + train/verify/plot/report | ~25-45 min |
| Run final protocol | `configs/final.yaml` (30 map/seed pairs, 6000 steps) | ~4-8 h |

All Colab runs write to `artifacts/colab_<mode>`; the committed
`artifacts/pilot` reference is never overwritten. Set
`USE_DRIVE_FOR_PROJECT=True` to keep the clone and outputs in Google Drive so
a disconnected runtime can resume. Set `FRESH_OUTPUT=True` to discard an
interrupted run and start over.

The same pipeline is available from the CLI:

```bash
python scripts/reproduce.py --config configs/pilot.yaml --out artifacts/colab_pilot
```

## Publish to GitHub

This checkout already targets the empty public repository
**`kishoreafk/Dyna-Q-Maze-Navigation--RL-solo-`**. From the project root:

```bash
git init -b main
git add .
git commit -m "CAVR-Dyna: research implementation, pilot artifacts, Colab notebook"
git remote add origin https://github.com/kishoreafk/Dyna-Q-Maze-Navigation--RL-solo-.git
git push -u origin main
```

If you use a different repository, create it on GitHub and replace the remote
URL (and the Colab link above) with your own. The committed artifacts are small
(a few MB); bulky per-step JSONL logs and resume markers are ignored by
`.gitignore` and regenerate at runtime.

## One-command reproducibility

`scripts/reproduce.py` runs the whole pipeline and writes to a separate output
directory, so the committed `artifacts/pilot` reference is protected:

```bash
pytest -q                                                      # 41 unit tests
python scripts/reproduce.py --config configs/smoke.yaml \
    --out artifacts/runs/smoke                                 # fast end-to-end check
python scripts/reproduce.py --config configs/pilot.yaml \
    --out artifacts/runs/pilot                                 # full pilot reproduction
python scripts/reproduce.py --config configs/final.yaml \
    --out artifacts/runs/final --skip-report                   # full protocol
```

Useful flags: `--skip-tests`, `--skip-train`, `--skip-verify`, `--skip-report`,
`--train-steps N`, `--force` (delete generated output first). `--report` also
generates `results.md` for non-pilot runs. Runs are resume-safe: completed
`(method, maze, scenario, rho, episode_seed)` keys are skipped when the output
directory already has a `metrics.csv`.

Under the hood (the same steps, runnable individually):

```bash
python scripts/make_manifest.py --config configs/pilot.yaml
python scripts/run_pilot.py --config configs/pilot.yaml          # calibration + canonical + grid
python scripts/train_model.py --config configs/pilot.yaml --steps 8000   # train + save weights
python scripts/verify_model.py --config configs/pilot.yaml        # held-out verification
python scripts/plot_results.py --art artifacts/pilot             # figures + tables
python scripts/make_results_report.py                            # reports/results.md
```

`scripts/run_pilot.py` accepts `--phase {manifest,calibration,canonical,grid}`
and `--scenarios/--methods` to run subsets. The full 30-pair protocol is
`scripts/run_final.py --config configs/final.yaml` (not executed in this
delivery; see the runtime note above).

## Repository layout

```
src/cavr/
  envs/        grid_env.py (private-map env + leak guard), maze_gen.py,
               change_scenarios.py
  models/      categorical_dirichlet.py, edge_belief.py, episodic_map.py,
               change_detector.py (two-level CUSUM)
  agents/      q_learning, dyna_q, dyna_q_plus, prioritized_sweeping,
               stochastic_dyna (Dirichlet + local forgetting), dstar_lite,
               cavr_dyna (+ cavr_nodetector, cavr_fixedB ablations)
  planning/    predecessors.py (versioned heap), counterfactual.py,
               probe.py, budget.py, dstar_lite.py
  evaluation/  runner.py, metrics.py, statistics.py, oracles.py,
               calibration.py
  utils/       seeding, logging, checkpoint
scripts/       make_manifest, run_pilot, run_final, reproduce, train_model,
               verify_model, plot_results, make_results_report, repair_logs
notebooks/     cavr_dyna_colab.ipynb (clone → install → test → run → download)
tests/         41 tests: env contract, model math, detector (incl. G1
               false-alarm stress), budget caps, stale-heap versions,
               leakage guards, determinism, baseline behaviors, CAVR
               behaviors, checkpoint roundtrip
configs/       smoke.yaml (fast CI/Colab check), pilot.yaml (executed),
               final.yaml (full protocol)
.github/       workflows/ci.yml (tests on 3.10-3.12 + smoke pipeline)
artifacts/     figures/, tables/, logs/ (JSONL.gz), checkpoints/ (weights),
               verification/, pilot/ (metrics + episodes CSVs, calibration),
               colab_*/ and runs/ (generated, git-ignored)
reports/       results.md (generated from real data), decisions.md,
               limitations.md
data/manifests/mazes_manifest.json   procedural mazes, split by ID, SHA256
```

## Artifacts map

| Artifact | Path |
| --- | --- |
| Model weights (Q, Dirichlet counts, edge beliefs, detector state) | `artifacts/checkpoints/cavr_trained*.npz`, per-run checkpoints |
| Metrics (640 runs) / episodes | `artifacts/pilot/metrics.csv`, `artifacts/pilot/episodes.csv` |
| Per-step JSONL logs (agent + evaluator streams) | `artifacts/pilot/logs/*.jsonl.gz`, `artifacts/logs/train_*.jsonl.gz` |
| Figures (canonical, learning curves, adaptation, Pareto, detector, beliefs) | `artifacts/figures/*.png` |
| Tables with per-condition summaries | `artifacts/tables/` |
| Calibration records | `artifacts/pilot/calibration_*.json` |
| Verification report (V1-V4) | `artifacts/verification/verification_report.{json,md}` |
| Training report | `artifacts/training_report.json` |
| Results / decisions / limitations | `reports/*.md` |

## Methods at a glance

| Method | Model | Change handling | Planning |
| --- | --- | --- | --- |
| q_learning | none | none | none |
| dyna_q | last-outcome | none | n=20 uniform |
| dyna_q_plus | last-outcome + age | age bonus | n=20 uniform + synthetic |
| prioritized_sweeping | last-outcome | none | heap n=20 + predecessors |
| dirichlet_dyna | Dirichlet categorical | none | n=20 expected backups |
| local_forgetting | Dirichlet categorical | CUSUM + local reset | n=32 uniform |
| dstar_lite | edge beliefs (MAP) | Bayes flip + streak rule | incremental search |
| cavr | Dirichlet + edge beliefs | pair-CUSUM + local forgetting | adaptive budget, counterfactual heap, probes |
| cavr_nodetector | as cavr | detector removed | as cavr |
| cavr_fixedB | as cavr | as cavr | fixed B=32 |

## Attribution and licenses

- Design: CAVR-Dyna 2026 research blueprint (provided as the specification).
- Inspired-by (not reproduced): Sutton 1990 Dyna; Moore & Atkeson 1993
  prioritized sweeping; Koenig & Likhachev 2002 D* Lite; 2019 incremental
  prioritized sweeping; 2023 local forgetting (PMLR v232); 2024 COPlanner,
  hallucinated-value Dyna, meta-gradient search control; 2026 ICLR ESWM /
  WIMLE / Sparse Imagination / HAUWM / Bayes-Adaptive MCTS; 2026 IEEE ICMA
  uncertainty-weighted Dyna; Oct 2026 CostGrad / SufficientPlan / adaptive
  robot metareasoning. See the blueprint for full citations.
- Code and generated artifacts: MIT (see `LICENSE`).
- No MovingAI or Zenodo dataset files are included or redistributed.
