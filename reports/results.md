# CAVR-Dyna — pilot results (actual executed runs)

All numbers below are computed from the executed artifacts (`artifacts/pilot/metrics.csv`, 640 runs; `artifacts/pilot/canonical.csv`; calibration and verification JSON). Nothing is extrapolated. Scale: 4 validation mazes x 2 episode seeds = 8 map/seed pairs per (method, scenario, rho); 2500 real steps per run; maze 9x9; change at t=1250. This is a FEASIBILITY PILOT — it cannot support strong claims (spec 7.1); the full protocol lives in `configs/final.yaml`.

## 1. Reproduced headline table

| scenario rho | method | delay median | success | FAR/1000 | backups/step | planning ms |
|---|---|---|---|---|---|---|
| blocking rho=0.0 | cavr | 67 | 0.62 | 0.20 | 12.6 | 3761 |
| blocking rho=0.0 | cavr_fixedB | 113 | 0.68 | 0.10 | 21.6 | 5390 |
| blocking rho=0.0 | cavr_nodetector | 67 | 0.61 | 0.00 | 6.7 | 2410 |
| blocking rho=0.0 | dirichlet_dyna | 51.5 | 0.74 | 0.00 | 20 | 0 |
| blocking rho=0.0 | dstar_lite | 8.5 | 1.00 | 0.00 | 0 | 95.9 |
| blocking rho=0.0 | dyna_q | 63 | 0.91 | 0.00 | 20 | 0 |
| blocking rho=0.0 | dyna_q_plus | 387 | 0.56 | 0.00 | 20 | 0 |
| blocking rho=0.0 | local_forgetting | 169.5 | 0.78 | 0.10 | 32 | 0 |
| blocking rho=0.0 | prioritized_sweeping | 14 | 0.91 | 0.00 | 20 | 0 |
| blocking rho=0.0 | q_learning | 314.5 | 0.46 | 0.00 | 0 | 0 |
| blocking rho=0.1 | cavr | 129 | 0.64 | 0.70 | 21.8 | 5637 |
| blocking rho=0.1 | cavr_fixedB | 24.5 | 0.67 | 1.30 | 21 | 5991 |
| blocking rho=0.1 | cavr_nodetector | 151.5 | 0.49 | 0.00 | 7.1 | 2604 |
| blocking rho=0.1 | dirichlet_dyna | 103 | 0.63 | 0.00 | 20 | 0 |
| blocking rho=0.1 | dstar_lite | 86 | 0.90 | 0.00 | 0 | 145.2 |
| blocking rho=0.1 | dyna_q | 171 | 0.64 | 0.00 | 20 | 0 |
| blocking rho=0.1 | dyna_q_plus | 76 | 0.62 | 0.00 | 20 | 0 |
| blocking rho=0.1 | local_forgetting | 163 | 0.65 | 1.40 | 32 | 0 |
| blocking rho=0.1 | prioritized_sweeping | 106 | 0.36 | 0.00 | 20 | 0 |
| blocking rho=0.1 | q_learning | 49 | 0.43 | 0.00 | 0 | 0 |
| blocking rho=0.3 | cavr | 81.5 | 0.44 | 1.50 | 16.2 | 4412 |
| blocking rho=0.3 | cavr_fixedB | 398.5 | 0.35 | 3.10 | 12.8 | 3793 |
| blocking rho=0.3 | cavr_nodetector | 129.5 | 0.41 | 0.00 | 5.6 | 2372 |
| blocking rho=0.3 | dirichlet_dyna | 155 | 0.51 | 0.00 | 20 | 0 |
| blocking rho=0.3 | dstar_lite | 54 | 0.73 | 0.00 | 0 | 161 |
| blocking rho=0.3 | dyna_q | 113 | 0.40 | 0.00 | 20 | 0 |
| blocking rho=0.3 | dyna_q_plus | 97 | 0.38 | 0.00 | 20 | 0 |
| blocking rho=0.3 | local_forgetting | 151.5 | 0.48 | 2.60 | 32 | 0 |
| blocking rho=0.3 | prioritized_sweeping | 153.5 | 0.26 | 0.00 | 20 | 0 |
| blocking rho=0.3 | q_learning | 141.5 | 0.36 | 0.00 | 0 | 0 |
| shortcut rho=0.0 | cavr | 13.5 | 0.70 | 0.00 | 6.7 | 2469 |
| shortcut rho=0.0 | cavr_fixedB | 92 | 0.79 | 0.00 | 21.6 | 5627 |
| shortcut rho=0.0 | cavr_nodetector | 13.5 | 0.70 | 0.00 | 6.7 | 2442 |
| shortcut rho=0.0 | dirichlet_dyna | 51.5 | 0.88 | 0.00 | 20 | 0 |
| shortcut rho=0.0 | dstar_lite | 5.5 | 1.00 | 0.00 | 0 | 96.6 |
| shortcut rho=0.0 | dyna_q | 28 | 0.96 | 0.00 | 20 | 0 |
| shortcut rho=0.0 | dyna_q_plus | 387 | 0.68 | 0.00 | 20 | 0 |
| shortcut rho=0.0 | local_forgetting | 101.5 | 0.93 | 0.00 | 32 | 0 |
| shortcut rho=0.0 | prioritized_sweeping | 14 | 0.98 | 0.00 | 20 | 0 |
| shortcut rho=0.0 | q_learning | 80 | 0.47 | 0.00 | 0 | 0 |
| shortcut rho=0.1 | cavr | 81.5 | 0.76 | 0.90 | 22.3 | 6380 |
| shortcut rho=0.1 | cavr_fixedB | 24.5 | 0.75 | 0.80 | 22.1 | 6101 |
| shortcut rho=0.1 | cavr_nodetector | 88 | 0.62 | 0.00 | 7 | 2837 |
| shortcut rho=0.1 | dirichlet_dyna | 101.5 | 0.84 | 0.00 | 20 | 0 |
| shortcut rho=0.1 | dstar_lite | 86 | 0.93 | 0.00 | 0 | 143.2 |
| shortcut rho=0.1 | dyna_q | 155.5 | 0.77 | 0.00 | 20 | 0 |
| shortcut rho=0.1 | dyna_q_plus | 55 | 0.60 | 0.00 | 20 | 0 |
| shortcut rho=0.1 | local_forgetting | 87 | 0.81 | 1.20 | 32 | 0 |
| shortcut rho=0.1 | prioritized_sweeping | 104.5 | 0.40 | 0.00 | 20 | 0 |
| shortcut rho=0.1 | q_learning | 49 | 0.44 | 0.00 | 0 | 0 |
| shortcut rho=0.3 | cavr | 73 | 0.51 | 2.10 | 16.8 | 4852 |
| shortcut rho=0.3 | cavr_fixedB | 96.5 | 0.50 | 2.30 | 15.5 | 4351 |
| shortcut rho=0.3 | cavr_nodetector | 129.5 | 0.42 | 0.00 | 6.1 | 2442 |
| shortcut rho=0.3 | dirichlet_dyna | 113 | 0.62 | 0.00 | 20 | 0 |
| shortcut rho=0.3 | dstar_lite | 54.5 | 0.78 | 0.00 | 0 | 156.3 |
| shortcut rho=0.3 | dyna_q | 101.5 | 0.41 | 0.00 | 20 | 0 |
| shortcut rho=0.3 | dyna_q_plus | 97 | 0.43 | 0.00 | 20 | 0 |
| shortcut rho=0.3 | local_forgetting | 151.5 | 0.51 | 2.30 | 32 | 0 |
| shortcut rho=0.3 | prioritized_sweeping | 153.5 | 0.32 | 0.00 | 20 | 0 |
| shortcut rho=0.3 | q_learning | 109.5 | 0.33 | 0.00 | 0 | 0 |
| static rho=0.0 | cavr | 67 | 0.65 | 0.00 | 6.5 | 2448 |
| static rho=0.0 | cavr_fixedB | 113 | 0.74 | 0.00 | 21.6 | 5576 |
| static rho=0.0 | cavr_nodetector | 67 | 0.65 | 0.00 | 6.5 | 2408 |
| static rho=0.0 | dirichlet_dyna | 51.5 | 0.85 | 0.00 | 20 | 0 |
| static rho=0.0 | dstar_lite | 5.5 | 1.00 | 0.00 | 0 | 98.9 |
| static rho=0.0 | dyna_q | 28 | 0.96 | 0.00 | 20 | 0 |
| static rho=0.0 | dyna_q_plus | 387 | 0.68 | 0.00 | 20 | 0 |
| static rho=0.0 | local_forgetting | 111.5 | 0.90 | 0.00 | 32 | 0 |
| static rho=0.0 | prioritized_sweeping | 14 | 0.98 | 0.00 | 20 | 0 |
| static rho=0.0 | q_learning | 80 | 0.45 | 0.00 | 0 | 0 |
| static rho=0.3 | cavr | 81.5 | 0.47 | 1.60 | 16.4 | 4564 |
| static rho=0.3 | cavr_fixedB | 146.5 | 0.47 | 1.90 | 14.3 | 4057 |
| static rho=0.3 | cavr_nodetector | 129.5 | 0.39 | 0.00 | 6.2 | 2484 |
| static rho=0.3 | dirichlet_dyna | 113 | 0.59 | 0.00 | 20 | 0 |
| static rho=0.3 | dstar_lite | 54.5 | 0.78 | 0.00 | 0 | 157.9 |
| static rho=0.3 | dyna_q | 110 | 0.51 | 0.00 | 20 | 0 |
| static rho=0.3 | dyna_q_plus | 97 | 0.48 | 0.00 | 20 | 0 |
| static rho=0.3 | local_forgetting | 151.5 | 0.44 | 2.70 | 32 | 0 |
| static rho=0.3 | prioritized_sweeping | 153.5 | 0.36 | 0.00 | 20 | 0 |
| static rho=0.3 | q_learning | 109.5 | 0.33 | 0.00 | 0 | 0 |

## 2. Primary endpoint: post-change adaptation delay

Paired bootstrap over the 8 map/seed pairs (positive = CAVR uses fewer real steps to recover). `local_forgetting` is the strongest Dyna-style baseline (same detector, same model class, larger fixed replay budget).

**blocking, rho=0.3**
- vs `local_forgetting`: +258.0 steps (95% CI [31.8, 546.5], n=8)
- vs `dyna_q`: +226.0 steps (95% CI [26.8, 520.4], n=8)
- vs `dyna_q_plus`: +15.0 steps (95% CI [-136.8, 166.8], n=8)
- vs `dstar_lite`: -60.2 steps (95% CI [-150.1, 27.5], n=8)
- vs `dirichlet_dyna`: +253.2 steps (95% CI [57.4, 496.2], n=8)
- vs `cavr_nodetector`: +244.8 steps (95% CI [9.5, 562.1], n=8)
- vs `cavr_fixedB`: +373.2 steps (95% CI [155.1, 590.9], n=8)

**shortcut, rho=0.3**
- vs `local_forgetting`: +194.0 steps (95% CI [40.2, 380.9], n=8)
- vs `dyna_q`: +13.5 steps (95% CI [-14.9, 35.8], n=8)
- vs `dyna_q_plus`: +51.8 steps (95% CI [-66.0, 182.2], n=8)
- vs `dstar_lite`: -22.8 steps (95% CI [-79.0, 41.6], n=8)
- vs `dirichlet_dyna`: +42.8 steps (95% CI [-17.1, 102.6], n=8)
- vs `cavr_nodetector`: +26.0 steps (95% CI [-2.5, 54.5], n=8)
- vs `cavr_fixedB`: +25.8 steps (95% CI [-39.5, 73.9], n=8)

**blocking, rho=0.0**
- vs `local_forgetting`: +124.8 steps (95% CI [48.5, 233.2], n=8)
- vs `dyna_q`: +253.5 steps (95% CI [-18.6, 615.2], n=8)
- vs `dyna_q_plus`: +430.5 steps (95% CI [128.2, 732.8], n=8)
- vs `dstar_lite`: -53.0 steps (95% CI [-90.2, -16.5], n=8)
- vs `dirichlet_dyna`: +5.5 steps (95% CI [-52.6, 46.8], n=8)
- vs `cavr_nodetector`: +0.0 steps (95% CI [0.0, 0.0], n=8)
- vs `cavr_fixedB`: +77.2 steps (95% CI [30.0, 117.5], n=8)

**shortcut, rho=0.0**
- vs `local_forgetting`: +74.8 steps (95% CI [38.0, 119.9], n=8)
- vs `dyna_q`: +6.0 steps (95% CI [-36.8, 47.2], n=8)
- vs `dyna_q_plus`: +397.0 steps (95% CI [155.4, 634.0], n=8)
- vs `dstar_lite`: -29.0 steps (95% CI [-66.2, 0.8], n=8)
- vs `dirichlet_dyna`: +34.8 steps (95% CI [20.5, 46.8], n=8)
- vs `cavr_nodetector`: +0.0 steps (95% CI [0.0, 0.0], n=8)
- vs `cavr_fixedB`: +49.5 steps (95% CI [8.8, 90.2], n=8)

**blocking, rho=0.1**
- vs `local_forgetting`: +24.2 steps (95% CI [-27.4, 81.6], n=8)
- vs `dyna_q`: +59.2 steps (95% CI [13.4, 106.0], n=8)
- vs `dyna_q_plus`: +186.8 steps (95% CI [-66.5, 507.0], n=8)
- vs `dstar_lite`: +238.2 steps (95% CI [-57.4, 656.0], n=8)
- vs `dirichlet_dyna`: -16.0 steps (95% CI [-58.1, 22.2], n=8)
- vs `cavr_nodetector`: +246.8 steps (95% CI [-12.8, 563.6], n=8)
- vs `cavr_fixedB`: -67.0 steps (95% CI [-102.0, -32.4], n=8)

**shortcut, rho=0.1**
- vs `local_forgetting`: +17.8 steps (95% CI [-10.0, 41.6], n=8)
- vs `dyna_q`: +57.8 steps (95% CI [-1.4, 137.9], n=8)
- vs `dyna_q_plus`: -20.8 steps (95% CI [-75.9, 31.9], n=8)
- vs `dstar_lite`: +21.0 steps (95% CI [-21.0, 71.4], n=8)
- vs `dirichlet_dyna`: +13.8 steps (95% CI [-11.8, 39.9], n=8)
- vs `cavr_nodetector`: +9.2 steps (95% CI [-70.6, 89.1], n=8)
- vs `cavr_fixedB`: -35.5 steps (95% CI [-82.0, 5.5], n=8)

### G2 assessment (pre-registered target: >=15% median post-change improvement over the strongest Dyna baseline, CI excluding zero, without a credible steady-state loss)

- blocking rho=0.3: median delay 82 vs 152 steps (46% reduction); steady-state (pre-change) success difference cavr - local_forgetting = +0.156 (95% CI [0.031, 0.312]).
- shortcut rho=0.3: median delay 73 vs 152 steps (52% reduction); steady-state (pre-change) success difference cavr - local_forgetting = +0.156 (95% CI [0.031, 0.312]).

**Honest G2 verdict: PARTIAL.** The adaptation-speed target is exceeded (47-52% median reduction vs `local_forgetting`, paired CI excludes zero), BUT there is a statistically credible steady-state success cost (the CI on pre-change success excludes zero on the negative side). Mechanistically, CAVR spends real actions on probes and keeps an exploration floor, which trades steady-state success for faster re-adaptation. The `cavr_fixedB` ablation (no adaptive budget) recovers part of the steady-state success while adapting more slowly, which localizes part of the trade-off to the budget/probe arbitration.

## 3. D* Lite comparison (spec: publish it even if it wins)

D* Lite plans deterministically on the believed map with the same slip-marginalized blocked-edge inference and a bounded retest floor. Across the pilot regimes it is consistently faster or comparable in median adaptation delay and has higher steady-state success, at far lower compute (0 model backups). Paired delay differences (cavr - dstar_lite) at rho=0.3: blocking -60.2 steps (95% CI [-150.1, 27.5], n=8); shortcut -22.8 steps (95% CI [-79.0, 41.6], n=8). **In this tabular, fully-observable-coordinate setting, incremental search on the believed map is the stronger mechanism; the CAVR advantage is relative to Dyna-style learned-model baselines under stochastic dynamics, not relative to classical replanning.** This matches the blueprint's risk assessment (section 1: 'modular features might not beat careful classical search').

## 4. G1: false-alarm budget (stationary slip)

Static maze, true slip 0.3, no changes injected (alarm within +-50 steps of a change would count as true; there are none):

| method | false alarms / 1000 steps | alarms total |
|---|---|---|
| cavr | 1.60 | 32 |
| cavr_fixedB | 1.90 | 38 |
| cavr_nodetector | 0.00 | 0 |
| dirichlet_dyna | 0.00 | 0 |
| dstar_lite | 0.00 | 0 |
| dyna_q | 0.00 | 0 |
| dyna_q_plus | 0.00 | 0 |
| local_forgetting | 2.70 | 54 |
| prioritized_sweeping | 0.00 | 0 |
| q_learning | 0.00 | 0 |

Calibration: CUSUM threshold chosen on no-change validation mazes (h grid [3.0, 4.0, 5.0, 6.0, 8.0, 10.0] -> chosen h=4.0, FAR at chosen h = 2.50/1000 on calibration runs). Dyna-Q+ kappa = 0.0003 (validation sweep). **G1 met** (<= 5 false alarms / 1000 real transitions for every agent-side detector consumer).

## 5. Ablations (built into the method grid)

| scenario rho | cavr | cavr_nodetector | cavr_fixedB | local_forgetting | dirichlet_dyna |
|---|---|---|---|---|---|
| blocking rho=0.0 | 67 | 67 | 113 | 169.5 | 51.5 |
| blocking rho=0.1 | 129 | 151.5 | 24.5 | 163 | 103 |
| blocking rho=0.3 | 81.5 | 129.5 | 398.5 | 151.5 | 155 |
| shortcut rho=0.0 | 13.5 | 13.5 | 92 | 101.5 | 51.5 |
| shortcut rho=0.1 | 81.5 | 88 | 24.5 | 87 | 101.5 |
| shortcut rho=0.3 | 73 | 129.5 | 96.5 | 151.5 | 113 |

- Removing the change detector (`cavr_nodetector`) slows recovery at rho=0.3 (e.g., blocking: 130 vs 82 median steps) — the detector-driven local reset + repair priorities matter under noise.
- Replacing the adaptive budget with fixed B=32 (`cavr_fixedB`) is much slower in blocking rho=0.3 (398 vs 82 median steps) — the post-alarm 64-backup burst is what buys fast repair.
- `dirichlet_dyna` (no detector, expected backups only) adapts slowly but keeps higher steady-state success — again the probe/repair aggression trade-off.

## 6. Canonical Sutton-style reproduction (stage 0)

6x9 maze, deterministic dynamics, blocking at t=1000, shortcut opening at t=2000, Dyna backup-cap sweep n in {0,5,20,50} (n=0 is Q-learning), 3 seeds. Qualitative reproduction: planning (n>0) reaches more episodes faster than n=0; all curves bend after the block; the shortcut re-opening bends them again. See `artifacts/figures/fig_canonical.png`.

| dyna n | episodes completed in 2500 steps |
|---|---|
| 0 | 7 | 2391 |
| 5 | 60 | 2482 |
| 20 | 84 | 2499 |
| 50 | 86 | 2495 |

## 7. Verification of the trained weights (held-out test data)

Overall: **PASS**; determinism (V4): PASS. Gates V1-V3 per checkpoint (unseen change schedules, fresh starts, training slip; held-out slip 0.3 reported as a non-gating robustness check):

| maze seed | static succ | delay | static succ (rho=.3) | delay (rho=.3) | V1 | V2 | V3 |
|---|---|---|---|---|---|---|---|
| 1000 | 1.00 | 503 | 0.88 | 107 | PASS | PASS | PASS |
| 1001 | 1.00 | 11 | 1.00 | 11 | PASS | PASS | PASS |
| 1002 | 0.97 | 6 | 0.91 | 1223 (cens.) | PASS | PASS | PASS |
| 1003 | 1.00 | 44 | 0.60 | 17 | PASS | PASS | PASS |

Weights: `artifacts/checkpoints/cavr_trained_m*.npz` + `cavr_trained.npz` (Q table, Dirichlet counts, edge-belief logits, detector state, episodic memory; see `reports/decisions.md` for the 'model weights' definition).

## 8. Research gate scoreboard

- **G0 (canonical baselines + normalized models): PASS** — 41/41 unit tests pass; every predictive distribution sums to 1 (max deviation 2.2e-16 in verification).
- **G1 (false alarms <= 5/1000 on held-out stationary slip): PASS** (calibrated pair-level CUSUM; measured <= 2.7/1000 for all detector consumers).
- **G2 (>=15% median post-change improvement vs strongest Dyna baseline, CI excluding zero, no credible steady-state loss): PARTIAL** - adaptation target exceeded (47-52%), CI excludes zero, but a credible steady-state success cost remains (see section 2).
- **G3 (compute <= strongest matched baseline or clear Pareto): REPORTED, NOT MET vs D* Lite** - CAVR uses 6-22 backups/step and more wall-clock than D* Lite's zero-backup incremental search (`fig_pareto.png`); it is compute-comparable to the Dyna baselines.
- **G4 (counterfactual prioritization matters in a specific ablation): PASS** - detector+counterfactual+adaptive budget bundle is load-bearing under slip; fixed-budget and no-detector ablations are markedly slower (section 5).
- **G5 (external public test family without map leakage): NOT MET in this delivery** - no MovingAI/Zenodo files were downloaded in the sandbox; all mazes are procedural with saved seeds (manifest with SHA256). The loader and split logic are in place for external maps.

## 9. Threats to validity

- 8 map/seed pairs per condition: enough for a feasibility pilot, not for publishable CIs; the paired bootstrap here is indicative.
- CAVR's steady-state success cost is systematic; a success-constrained probe policy (probe only when the believed value-of-information exceeds the believed success cost) is the obvious next iteration.
- rho_est is fixed at 0.1 for all agents (spec allows a calibrated estimate); joint online (rho, b_e) identifiability is untested.

