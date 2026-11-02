# Implementation decisions (spec section 13: "document the choice in reports/decisions.md")

Every unspecified detail below was resolved with the simplest reproducible
option. Each choice lists the consequence.

## Scope and scale

1. **Pilot-scale execution.** The blueprint's full protocol (30 map/seed
   pairs, 11x11 mazes, 4 scenarios, 3 slips, 6000 steps) is configured in
   `configs/final.yaml` but was NOT executed in this delivery (sandbox
   runtime). The executed pilot (`configs/pilot.yaml`) is the same pipeline
   scaled to 4 validation mazes x 2 episode seeds = 8 map/seed pairs,
   9x9 mazes, 2500 steps, scenarios {static, blocking, shortcut}, rho
   {0, 0.1, 0.3}. Consequence: CIs are indicative only (feasibility pilot,
   spec 7.1 explicitly allows this).
2. **Stage 4 (optional <1M-parameter neural prior) not implemented.** The
   blueprint gates it on earlier stages passing and marks it optional; the
   tabular stack is complete. Consequence: "model weights" = tabular
   checkpoint arrays (below).
3. **External datasets not downloaded.** Neither MovingAI maps nor the 2026
   Zenodo orthogonal-maze dataset were fetched in the sandbox. Per the
   master prompt ("do not fake a successfully downloaded dataset"), all
   mazes are procedural with saved seeds, split by maze identity, manifest
   with SHA256 (`data/manifests/mazes_manifest.json`). Licensing notes:
   MovingAI maps are ODC-By (attribution required, per-map exceptions
   possible); the Zenodo record 19990818 displays "Copyright (C) 2026 The
   Authors" without a permissive license — its raw files must not be
   redistributed. Consequence: G5 (external validation) is NOT met in this
   delivery; the split/manifest machinery is dataset-agnostic and ready.

## Environment

4. **Reward convention:** +1 at goal (terminal, replaces the step reward),
   -0.01 per step, additional -0.02 on any self-loop (blocked or
   out-of-bounds). Fixed across every method. gamma = 0.97.
5. **Truncation vs terminal:** reaching the goal sets terminal (no
   bootstrap); hitting the episode step cap sets truncation and the agent
   still bootstraps from the truncated next state (standard time-limit
   handling; spec test `test_q_terminal` distinguishes the two).
6. **Starts and goals:** one fixed goal per maze (Q is goal-indexed by
   construction; the spec's goal-conditioning concern is documented rather
   than silently ignored); episode starts vary over a seeded set of 31
   cells (start cell + 30 random non-goal cells).
7. **Scenario construction:** blocking closes an edge of the current
   shortest path that keeps the maze connected and strictly increases the
   path length; shortcut opens the closed edge with the largest path-length
   reduction, falling back to the edge with the largest total goal-distance
   improvement on fully braided mazes (a "frontier opening" with identical
   discovery semantics). Schedules are deterministic, evaluator-only.

## Models

8. **Dirichlet categories:** K=5 {N, E, S, W, self}, beta=0.5 prior.
   Categories map the OBSERVED outcome (move direction or stay), so the
   learned categorical already marginalizes action slip implicitly.
   Out-of-bounds categories map to self (blocked semantics).
9. **Epistemic uncertainty:** sum of Dirichlet posterior variances
   Var_k = a_k(a0-a_k)/(a0^2(a0+1)), normalized by 0.2 (chosen so an
   unobserved (s,a) maps to ~1.0). Predictive entropy is kept as the
   separate aleatoric estimate; the two are never mixed
   (`test_epistemic_decays_with_data_but_entropy_stays`).
10. **Edge beliefs:** logistic parameterization, prior b0=0.6, updates are
    3 inner gradient-ascent steps on the slip-marginalized log-likelihood
    (approximates a stronger Bayes step so a truly blocked edge crosses the
    MAP boundary in ~2 contradictory observations while stationary slip
    keeps beliefs diffuse; validated by tests). Lazy exponential decay
    toward the prior (rate 0.002/step). Symmetry is structural (one
    canonical undirected key).
11. **Change detection:** two-level CUSUM. The per-(s,a) pair statistic
    accumulates winsorized (cap 2.0) excess surprise z - H(p_before) with a
    slow per-pair drift (0.02); a global statistic is kept for logging.
    The adaptive null H(p_before) = predictive entropy makes a
    well-modeled stationary process non-accumulating. Localization is free
    (the crossing pair). Threshold h calibrated on no-change validation
    mazes, rho=0.3, target <= 3/1000 (60% of the 5/1000 budget): chosen
    h=4.0 (FAR 2.5/1000 on calibration runs; measured 0.7-2.7/1000 in the
    pilot grid). Local forgetting resets Dirichlet counts in the
    1-radius cell neighborhood to the prior, then re-incorporates the
    latest PREVIOUS record per pair; the new transition is incorporated
    exactly once after the reset (ordering documented in the module
    docstring; "never double-count" is unit-tested).

## Planning and control

12. **Counterfactual edge impact D_e:** closed-form one-edge flip with all
    other beliefs fixed: D_e = |P(u) * [(r_move - r_stay) + gamma (V(y) -
    V(x))]|, maximized over the commanded/slipped direction. This is a
    HEURISTIC influence estimate, not exact information gain (spec caution).
13. **Repair priorities:** score = (p_change + 0.2 * edge uncertainty) *
    D_e * reach_weight + 0.05 * age_term, pushed as (state, action) backup
    targets; priority clipped to >= 0.05 so repairs outrank the usefulness
    floor theta_useful = 0.01.
14. **Reachability:** discounted probability-mass rollout, H=12, discount
    0.9, epsilon=0.1 policy, using the edge-mixture predictor only (no true
    map); refreshed every 20 steps and on alarms (compute).
15. **Budget controller:** rule-based levels {0, 8, 16, 32, 64} from public
    features (recent alarm -> 64; alarm strength >= 0.5 -> 32; top repair
    priority >= 0.15 -> 16; empty heap and mean epistemic < 0.10 -> 0;
    else 8). Learned MLP gating explicitly out of scope. Early stop when
    the popped priority falls below theta_useful.
16. **Probing:** real-action probes distinct from backups. Scores combine
    4b(1-b) uncertainty weighted by reachability and a Dyna-Q+-style
    retest term (1-b) * potential_gain * sqrt(age)/30 for believed-blocked
    edges; travel over believed-open edges (BFS, b>0.5); commitment
    probability 0.9, min gap 8 steps; explicit frontier floor: 2% random
    actions + epsilon=0.1 — the nonzero exploration floor required by the
    spec (shortcut rediscovery is unit-tested).
17. **D* Lite baseline:** goal-rooted D* Lite (Koenig & Likhachev 2002)
    with keys using h(agent, vertex) and the standard km correction;
    believed-open edges cost 1, believed-blocked INF; blocked-edge
    inference uses the same slip-marginalized Bayes update plus a
    consecutive-self-loop hard-flip rule (3 loops at rho_est<=0.15, else 5)
    — classic terrain observation, agent-side only; bounded retest floor
    (every 50 steps, p=0.5, oldest believed-blocked adjacent edge).
    Full-rebuild fallback guards queue inconsistencies.
18. **Dyna-Q+ synthetic actions:** never-tried actions of visited states
    are modeled as self-transitions with zero reward, sampled with
    probability 0.2 during planning; age bonus kappa*sqrt(tau) is added in
    simulated backups only (never to logged real rewards), kappa=0.0003
    from the validation sweep.

## Logging and checkpoints

19. **Per-step logs:** gzipped JSONL with the spec-required fields
    (step, maze_id, seed, action, reward, success, model_uncertainty,
    predicted_change_probability, budget_assigned, backups_executed,
    queue_operations, wall_time_ms, model_version, proposed_probe,
    actual_action). Evaluator-only ground truth (change events, true edge
    set, schedule) goes to a separate `*_eval.jsonl.gz` stream and to the
    run summary — never into the agent stream. To bound artifact size,
    per-step logs cover the episode_seed==0 subset (~80 of 640 runs) plus
    all training/verification streams; all runs contribute complete
    metrics/episode CSVs.
20. **"Model weights":** `artifacts/checkpoints/*.npz` containing the Q
    table, Dirichlet counts (+ reward moments), edge-belief logits,
    detector statistics, episodic memory records and the full agent config
    in metadata. Reload is exercised by `test_cavr_checkpoint_roundtrip`
    and by `scripts/verify_model.py` end-to-end.
21. **CSV repair:** the first grid execution wrote metrics/episodes rows
    through a misused csv helper (rows became repr'd single fields). The
    writer call sites were fixed and the grid re-executed (results are
    seed-deterministic and identical); `scripts/repair_logs.py` remains in
    the repo documenting the repair path for any legacy artifact.

## Verification protocol

22. **Verification gates** (scripts/verify_model.py, distinct from the
    research G-gates): V1 static success >= 0.7 before the first unseen
    change; V2 rolling-success recovery to 90% of the pre-change level
    within 30% of the horizon; V3 all predictive distributions sum to 1
    (< 1e-6) and edge beliefs symmetric; V4 identical seed -> identical
    streams with reloaded weights. Gates run at the TRAINING slip with
    unseen change schedules (new edges and times chosen by the evaluator);
    a held-out slip rho=0.3 pass is reported as a non-gating robustness
    check (distribution shift). All four mazes PASS.

## Known code-level limitations

23. The probe selector's travel path uses believed-open BFS only; if the
    believed map disconnects the target, the probe is skipped that step
    (no evidence-seeking path through believed-blocked terrain).
24. `PredecessorGraph` syncs (s,a) contributions eagerly on every model
    change affecting them; on local reset it resyncs the whole region
    (bounded by region size).
25. Wall-clock measurements are single-process, sandbox-contended, and
    used only for Pareto comparisons, never as absolute claims.
