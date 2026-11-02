# Scientific limitations and negative findings (honest disclosure per spec 12)

1. **D* Lite wins several regimes.** In this tabular, coordinate-observable
   maze setting, incremental search over the believed map (with the same
   slip-marginalized blocked-edge inference and a bounded retest floor)
   adapts as fast or faster than CAVR and with higher steady-state success,
   at a fraction of the compute. The CAVR advantage demonstrated here is
   relative to Dyna-style learned-model baselines (local forgetting, Dyna-Q,
   Dyna-Q+, Dirichlet Dyna), not relative to classical incremental
   replanning. Any novelty claim must be narrowed accordingly (spec 12:
   claims to avoid).

2. **Steady-state success cost.** CAVR's probing floor and repair-driven
   exploration trade steady-state success for adaptation speed: the paired
   pre-change success difference vs `local_forgetting` at rho=0.3 is
   -0.156 (95% CI [-0.312, -0.031]). The pre-registered G2 gate required
   adaptation gains WITHOUT a credible steady-state loss; the gain is real
   (47-52% median delay reduction, CI excluding zero) but the no-loss
   condition is not met. The `cavr_fixedB` ablation shows the adaptive
   budget contributes to both effects.

3. **Pilot scale.** 8 map/seed pairs per condition, 4 mazes, 2500 steps.
   This supports feasibility statements only; it cannot support strong
   claims (spec 7.1). The full protocol is configured in
   `configs/final.yaml` and was not executed here.

4. **No external-dataset validation (G5 not met).** All mazes are
   procedural with saved seeds; no MovingAI or Zenodo files were downloaded
   or redistributed. OOD behavior across map families is untested.

5. **Slip estimate is fixed, not estimated.** All agents receive
   rho_est=0.1 regardless of the true slip (the spec permits a held-out
   calibrated estimate). Joint online identifiability of (slip, edge-open
   probability) is untested; a single self-loop is genuinely ambiguous.

6. **Passive detection limits.** A change to an edge the agent never
   touches is undetectable by construction; discovery of opened shortcuts
   relies on the retest floor. Detection delay under sparse evidence is
   bounded by visitation, not by the detector (identifiability limitation
   noted in spec 15.1).

7. **Counterfactual influence is a heuristic.** D_e with other beliefs
   marginalized is NOT a value-of-information computation; the blueprint's
   caution applies, and the expensive ground-truth planning-improvement
   diagnostic was not run.

8. **Compute claims are relative.** Wall-clock numbers come from a
   single-process sandbox and are used only for method comparisons in the
   same environment; no absolute speed claims are made.

9. **Stage-4 neural prior absent.** The optional <1M-parameter learned
   prior was not implemented; nothing here should be read as reproducing
   ESWM, WIMLE, Sparse Imagination or CostGrad — only their general
   mechanisms informed the design (episodic local memory, confidence-
   weighted imagination, coverage floors, decision-dependent budgets).

10. **Reward-shaping sensitivity unexplored.** The collision penalty
    (-0.02) and step penalty (-0.01) were fixed once; results may differ
    under other conventions (spec requires fixing across methods, which
    was done).
