# CAVR-Dyna verification report (held-out test data)

- checkpoints verified: 4
- held-out verification slip: rho=0.3 (training slip was 0.1)
- schedules: NEW unseen edges/times chosen by the evaluator

| maze | static success | adaptation delay (steps) | censored | alarms | max |sum p - 1| | V1 | V2 | V3 |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1.00 | 503 | False | 6 | 2.22e-16 | PASS | PASS | PASS |
| 1001 | 1.00 | 11 | False | 9 | 2.22e-16 | PASS | PASS | PASS |
| 1002 | 0.97 | 6 | False | 7 | 2.22e-16 | PASS | PASS | PASS |
| 1003 | 1.00 | 44 | False | 7 | 2.22e-16 | PASS | PASS | PASS |

- V4 determinism (reloaded weights, identical seed): PASS

**Overall: PASS**

Notes: gates are verification gates for the saved weights, not the
research G-gates; see reports/results.md for the research evaluation.
