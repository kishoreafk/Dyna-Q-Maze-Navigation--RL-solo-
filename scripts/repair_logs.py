#!/usr/bin/env python3
"""One-off repair: the first grid execution wrote metrics.csv / episodes.csv
rows via csv.writerow on nested lists (whole rows became single quoted
repr-fields). The data itself is intact; this script re-parses each line
with ast.literal_eval and rewrites proper CSVs. Idempotent.
"""
from __future__ import annotations

import ast
import csv
import os
import re
import sys

_NAN_RE = re.compile(r"(?<![\w'])nan(?![\w'])")


def _parse_repr(s: str):
    """literal_eval with nan token handling."""
    return ast.literal_eval(_NAN_RE.sub("None", s))

METRICS_HEADER = ["method", "maze_seed", "maze_id", "scenario", "rho", "episode_seed",
                  "steps", "t_change", "n_episodes", "success_rate", "success_pre",
                  "success_post", "adaptation_delay", "adaptation_censored",
                  "first_success_delay", "auc_success", "post_change_regret",
                  "mean_return_per_ep", "false_per_1000", "n_alarms", "n_false_alarms",
                  "detection_delay", "changes_detected", "changes_total",
                  "backups_total", "backups_per_step", "probes_total", "planning_ms",
                  "queue_ops", "run_wall_s"]

EP_HEADER = ["method", "maze_seed", "scenario", "rho", "episode_seed",
             "ep", "start_step", "end_step", "steps", "success", "ret"]


def _iter_rows(path: str):
    """Yield parsed Python lists from CSV lines whose single field is a
    repr'd list (handles the csv quoting around the repr)."""
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            if row[0] == "method":  # header
                continue
            if len(row) == 1 and row[0].strip().startswith("["):
                yield _parse_repr(row[0])
            else:
                yield row


def repair_metrics(path: str) -> int:
    rows = []
    for val in _iter_rows(path):
        if len(val) == 1 and isinstance(val[0], list):
            val = val[0]
        assert len(val) == len(METRICS_HEADER), f"bad row len {len(val)}"
        rows.append(val)
    tmp = path + ".fixed"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(METRICS_HEADER)
        for r in rows:
            w.writerow([round(v, 6) if isinstance(v, float) else v for v in r])
    os.replace(tmp, path)
    return len(rows)


def repair_episodes(path: str) -> int:
    rows = []
    for val in _iter_rows(path):
        # each line is a list of per-episode rows (each 11 fields)
        if len(val) == 11 and not isinstance(val[0], list):
            rows.append(val)
        else:
            for row in val:
                assert len(row) == len(EP_HEADER), f"bad ep row len {len(row)}"
                rows.append(row)
    tmp = path + ".fixed"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EP_HEADER)
        for r in rows:
            w.writerow([round(v, 6) if isinstance(v, float) else v for v in r])
    os.replace(tmp, path)
    return len(rows)


if __name__ == "__main__":
    art = sys.argv[1] if len(sys.argv) > 1 else "artifacts/pilot"
    n1 = repair_metrics(os.path.join(art, "metrics.csv"))
    n2 = repair_episodes(os.path.join(art, "episodes.csv"))
    print(f"metrics.csv: {n1} runs; episodes.csv: {n2} episodes")
