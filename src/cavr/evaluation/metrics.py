"""Episode/step metrics: rolling success, adaptation delay, AULC, regret,
false-alarm statistics."""
from __future__ import annotations

import numpy as np


def rolling_success(ep_records: list[dict], window: int = 8) -> np.ndarray:
    """Rolling mean of per-episode success (trailing window)."""
    succ = np.asarray([e["success"] for e in ep_records], dtype=float)
    out = np.empty(len(succ))
    for i in range(len(succ)):
        lo = max(0, i - window + 1)
        out[i] = succ[lo:i + 1].mean()
    return out


def success_rate(ep_records: list[dict], t0: int = 0, t1: int | None = None) -> float:
    sel = [e for e in ep_records if e["end_step"] >= t0 and (t1 is None or e["end_step"] < t1)]
    if not sel:
        return float("nan")
    return float(np.mean([e["success"] for e in sel]))


def adaptation_delay(ep_records: list[dict], t_change: int, window: int = 8,
                     frac: float = 0.9) -> tuple[float, bool]:
    """Steps from t_change until the rolling success rate recovers to frac of
    the stable pre-change level. Returns (delay, censored).

    If the agent had no stable pre-change success, the metric degrades to
    'first success after the change' (discovery delay).
    """
    pre = [e for e in ep_records if e["end_step"] < t_change]
    post = [e for e in ep_records if e["end_step"] >= t_change]
    if not post:
        return float("nan"), True
    rs = rolling_success(ep_records, window)
    S_pre = 0.0
    if pre:
        n_pre = min(window, len(pre))
        S_pre = float(np.mean([e["success"] for e in pre[-n_pre:]]))
    threshold = frac * max(S_pre, 1.0 / window)
    t_end = ep_records[-1]["end_step"]
    for e, r in zip(ep_records, rs):
        if e["end_step"] >= t_change and r >= threshold:
            return float(e["end_step"] - t_change), False
    return float(t_end - t_change), True


def first_success_delay(ep_records: list[dict], t_change: int) -> tuple[float, bool]:
    for e in ep_records:
        if e["end_step"] >= t_change and e["success"]:
            return float(e["end_step"] - t_change), False
    t_end = ep_records[-1]["end_step"] if ep_records else t_change
    return float(t_end - t_change), True


def auc_success(ep_records: list[dict], total_steps: int, window: int = 8) -> float:
    """Time-weighted area under the rolling-success curve, normalized to [0,1]."""
    if not ep_records:
        return 0.0
    rs = rolling_success(ep_records, window)
    total = 0.0
    for e, r in zip(ep_records, rs):
        total += r * e["steps"]
    return float(total / max(1, total_steps))


def mean_reward_rates(ep_records: list[dict], t_change: int, warmup: int = 200):
    pre = [e for e in ep_records if e["end_step"] >= warmup and e["start_step"] < t_change]
    post = [e for e in ep_records if e["start_step"] >= t_change]
    r_pre = (sum(e["ret"] for e in pre) / max(1, sum(e["steps"] for e in pre))) if pre else float("nan")
    r_post = (sum(e["ret"] for e in post) / max(1, sum(e["steps"] for e in post))) if post else float("nan")
    return r_pre, r_post


def post_change_regret(ep_records: list[dict], t_change: int, warmup: int = 200) -> float:
    """Signed cumulative reward change after the event: negative = loss,
    positive = improvement (e.g., discovered shortcut)."""
    r_pre, r_post = mean_reward_rates(ep_records, t_change, warmup)
    if r_pre != r_pre or r_post != r_post:
        return float("nan")
    post_steps = sum(e["steps"] for e in ep_records if e["start_step"] >= t_change)
    return float((r_post - r_pre) * post_steps)


def alarm_stats(alarm_ts: list[int], change_ts: list[int], total_steps: int,
                window: int = 50) -> dict:
    """False-alarm statistics. An alarm is 'true' if it falls within +/-window
    steps of a real change; detection delay measured forward from the change."""
    alarms = sorted(alarm_ts)
    true_hits = []
    remaining = list(change_ts)
    false_ts = []
    for a in alarms:
        matched = False
        for ct in list(remaining):
            if abs(a - ct) <= window:
                true_hits.append(a - ct)
                remaining.remove(ct)
                matched = True
                break
        if not matched:
            false_ts.append(a)
    detection_delay = float("nan")
    if change_ts:
        for ct in sorted(change_ts):
            cand = [a for a in alarms if 0 <= a - ct <= 200]
            if cand:
                detection_delay = float(cand[0] - ct)
                break
    return {
        "n_alarms": len(alarms),
        "n_false": len(false_ts),
        "false_per_1000": 1000.0 * len(false_ts) / max(1, total_steps),
        "detection_delay": detection_delay,
        "n_changes_detected": len(true_hits),
        "n_changes_total": len(change_ts),
    }
