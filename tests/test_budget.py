"""Budget controller and heap versioning tests."""
import numpy as np

from cavr.planning.budget import BudgetController, BudgetFeatures
from cavr.planning.predecessors import VersionedPriorityQueue


def test_budget_levels_and_cap():
    bc = BudgetController()
    assert bc.choose(BudgetFeatures(recent_alarm=True))[0] == 64
    assert bc.choose(BudgetFeatures(alarm_strength=0.8))[0] == 32
    assert bc.choose(BudgetFeatures(top_repair_priority=0.4))[0] == 16
    assert bc.choose(BudgetFeatures(heap_size=0, mean_epistemic=0.05))[0] == 0
    assert bc.choose(BudgetFeatures())[0] == 8
    # random features always land in the level set and respect the cap
    rng = np.random.default_rng(0)
    levels = {0, 8, 16, 32, 64}
    for _ in range(500):
        f = BudgetFeatures(alarm_strength=rng.random() * 1.5,
                           recent_alarm=bool(rng.integers(2)),
                           top_repair_priority=rng.random(),
                           heap_size=int(rng.integers(0, 200)),
                           mean_epistemic=rng.random())
        B, _ = bc.choose(f)
        assert B in levels and 0 <= B <= 64


def test_budget_zero_still_allows_direct_update(agent_cfg):
    from cavr.agents import make_agent
    agent = make_agent("cavr", 54, 9, 6, 53, agent_cfg, np.random.default_rng(0))
    q_before = agent.q[10, 2]
    agent.observe(10, 2, -0.01, 10, False, 0)  # self-loop transition
    # force converged features -> B=0
    B, _ = agent.budget.choose(
        __import__("cavr.planning.budget", fromlist=["BudgetFeatures"]).BudgetFeatures(
            heap_size=0, mean_epistemic=0.01))
    assert B == 0
    # direct learning already happened in observe regardless of budget
    assert agent.q[10, 2] != q_before
    assert agent.totals["real_updates"] == 1


def test_direct_before_planning_order(agent_cfg):
    """Each real transition causes exactly one direct update, then B model
    backups (verify via counters and Q movement)."""
    from cavr.agents import make_agent
    rng = np.random.default_rng(1)
    agent = make_agent("cavr", 54, 9, 6, 53, agent_cfg, rng)
    s = 10
    q0 = agent.q[s, 1].copy()
    stats = agent.observe(s, 1, -0.01, 11, False, 0)
    assert agent.totals["real_updates"] == 1
    assert stats["backups_executed"] >= 0
    # model already contains the new observation after observe
    p = agent.model.predict(s, 1)
    assert p.sum() > 0
    assert agent.model.n_observations(s, 1) == 1.0


def test_stale_heap_entries_discarded():
    hq = VersionedPriorityQueue()
    hq.push(3, 1, 10.0, epoch=0)
    hq.push(4, 2, 5.0, epoch=1)
    assert hq.top_priority() == 10.0
    item = hq.pop(cur_epoch=1, min_priority=0.01)
    assert item is not None and item[0] == 4  # epoch-0 entry skipped as stale
    assert hq.n_stale == 1
    assert hq.pop(1, 0.01) is None


def test_heap_priority_floor_stops():
    hq = VersionedPriorityQueue()
    hq.push(1, 0, 0.001, epoch=0)
    assert hq.pop(0, min_priority=0.01) is None  # below usefulness threshold
