"""Model correctness tests: normalization, symmetry, expected backups,
terminal bootstrap, local reset semantics."""
import numpy as np
import pytest

from cavr.models.categorical_dirichlet import CategoricalDirichlet, category_of
from cavr.models.edge_belief import EdgeBeliefModel
from cavr.models.change_detector import ChangeDetector
from cavr.envs.maze_gen import edge_key


def test_predict_sums_to_one():
    m = CategoricalDirichlet(16, 4, beta=0.5)
    for s in range(16):
        for a in range(4):
            p = m.predict(s, a)
            assert abs(p.sum() - 1.0) < 1e-12


def test_category_mapping():
    W = 4
    assert category_of(5, 1, W) == 0   # N
    assert category_of(5, 6, W) == 1   # E
    assert category_of(5, 9, W) == 2   # S
    assert category_of(5, 4, W) == 3   # W
    assert category_of(5, 5, W) == 4   # self
    with pytest.raises(ValueError):
        category_of(5, 15, W)


def test_expected_backup_matches_manual():
    """Two-outcome example computed by hand (spec: expected backup equals the
    manually calculated expectation)."""
    m = CategoricalDirichlet(4, 2, beta=0.0)  # no prior mass -> exact counts
    # force counts: (s=0,a=0): 0.7 to E neighbor (s=1), 0.3 self
    m.counts[0, 0] = np.array([1e-9, 0.7, 1e-9, 1e-9, 0.3])
    m.r_cnt[0, 0, 1] = 1.0
    m.r_sum[0, 0, 1] = -0.01
    m.r_cnt[0, 0, 4] = 1.0
    m.r_sum[0, 0, 4] = -0.03
    # value function: V(s1)=0.5, V(s0)=0.2 (we patch q below in the agent)
    q = np.zeros((4, 4))
    q[1] = 0.5
    q[0] = 0.2

    class _AgentStub:
        gamma = 0.97
        q_ref = q
        goal_idx = 3

        def V(self):
            v = self.q_ref.max(axis=1)
            v[self.goal_idx] = 0.0
            return v

    from cavr.agents.stochastic_dyna import StochasticDynaBase
    agent = StochasticDynaBase.__new__(StochasticDynaBase)
    agent.gamma = 0.97
    agent.q = q
    agent.goal_idx = 3
    agent.model = m
    agent.alpha_model = 0.0  # no actual update; we only inspect the target
    agent.rng = np.random.default_rng(0)
    agent._V = q.max(axis=1)
    agent._V[3] = 0.0
    targ = agent.expected_target(0, 0)
    manual = 0.7 * (-0.01 + 0.97 * 0.5) + 0.3 * (-0.03 + 0.97 * 0.2)
    assert abs(targ - manual) < 1e-9


def test_epistemic_decays_with_data_but_entropy_stays():
    m = CategoricalDirichlet(16, 4, beta=0.5)
    e0 = m.epistemic(5, 0)
    # feed uniform-looking data (aleatoric slip): after many observations the
    # epistemic term must vanish while predictive entropy stays high
    rng = np.random.default_rng(0)
    for _ in range(400):
        k = 4 if rng.random() < 0.3 else int(rng.integers(0, 4))
        s2 = {0: 1, 1: 6, 2: 9, 3: 4}.get(k, 5)  # neighbors of state 5 (W=4)
        m.update(5, 0, s2, -0.01)
    e1 = m.epistemic(5, 0)
    ent = m.entropy(5, 0)
    assert e0 > 0.9 and e1 < 0.15
    assert ent > 1.0  # still multi-modal predictive


def test_terminal_zero_bootstrap(agent_cfg):
    from cavr.agents import make_agent
    rng = np.random.default_rng(0)
    agent = make_agent("q_learning", 16, 4, 4, 15, agent_cfg, rng)
    agent.q[7, 1] = 0.3
    agent.q[15] = 0.0  # goal row stays zero
    delta = agent.direct_update(7, 1, 1.0, 15, terminal=True)
    assert abs(delta - (1.0 - 0.3)) < 1e-12   # target = r only
    assert agent.q[15].sum() == 0.0           # never bootstrapped into goal


def test_edge_belief_symmetry_and_bounded_update():
    e = EdgeBeliefModel(3, 3, b0=0.6, rho=0.3)
    e.set_b(((0, 0), (0, 1)), 0.8)
    assert e.b(((0, 1), (0, 0))) == e.b(((0, 0), (0, 1)))  # canonical storage
    # rho=0: observed move -> belief increases; observed stay -> decreases
    e2 = EdgeBeliefModel(3, 3, b0=0.6, rho=0.0)
    b_before = e2.b(((0, 0), (0, 1)))
    e2.update(0, 1, 1, t=0)   # commanded E, moved E: edge confirmed open
    assert e2.b(((0, 0), (0, 1))) > b_before
    e3 = EdgeBeliefModel(3, 3, b0=0.6, rho=0.0)
    b_before3 = e3.b(((0, 0), (0, 1)))
    e3.update(0, 1, 0, t=0)   # commanded E, stayed: evidence of blockage
    assert e3.b(((0, 0), (0, 1))) < b_before3


def test_edge_belief_mixture_no_collapse_under_slip():
    """A single self-loop at rho=0.3 must NOT collapse b to ~0 (ambiguous)."""
    e = EdgeBeliefModel(3, 3, b0=0.9, rho=0.3)
    e.update(0, 1, 0, t=0)  # one stay
    assert e.b(((0, 0), (0, 1))) > 0.5


def test_edge_predict_mixture_sums_to_one():
    e = EdgeBeliefModel(3, 3, b0=0.6, rho=0.3)
    for s in range(9):
        for a in range(4):
            out = e.predict_mixture(s, a)
            assert abs(sum(out.values()) - 1.0) < 1e-12


def test_local_reset_keeps_latest_observation():
    m = CategoricalDirichlet(9, 3, beta=0.5)
    from cavr.agents.stochastic_dyna import StochasticDynaBase
    from cavr.models.episodic_map import EpisodicMap
    agent = StochasticDynaBase.__new__(StochasticDynaBase)
    agent.model = m
    agent.memory = EpisodicMap(capacity_per_sa=4)
    agent.gamma = 0.97
    agent.q = np.zeros((9, 4))
    agent.goal_idx = 8
    agent.alpha_model = 0.1
    # history: (4,0)->4 (stay) observed 5 times (old belief: blocked)
    for t in range(5):
        m.update(4, 0, 4, -0.03)
        agent.memory.append(4, 0, -0.03, 4, t)
    # NEW contradictory observation: move W (s2=3) at t=100 (in memory only)
    agent.memory.append(4, 0, -0.01, 3, 100)
    agent.forget_region({4}, cur_t=100)
    # after reset, the latest OLD record (stay, t<100) is kept; the new
    # observation is incorporated exactly once by the caller (observe loop)
    p_after_reset = m.predict(4, 0).copy()
    m.reobserve(4, 0, 3, -0.01)  # incorporate the new observation ONCE
    p_final = m.predict(4, 0)
    k_new = category_of(4, 3, 3)
    k_old = category_of(4, 4, 3)
    assert p_final[k_new] > 0.2            # new evidence present
    assert p_final[k_new] > p_after_reset[k_new]
    # no double counting: counts = prior + 1 new (the latest-old for THIS pair
    # is the new record itself, so nothing older is re-added for (4,0))
    total = m.counts[4, 0].sum()
    assert abs(total - (5 * 0.5 + 1.0)) < 1e-9
    # older pairs in the region keep their latest pre-alarm record
    # (different (s,a): old evidence must survive the reset)
    m2 = CategoricalDirichlet(9, 3, beta=0.5)
    mem2 = EpisodicMap(capacity_per_sa=4)
    for t in range(4):
        m2.update(3, 2, 6, -0.01)
        mem2.append(3, 2, -0.01, 6, t)
    agent2 = StochasticDynaBase.__new__(StochasticDynaBase)
    agent2.model = m2
    agent2.memory = mem2
    StochasticDynaBase.forget_region(agent2, {3, 4}, cur_t=100)
    assert m2.n_observations(3, 2) == 1.0  # latest old record retained
