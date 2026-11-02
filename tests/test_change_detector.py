"""Change detector tests: pre-update surprise, false-alarm budget, real change."""
import numpy as np

from cavr.models.change_detector import ChangeDetector
from cavr.models.categorical_dirichlet import CategoricalDirichlet, category_of


def test_surprise_uses_pre_update_prediction():
    d = ChangeDetector(h=3.0)
    p_before = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
    z1, g1 = d.score_pre_update(p_before.copy(), 0)
    # expected surprise under the model = entropy of p_before
    mu_expected = float(-(p_before * np.log(np.clip(p_before, 1e-12, None))).sum())
    assert abs(z1 - (-np.log(0.1))) < 1e-9
    assert abs(g1 - (z1 - mu_expected - d.drift)) < 1e-9
    # the score depends only on the frozen values: overwrite and re-check
    p2 = p_before.copy()
    p2[0] = 0.5
    z2, _ = d.score_pre_update(p2, 0)
    assert abs(z2 - (-np.log(0.5))) < 1e-9


def test_cusum_alarms_on_sequential_surprise():
    d = ChangeDetector(h=2.0, drift=0.0)
    p = np.array([0.05, 0.05, 0.05, 0.05, 0.8])  # model says "stay likely"
    fired = False
    for t in range(50):
        z, g = d.score_pre_update(p, 1)  # reality keeps contradicting
        if d.confirm(g, t):
            fired = True
            break
    assert fired


def test_stationary_slip_false_alarm_budget(maze, agent_cfg):
    """G1-style stress: stationary rho=0.3 must produce few alarms with the
    CALIBRATED detector (pair-level CUSUM, h=5.0; measured FAR~0.7/1000,
    hard-block detection lag ~2 steps)."""
    from cavr.agents import make_agent
    from cavr.envs.grid_env import GridEnv
    from cavr.utils.seeding import run_seed, make_rng
    total, alarms = 0, 0
    for seed in (701, 702):
        env = GridEnv(maze, (5, 8), "static", 0.3, 200, 10**9,
                      env_seed=run_seed(seed, "env"))
        agent = make_agent("local_forgetting", env.n_states, env.W, env.H,
                           env.state_idx((5, 8)), {**agent_cfg, "h": 5.0},
                           make_rng(run_seed(seed, "agent")), rho_est=0.3)
        agent.do_reset = False
        s = env.state_idx(env.reset())
        for t in range(1500):
            a = agent.act(s, t)
            s2c, r, term, trunc, _ = env.step(a)
            s2 = env.state_idx(s2c)
            stats = agent.observe(s, a, r, s2, term, t)
            alarms += int(stats.get("alarm", False))
            if term or trunc:
                s = env.state_idx(env.reset())
            else:
                s = s2
        total += 1500
    assert alarms / total * 1000.0 <= 5.0


def test_detector_fires_on_hard_block(maze, agent_cfg):
    """Under rho=0 a hard block quickly produces a reliable alarm."""
    from cavr.agents import make_agent
    from cavr.envs.grid_env import GridEnv
    from cavr.utils.seeding import run_seed, make_rng
    from cavr.envs.maze_gen import edge_key, neighbor_cell
    # find an edge on a well-traveled corridor: use commanded moves along an open path
    env = GridEnv(maze, (5, 8), "static", 0.0, 200, 10**9, env_seed=1)
    agent = make_agent("local_forgetting", env.n_states, env.W, env.H,
                       env.state_idx((5, 8)), {**agent_cfg, "h": 3.0},
                       make_rng(run_seed(9, "agent")), rho_est=0.0)
    # train the model on a fixed shuttle route (0,0)<->(0,1) for 200 steps
    s = env.state_idx(env.reset(start=(0, 0)))
    toggle = 1
    alarms = 0
    for t in range(300):
        # shuttle E/W; at t=150 the env closes the shuttle edge (change)
        if t == 150:
            env._state["open"].discard(edge_key((0, 0), (0, 1)))
        a = 1 if toggle else 3
        s2c, r, term, trunc, _ = env.step(a)
        s2 = env.state_idx(s2c)
        stats = agent.observe(s, a, r, s2, term, t)
        alarms += int(stats.get("alarm", False))
        if s2c == (0, 1) and toggle:
            toggle = 0
        elif s2c == (0, 0) and not toggle:
            toggle = 1
        s = s2
    assert alarms >= 1  # the block was detected
