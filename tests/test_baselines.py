"""Baseline behavior tests (spec: baseline-first mandate)."""
import numpy as np

from cavr.agents import make_agent
from cavr.envs.grid_env import GridEnv
from cavr.envs.maze_gen import maze_open_set
from cavr.utils.seeding import run_seed, make_rng


def _run_agent(env, agent, steps, start=(0, 0), record_eps=True):
    s = env.state_idx(env.reset(start=start))
    ep_records = []
    cur_ret = 0.0
    cur_steps = 0
    for t in range(steps):
        a = agent.act(s, t)
        s2c, r, term, trunc, _ = env.step(a)
        s2 = env.state_idx(s2c)
        agent.observe(s, a, r, s2, term, t)
        cur_ret += r
        cur_steps += 1
        if term or trunc:
            ep_records.append({"end_step": t, "success": int(term),
                               "steps": cur_steps, "ret": cur_ret})
            cur_ret, cur_steps = 0.0, 0
            s = env.state_idx(env.reset())
        else:
            s = s2
    return ep_records


def _success_after(records, t0):
    sel = [e for e in records if e["end_step"] >= t0]
    if not sel:
        return 0.0
    return float(np.mean([e["success"] for e in sel]))


def test_dyna_q_learns_static_maze(maze, agent_cfg):
    env = GridEnv(maze, (5, 8), "static", 0.0, 150, 10**9, env_seed=1)
    agent = make_agent("dyna_q", env.n_states, env.W, env.H, env.state_idx((5, 8)),
                       agent_cfg, make_rng(run_seed(1, "a")))
    recs = _run_agent(env, agent, 1500)
    assert len(recs) > 4
    assert _success_after(recs, recs[len(recs) // 2]["end_step"]) > 0.5


def test_dyna_q_plus_finds_shortcut_faster(maze, agent_cfg):
    """Blocking scenario: Dyna-Q+ (age bonus) should re-solve the maze no
    slower than vanilla Dyna-Q after the path is blocked (qualitative)."""
    results = {}
    for name in ("dyna_q", "dyna_q_plus"):
        delays = []
        for sd in (3, 4, 5):
            env = GridEnv(maze, (5, 8), "blocking", 0.0, 150, 700,
                          env_seed=run_seed(sd, "env"))
            agent = make_agent(name, env.n_states, env.W, env.H,
                               env.state_idx((5, 8)), agent_cfg,
                               make_rng(run_seed(sd, name)))
            recs = _run_agent(env, agent, 1400)
            post = [e for e in recs if e["end_step"] >= 700]
            d = next((e["end_step"] - 700 for e in post if e["success"]), 700)
            delays.append(d)
        results[name] = float(np.mean(delays))
    assert results["dyna_q_plus"] <= results["dyna_q"] + 50  # not slower (tolerance)


def test_prioritized_sweeping_uses_queue(maze, agent_cfg):
    env = GridEnv(maze, (5, 8), "static", 0.0, 150, 10**9, env_seed=1)
    agent = make_agent("prioritized_sweeping", env.n_states, env.W, env.H,
                       env.state_idx((5, 8)), agent_cfg, make_rng(run_seed(2, "a")))
    recs = _run_agent(env, agent, 1200)
    assert agent.totals["backups"] > 0 and agent.totals["queue_ops"] > 0
    assert _success_after(recs, 800) > 0.5


def test_dstar_lite_solves_known_map_and_replans(maze, agent_cfg):
    # (a) known map: D* Lite reaches the goal along a shortest path
    env = GridEnv(maze, (5, 8), "static", 0.0, 150, 10**9, env_seed=1)
    agent = make_agent("dstar_lite", env.n_states, env.W, env.H,
                       env.state_idx((5, 8)), agent_cfg, make_rng(run_seed(1, "a")))
    agent.seed_map(maze_open_set(maze))
    recs = _run_agent(env, agent, 300)
    assert recs and recs[0]["success"] == 1 and recs[0]["steps"] <= 16
    # (b) unknown map with a block mid-run: replans and still solves
    env2 = GridEnv(maze, (5, 8), "blocking", 0.0, 150, 40, env_seed=2)
    agent2 = make_agent("dstar_lite", env2.n_states, env2.W, env2.H,
                        env2.state_idx((5, 8)), agent_cfg, make_rng(run_seed(2, "a")))
    recs2 = _run_agent(env2, agent2, 600)
    succ = [e for e in recs2 if e["success"]]
    assert succ  # eventually solves despite the unseen block


def test_identical_observation_access(maze, agent_cfg):
    """All methods see the same public observation schema (types, not values)."""
    schema = None
    for name in ("q_learning", "dyna_q", "dyna_q_plus", "prioritized_sweeping",
                 "dirichlet_dyna", "local_forgetting", "dstar_lite", "cavr"):
        env = GridEnv(maze, (5, 8), "static", 0.1, 100, 10**9,
                      env_seed=run_seed(5, "env"))
        agent = make_agent(name, env.n_states, env.W, env.H, env.state_idx((5, 8)),
                           agent_cfg, make_rng(run_seed(5, name)))
        s = env.state_idx(env.reset())
        for t in range(20):
            a = agent.act(s, t)
            s2c, r, term, trunc, info = env.step(a)
            s2 = env.state_idx(s2c)
            stats = agent.observe(s, a, r, s2, term, t)
            sig = (type(s2c).__name__, type(r).__name__, type(term).__name__,
                   type(trunc).__name__, type(info).__name__,
                   tuple(sorted(stats.keys())))
            if schema is None:
                schema = sig
            else:
                assert sig == schema, f"{name} observation schema mismatch"
            if term or trunc:
                s = env.state_idx(env.reset())
            else:
                s = s2


def test_dirichlet_and_local_forgetting_learn(maze, agent_cfg):
    for name in ("dirichlet_dyna", "local_forgetting"):
        env = GridEnv(maze, (5, 8), "static", 0.1, 150, 10**9, env_seed=1)
        agent = make_agent(name, env.n_states, env.W, env.H, env.state_idx((5, 8)),
                           agent_cfg, make_rng(run_seed(6, name)))
        recs = _run_agent(env, agent, 1500)
        assert _success_after(recs, 1000) > 0.4, name
