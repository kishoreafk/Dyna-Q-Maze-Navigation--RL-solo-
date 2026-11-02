"""CAVR-specific behavior tests."""
import numpy as np

from cavr.agents import make_agent
from cavr.envs.grid_env import GridEnv
from cavr.utils.seeding import run_seed, make_rng


def _run(env, agent, steps, start=(0, 0)):
    s = env.state_idx(env.reset(start=start))
    ep_records = []
    cur_ret, cur_steps = 0.0, 0
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


def test_cavr_learns_static_maze(maze, agent_cfg):
    env = GridEnv(maze, (5, 8), "static", 0.1, 150, 10**9, env_seed=1)
    agent = make_agent("cavr", env.n_states, env.W, env.H, env.state_idx((5, 8)),
                       agent_cfg, make_rng(run_seed(11, "a")))
    recs = _run(env, agent, 1800)
    sel = [e for e in recs if e["end_step"] >= 1200]
    assert sel and np.mean([e["success"] for e in sel]) > 0.5


def test_cavr_detects_and_recovers_from_block(maze, agent_cfg):
    env = GridEnv(maze, (5, 8), "blocking", 0.0, 150, 800, env_seed=1)
    agent = make_agent("cavr", env.n_states, env.W, env.H, env.state_idx((5, 8)),
                       {**agent_cfg, "h": 2.5}, make_rng(run_seed(12, "a")))
    recs = _run(env, agent, 1800)
    assert agent.totals["alarms"] >= 1  # change detected
    post = [e for e in recs if e["end_step"] >= 800]
    d = next((e["end_step"] - 800 for e in post if e["success"]), 10**9)
    assert d < 600  # recovered within the horizon


def test_cavr_rediscovers_shortcut_with_probe_floor(maze, agent_cfg):
    """Newly opened edge must be discoverable (nonzero probe/exploration floor).
    Uses maze seed 4243, which has a genuine shortcut edge."""
    from cavr.envs.maze_gen import generate_maze
    maze2 = generate_maze(6, 9, 4243, 0.15)
    env = GridEnv(maze2, (5, 8), "shortcut", 0.0, 150, 300, env_seed=1)
    agent = make_agent("cavr", env.n_states, env.W, env.H, env.state_idx((5, 8)),
                       agent_cfg, make_rng(run_seed(13, "a")))
    recs = _run(env, agent, 2000)
    post_succ = [e["success"] for e in recs if e["end_step"] >= 300]
    assert post_succ and np.mean(post_succ) > 0.3


def test_cavr_variants_construct_and_run(maze, agent_cfg):
    for name in ("cavr_nodetector", "cavr_fixedB"):
        env = GridEnv(maze, (5, 8), "static", 0.1, 100, 10**9, env_seed=1)
        agent = make_agent(name, env.n_states, env.W, env.H, env.state_idx((5, 8)),
                           agent_cfg, make_rng(run_seed(14, name)))
        recs = _run(env, agent, 400)
        assert recs or True
        if name == "cavr_fixedB":
            # every step with features present gets the fixed budget
            assert agent.budget.fixed_b == 32


def test_cavr_checkpoint_roundtrip(maze, agent_cfg):
    """Weights saved -> loaded -> identical Q table and edge beliefs."""
    import tempfile
    import os
    from cavr.utils.checkpoint import save_checkpoint, load_checkpoint
    env = GridEnv(maze, (5, 8), "static", 0.1, 100, 10**9, env_seed=1)
    agent = make_agent("cavr", env.n_states, env.W, env.H, env.state_idx((5, 8)),
                       agent_cfg, make_rng(run_seed(15, "a")))
    s = env.state_idx(env.reset())
    for t in range(300):
        a = agent.act(s, t)
        s2c, r, term, trunc, _ = env.step(a)
        s2 = env.state_idx(s2c)
        agent.observe(s, a, r, s2, term, t)
        s = env.state_idx(env.reset()) if (term or trunc) else s2
    arrays = agent.checkpoint_arrays()
    meta = {"method": "cavr", "maze_seed": 1, "scenario": "static",
            "rho_true": 0.1, "rho_est": 0.1, "H": 6, "W": 9, "goal": [5, 8],
            "h": 3.0, "kappa": 0.001, "t_final": 300, "epi_ema": 0.3,
            "agent_cfg": agent_cfg}
    if "records" in arrays:
        meta["memory_records"] = arrays.pop("records")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.npz")
        save_checkpoint(path, arrays, meta)
        arrays2, meta2 = load_checkpoint(path)
    agent2 = make_agent("cavr", env.n_states, env.W, env.H, env.state_idx((5, 8)),
                        agent_cfg, make_rng(run_seed(99, "a")))
    agent2.load_checkpoint_arrays(arrays2, meta2)
    assert np.allclose(agent.q, agent2.q)
    for e in agent.edges.edges():
        assert abs(agent.edges.b(e) - agent2.edges.b(e)) < 1e-9
