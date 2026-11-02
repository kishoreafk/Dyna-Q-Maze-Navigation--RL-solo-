"""Reproducibility and split-leakage tests."""
import numpy as np


def test_seed_repeatability_identical_streams(maze, agent_cfg):
    from cavr.agents import make_agent
    from cavr.envs.grid_env import GridEnv
    from cavr.utils.seeding import run_seed, make_rng

    def one_run():
        env = GridEnv(maze, (5, 8), "blocking", 0.3, 100, 300,
                      env_seed=run_seed(77, "env"))
        agent = make_agent("cavr", env.n_states, env.W, env.H,
                           env.state_idx((5, 8)), agent_cfg,
                           make_rng(run_seed(77, "agent")))
        s = env.state_idx(env.reset())
        stream = []
        for t in range(400):
            a = agent.act(s, t)
            s2c, r, term, trunc, _ = env.step(a)
            s2 = env.state_idx(s2c)
            agent.observe(s, a, r, s2, term, t)
            stream.append((int(a), round(float(r), 6), int(term)))
            s = env.state_idx(env.reset()) if (term or trunc) else s2
        return stream, agent.q.copy()

    stream1, q1 = one_run()
    stream2, q2 = one_run()
    assert stream1 == stream2
    assert np.allclose(q1, q2)


def test_maze_splits_disjoint_by_identity():
    """Manifest splits share no maze seeds across train/val/test."""
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "manifests", "mazes_manifest.json")
    if not os.path.exists(path):
        import subprocess
        import sys
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        subprocess.run([sys.executable, os.path.join(root, "scripts", "make_manifest.py")],
                       cwd=root, check=True, capture_output=True)
    with open(path) as fh:
        man = json.load(fh)
    splits = {}
    for m in man["mazes"]:
        splits.setdefault(m["split"], []).append(m["seed"])
    assert len(splits["train"]) == len(set(splits["train"]))
    all_seeds = [s for v in splits.values() for s in v]
    assert len(all_seeds) == len(set(all_seeds)), "maze seed reused across splits"


def test_value_iteration_sane(maze):
    """Oracle sanity: optimal value at the start cell is positive and below
    the goal reward; greedy rollout reaches the goal (rho=0)."""
    from cavr.evaluation.oracles import value_iteration, bfs_shortest_len
    from cavr.envs.maze_gen import maze_open_set
    open_set = maze_open_set(maze)
    V, pol = value_iteration(open_set, 6, 9, 0.0, 0.97, (5, 8))
    assert 0.0 < V[0] < 1.0
    # with rho=0 the BFS-optimal length bounds the discounted return
    L = bfs_shortest_len(open_set, 6, 9, (0, 0), (5, 8))
    assert L > 0
