"""Leakage guards (spec: no true-map/time/slip leakage)."""
import numpy as np
import pytest

from cavr.agents import make_agent, ALL_METHODS
from cavr.envs.grid_env import GridEnv, LeakageError
from cavr.utils.seeding import run_seed, make_rng


def test_private_attrs_raise_when_guarded(maze):
    env = GridEnv(maze, (5, 8), "blocking", 0.3, 100, 50, env_seed=1)
    env.enable_leak_guard()
    for attr in ("_open", "_schedule", "_rho", "_rng"):
        with pytest.raises(LeakageError):
            getattr(env, attr)
    # environment keeps stepping correctly under guard
    env.reset(start=(0, 0))
    s2, r, term, trunc, _ = env.step(0)
    assert isinstance(s2, tuple)


def test_all_agents_run_clean_under_leak_guard(maze, agent_cfg):
    """Architectural guarantee: agents never receive the env object; this
    test additionally runs every method with the guard enabled so any
    accidental private access fails loudly."""
    for name in ALL_METHODS:
        env = GridEnv(maze, (5, 8), "blocking", 0.3, 100, 60, env_seed=1)
        env.enable_leak_guard()
        agent = make_agent(name, env.n_states, env.W, env.H, env.state_idx((5, 8)),
                           agent_cfg, make_rng(run_seed(1, name)))
        s = env.state_idx(env.reset())
        for t in range(80):
            a = agent.act(s, t)
            s2c, r, term, trunc, _ = env.step(a)
            s2 = env.state_idx(s2c)
            agent.observe(s, a, r, s2, term, t)
            if term or trunc:
                s = env.state_idx(env.reset())
            else:
                s = s2
        # change still applied despite the guard
        assert len(env.evaluator_state()["events"]) >= 1


def test_agent_source_does_not_reference_env_private_fields():
    import inspect
    import cavr.agents as agents_pkg
    import os
    src_dir = os.path.dirname(inspect.getfile(agents_pkg))
    banned = ["_open", "_schedule", "_rho", "_rng", "true_edges", "change_step",
              "slipped_direction", "evaluator_state"]
    for fn in os.listdir(src_dir):
        if not fn.endswith(".py"):
            continue
        with open(os.path.join(src_dir, fn)) as fh:
            src = fh.read()
        for b in banned:
            assert b not in src, f"agent source {fn} references banned symbol {b}"
