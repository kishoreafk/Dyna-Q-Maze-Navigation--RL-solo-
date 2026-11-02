import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cavr.envs.maze_gen import generate_maze, maze_open_set
from cavr.envs.grid_env import GridEnv

MAZE_SEED = 4242
H, W = 6, 9


@pytest.fixture()
def maze():
    return generate_maze(H, W, MAZE_SEED, 0.15)


@pytest.fixture()
def env(maze):
    return GridEnv(maze, (H - 1, W - 1), "static", 0.0, 120, 10**9,
                   env_seed=1, start=(0, 0))


@pytest.fixture()
def agent_cfg():
    return {
        "alpha": 0.1, "alpha_model": 0.15, "gamma": 0.97, "eps": 0.1,
        "beta": 0.5, "b0": 0.6, "edge_lr": 0.4, "rho_est": 0.1,
        "dyna_n": 10, "ps_n": 10, "ps_theta": 1e-3, "lf_n": 16, "cap": 64,
        "theta_useful": 0.01, "theta_push": 0.05, "theta_probe": 0.12,
        "probe_min_gap": 8, "budget_theta_high": 0.15, "budget_theta_lo": 0.10,
        "drift": 0.01, "cooldown": 15, "kappa": 0.001, "h": 3.0,
        "dstar_retest_interval": 50,
    }
