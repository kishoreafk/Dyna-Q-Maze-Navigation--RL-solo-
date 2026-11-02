"""Environment contract tests (spec 10)."""
import numpy as np

from cavr.envs.grid_env import GridEnv, STEP_REWARD, COLLISION_REWARD, GOAL_REWARD
from cavr.envs.maze_gen import generate_maze, edge_key, neighbor_cell, maze_open_set


def test_static_transition_deterministic(maze):
    env = GridEnv(maze, (5, 8), "static", rho=0.0, max_steps=100, t_change=10**9,
                  env_seed=3, start=(0, 0))
    open_set = maze_open_set(maze)
    env.reset(start=(1, 1))
    s2, r, term, trunc, info = env.step(2)  # S
    # with rho=0 the executed direction equals the commanded one
    nb = neighbor_cell((1, 1), 2)
    if edge_key((1, 1), nb) in open_set:
        assert s2 == nb and r == STEP_REWARD
    else:
        assert s2 == (1, 1) and r == STEP_REWARD + COLLISION_REWARD


def test_illegal_edge_causes_self_loop(maze):
    # find a blocked internal edge away from the goal and push against it
    open_set = maze_open_set(maze)
    goal = (5, 8)
    blocked = None
    for r in range(6):
        for c in range(9):
            for a in range(4):
                nb = neighbor_cell((r, c), a)
                if nb is None:
                    continue
                e = edge_key((r, c), nb)
                if e not in open_set and (r, c) != goal and nb != goal:
                    blocked = ((r, c), a, nb)
                    break
        if blocked:
            break
    assert blocked is not None
    (cell, a, _nb) = blocked
    env = GridEnv(maze, (5, 8), "static", rho=0.0, max_steps=100, t_change=10**9,
                  env_seed=5, start=cell)
    env.reset(start=cell)
    s2, r, term, trunc, _ = env.step(a)
    assert s2 == cell and r == STEP_REWARD + COLLISION_REWARD and not term and not trunc


def test_goal_reward_terminal(maze):
    from cavr.envs.maze_gen import shortest_path
    goal = (5, 8)
    open_set = maze_open_set(maze)
    path = shortest_path(open_set, 6, 9, (0, 0), goal)
    assert len(path) >= 2
    prev = path[-2]
    a = [(0, 1), (1, 0)].index((goal[0] - prev[0], goal[1] - prev[1])) + 1  # S or E
    env = GridEnv(maze, goal, "static", rho=0.0, max_steps=100, t_change=10**9,
                  env_seed=7, start=prev)
    env.reset(start=prev)
    s2, r, term, trunc, _ = env.step(a)
    assert s2 == goal and term and r == GOAL_REWARD and not trunc


def test_truncation_flag(maze):
    env = GridEnv(maze, (5, 8), "static", rho=0.0, max_steps=3, t_change=10**9,
                  env_seed=9, start=(0, 0))
    env.reset(start=(0, 0))
    term = False
    for _ in range(3):
        s2, r, term, trunc, _ = env.step(0)
    assert trunc and not term


def test_slip_distribution_monte_carlo(maze):
    """Frequencies match the analytical action-noise formula. Layout: 3x3
    fully open EXCEPT the south edge of (1,1). Commanding N from (1,1):
    P(N)=0.7, P(E)=0.1, P(W)=0.1, P(stay)=0.1 (slipped S, blocked), P(S)=0."""
    full = {"H": 3, "W": 3, "seed": 0, "braid": 1.0, "open": []}
    edges = []
    for r in range(3):
        for c in range(3):
            for nb in [neighbor_cell((r, c), 1), neighbor_cell((r, c), 2)]:
                if nb and 0 <= nb[0] < 3 and 0 <= nb[1] < 3:
                    e = edge_key((r, c), nb)
                    if e not in edges and e != edge_key((1, 1), (2, 1)):
                        edges.append(e)
    full["open"] = [list(e) for e in edges]
    env = GridEnv(full, (2, 2), "static", rho=0.3, max_steps=10**6, t_change=10**9,
                  env_seed=11, start=(1, 1))
    counts = np.zeros(5)
    N = 40000
    for _ in range(N):
        env.reset(start=(1, 1))  # always command from the same cell
        s2, *_ = env.step(0)  # N
        if s2 == (1, 1):
            counts[4] += 1
        elif s2 == (0, 1):
            counts[0] += 1
        elif s2 == (1, 2):
            counts[1] += 1
        elif s2 == (2, 1):
            counts[2] += 1
        elif s2 == (1, 0):
            counts[3] += 1
    freq = counts / N
    assert abs(freq[0] - 0.7) < 0.02
    for k in (1, 3, 4):
        assert abs(freq[k] - 0.1) < 0.02, f"freq[{k}]={freq[k]}"
    assert freq[2] < 0.001


def test_change_schedule_applies_at_t(maze):
    env = GridEnv(maze, (5, 8), "blocking", rho=0.0, max_steps=100, t_change=5,
                  env_seed=13, start=(0, 0))
    before = set(map(lambda e: (tuple(e[0]), tuple(e[1])),
                     env.evaluator_state()["open_edges"]))
    env.reset(start=(0, 0))
    for _ in range(5):
        env.step(0)
    after = set(map(lambda e: (tuple(e[0]), tuple(e[1])),
                    env.evaluator_state()["open_edges"]))
    assert len(before - after) == 1  # exactly one edge closed
    events = env.evaluator_state()["events"]
    assert len(events) == 1 and events[0]["t"] == 5 and events[0]["kind"] == "blocking"
