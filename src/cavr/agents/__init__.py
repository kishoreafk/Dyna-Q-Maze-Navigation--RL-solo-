from .base import BaseAgent
from .q_learning import QLearningAgent
from .dyna_q import DynaQAgent
from .dyna_q_plus import DynaQPlusAgent
from .prioritized_sweeping import PrioritizedSweepingAgent
from .stochastic_dyna import DirichletDynaAgent, LocalForgettingAgent
from .dstar_lite import DStarLiteAgent
from .cavr_dyna import CAVRDynaAgent


def make_agent(name: str, n_states: int, W: int, H: int, goal_idx: int,
               cfg: dict, rng, rho_est: float = 0.1) -> BaseAgent:
    if name == "q_learning":
        return QLearningAgent(n_states, W, H, goal_idx, cfg, rng)
    if name == "dyna_q":
        return DynaQAgent(n_states, W, H, goal_idx, cfg, rng)
    if name == "dyna_q_plus":
        return DynaQPlusAgent(n_states, W, H, goal_idx, cfg, rng)
    if name == "prioritized_sweeping":
        return PrioritizedSweepingAgent(n_states, W, H, goal_idx, cfg, rng)
    if name == "dirichlet_dyna":
        return DirichletDynaAgent(n_states, W, H, goal_idx, cfg, rng)
    if name == "local_forgetting":
        return LocalForgettingAgent(n_states, W, H, goal_idx, cfg, rng, rho_est=rho_est)
    if name == "dstar_lite":
        return DStarLiteAgent(n_states, W, H, goal_idx, cfg, rng, rho_est=rho_est)
    if name == "cavr":
        return CAVRDynaAgent(n_states, W, H, goal_idx, cfg, rng, rho_est=rho_est)
    if name == "cavr_nodetector":
        return CAVRDynaAgent(n_states, W, H, goal_idx, cfg, rng, rho_est=rho_est,
                             detect_changes=False)
    if name == "cavr_fixedB":
        return CAVRDynaAgent(n_states, W, H, goal_idx, cfg, rng, rho_est=rho_est,
                             adaptive_budget=False)
    raise ValueError(f"unknown method {name}")


ALL_METHODS = ["q_learning", "dyna_q", "dyna_q_plus", "prioritized_sweeping",
               "dirichlet_dyna", "local_forgetting", "dstar_lite",
               "cavr", "cavr_nodetector", "cavr_fixedB"]

__all__ = ["BaseAgent", "QLearningAgent", "DynaQAgent", "DynaQPlusAgent",
           "PrioritizedSweepingAgent", "DirichletDynaAgent", "LocalForgettingAgent",
           "CAVRDynaAgent", "make_agent", "ALL_METHODS"]
