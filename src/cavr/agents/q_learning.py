"""Tabular Q-learning (baseline 1)."""
from __future__ import annotations

from .base import BaseAgent


class QLearningAgent(BaseAgent):
    name = "q_learning"
