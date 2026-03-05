"""Random-move baseline agent."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.env.gomoku_env import GomokuEnv


class RandomAgent:
    """Selects uniformly random legal moves."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: GomokuEnv) -> int:
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("No legal actions available.")
        return int(self.rng.choice(legal))
