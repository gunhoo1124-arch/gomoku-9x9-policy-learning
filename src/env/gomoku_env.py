"""9x9 Gomoku environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rules import board_full, check_five_in_a_row


@dataclass
class StepResult:
    """Typed step output container."""

    board: np.ndarray
    reward: float
    done: bool
    info: dict


class GomokuEnv:
    """
    A minimal Gomoku environment for 9x9 play.

    Board encoding:
    - `0`: empty
    - `1`: black
    - `-1`: white
    """

    def __init__(self, board_size: int = 9, win_length: int = 5) -> None:
        if board_size != 9:
            raise ValueError("This project expects a fixed board size of 9.")
        if win_length < 5:
            raise ValueError("Win length must be at least 5 for Gomoku.")

        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.winner: Optional[int] = None
        self.last_action: Optional[int] = None

    def reset(self) -> np.ndarray:
        """Reset the board and return a copy of the initial state."""
        self.board.fill(0)
        self.current_player = 1
        self.winner = None
        self.last_action = None
        return self.board.copy()

    def copy(self) -> "GomokuEnv":
        """Return a deep copy of the environment state."""
        clone_env = GomokuEnv(board_size=self.board_size, win_length=self.win_length)
        clone_env.board = self.board.copy()
        clone_env.current_player = self.current_player
        clone_env.winner = self.winner
        clone_env.last_action = self.last_action
        return clone_env

    def clone(self) -> "GomokuEnv":
        """Alias for `copy()`."""
        return self.copy()

    def action_to_rc(self, action: int) -> tuple[int, int]:
        """Convert action index `[0, 80]` to `(row, col)`."""
        if not isinstance(action, (int, np.integer)):
            raise TypeError(f"Action must be an integer, got {type(action)!r}.")

        action_int = int(action)
        max_action = self.board_size * self.board_size
        if action_int < 0 or action_int >= max_action:
            raise ValueError(
                f"Action must be in [0, {max_action - 1}], got {action_int}."
            )
        return divmod(action_int, self.board_size)

    def rc_to_action(self, row: int, col: int) -> int:
        """Convert `(row, col)` to action index."""
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(
                f"Row/col out of range for {self.board_size}x{self.board_size}: "
                f"({row}, {col})."
            )
        return row * self.board_size + col

    def legal_actions(self) -> list[int]:
        """Return all currently legal action indices."""
        flat = self.board.ravel()
        return np.flatnonzero(flat == 0).astype(np.int64).tolist()

    def is_terminal(self) -> bool:
        """Return `True` when game is won or drawn."""
        return self.winner is not None

    def outcome_for_player(self, player: int) -> float:
        """
        Terminal value from a specific player's perspective.

        Returns:
        - `+1.0` if `player` won
        - `-1.0` if `player` lost
        - `0.0` for draw or non-terminal
        """
        if self.winner is None or self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply one action and return `(board, reward, done, info)`.

        Reward is from the player-who-just-moved perspective:
        - win: `+1.0`
        - draw/non-terminal: `0.0`
        """
        if self.is_terminal():
            raise RuntimeError("Cannot call step(): game is already terminal.")

        row, col = self.action_to_rc(action)
        if self.board[row, col] != 0:
            raise ValueError(
                f"Illegal move at ({row}, {col}): cell is already occupied."
            )

        player = self.current_player
        self.board[row, col] = player
        self.last_action = int(action)

        reward = 0.0
        done = False
        info: dict[str, object] = {
            "player": player,
            "row": row,
            "col": col,
            "action": int(action),
        }

        if check_five_in_a_row(self.board, row, col, player, self.win_length):
            self.winner = player
            reward = 1.0
            done = True
            info["result"] = "win"
            info["winner"] = player
            info["next_player_reward"] = -1.0
        elif board_full(self.board):
            self.winner = 0
            reward = 0.0
            done = True
            info["result"] = "draw"
            info["next_player_reward"] = 0.0
        else:
            self.current_player = -player

        return self.board.copy(), reward, done, info

    def board_to_string(self) -> str:
        """Build a human-readable board string."""
        symbols = {1: "X", -1: "O", 0: "."}
        header = "   " + " ".join(str(i) for i in range(self.board_size))
        rows: list[str] = [header]
        for r in range(self.board_size):
            cells = " ".join(symbols[int(v)] for v in self.board[r])
            rows.append(f"{r:2d} {cells}")
        return "\n".join(rows)

    def render(self) -> None:
        """Print the board to stdout."""
        print(self.board_to_string())
