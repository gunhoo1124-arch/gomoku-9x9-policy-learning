"""Utilities for logging played games into supervised training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from src.utils.encoding import encode_board


LogPolicy = Literal["human", "all"]


@dataclass
class GameDataLogger:
    """
    Collect and persist (state, action) samples in training-compatible `.npz` format.

    Saved keys:
    - `states`: float32 array with shape `(N, 3, 9, 9)`
    - `actions`: int64 array with shape `(N,)`
    - `current_players`: int8 array with values `{-1, 1}` and shape `(N,)`
    - `outcomes`: int8 array with values `{1, 0, -1}` and shape `(N,)`
    """

    log_policy: LogPolicy = "human"
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    current_players: list[int] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)
    _active_game_starts: list[int] = field(default_factory=list, init=False)

    def start_game(self) -> None:
        """Mark the beginning of a new logged game."""
        self._active_game_starts.append(len(self.actions))

    def should_log(self, actor: str) -> bool:
        if self.log_policy == "all":
            return True
        return actor == "human"

    def record(self, board: np.ndarray, current_player: int, action: int, actor: str) -> None:
        """
        Record one supervised sample if policy allows this actor type.

        Args:
            board: Raw board at time t (before action is applied).
            current_player: Side to move at time t.
            action: Chosen action label in `[0, 80]`.
            actor: Move source label (e.g. `human`, `heuristic`, `model`).
        """
        if not self.should_log(actor):
            return

        if not self._active_game_starts:
            # Backward-compatible safeguard: if start_game() was not called, treat all
            # collected samples as one active game.
            self._active_game_starts.append(0)

        self.states.append(encode_board(board, current_player))
        self.actions.append(int(action))
        self.current_players.append(int(np.sign(current_player)) or 0)
        self.outcomes.append(0)

    def finalize_game(self, winner: int | None) -> None:
        """
        Assign outcome labels to all moves in the most recently started active game.

        `winner` follows env convention:
        - `1`: black win
        - `-1`: white win
        - `0`: draw
        - `None`: unknown/incomplete game (defaults to 0)
        """
        if not self._active_game_starts:
            return

        if winner is not None and winner not in (1, -1, 0):
            raise ValueError(f"Winner must be one of {{-1, 0, 1}}, got {winner!r}.")

        start = self._active_game_starts.pop(0)
        if start >= len(self.outcomes):
            return

        if winner is None:
            winner = 0

        for i in range(start, len(self.outcomes)):
            mover = self.current_players[i]
            if mover == 0:
                self.outcomes[i] = 0
            elif winner == 0:
                self.outcomes[i] = 0
            elif mover == winner:
                self.outcomes[i] = 1
            else:
                self.outcomes[i] = -1

    def clear_buffer(self) -> None:
        """Clear in-memory logging buffers without touching existing dataset file."""
        self.states.clear()
        self.actions.clear()
        self.current_players.clear()
        self.outcomes.clear()
        self._active_game_starts.clear()

    def save(self, output_path: Path, append: bool = True) -> tuple[int, int]:
        """
        Save logged data to `.npz`.

        Returns:
            `(num_added, total_samples_after_save)`.
        """
        num_added = len(self.actions)
        if num_added == 0:
            return 0, self._existing_count(output_path) if append and output_path.exists() else 0

        new_states = np.asarray(self.states, dtype=np.float32)
        new_actions = np.asarray(self.actions, dtype=np.int64)
        new_players = np.asarray(self.current_players, dtype=np.int8)
        new_outcomes = np.asarray(self.outcomes, dtype=np.int8)

        if append and output_path.exists():
            old_states, old_actions, old_players, old_outcomes = self._load_existing_dataset(
                output_path
            )
            all_states = np.concatenate([old_states, new_states], axis=0)
            all_actions = np.concatenate([old_actions, new_actions], axis=0)
            all_players = np.concatenate([old_players, new_players], axis=0)
            all_outcomes = np.concatenate([old_outcomes, new_outcomes], axis=0)
        else:
            all_states = new_states
            all_actions = new_actions
            all_players = new_players
            all_outcomes = new_outcomes

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            states=all_states,
            actions=all_actions,
            current_players=all_players,
            outcomes=all_outcomes,
        )
        return num_added, int(all_actions.shape[0])

    @staticmethod
    def _existing_count(path: Path) -> int:
        states, _, _, _ = GameDataLogger._load_existing_dataset(path)
        return int(states.shape[0])

    @staticmethod
    def _load_existing_dataset(
        path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path) as data:
            if "states" not in data or "actions" not in data:
                raise ValueError(
                    f"Existing file at {path.resolve()} is missing 'states'/'actions' arrays."
                )
            states = data["states"].astype(np.float32)
            actions = data["actions"].astype(np.int64)
            if "current_players" in data:
                current_players = data["current_players"].astype(np.int8)
            else:
                current_players = np.zeros((states.shape[0],), dtype=np.int8)

            if "outcomes" in data:
                outcomes = data["outcomes"].astype(np.int8)
            else:
                outcomes = np.zeros((states.shape[0],), dtype=np.int8)

        if states.ndim != 4 or states.shape[1:] != (3, 9, 9):
            raise ValueError(
                f"Existing states must have shape (N, 3, 9, 9), got {states.shape}."
            )
        if actions.ndim != 1:
            raise ValueError(f"Existing actions must have shape (N,), got {actions.shape}.")
        if current_players.ndim != 1:
            raise ValueError(
                f"Existing current_players must have shape (N,), got {current_players.shape}."
            )
        if outcomes.ndim != 1:
            raise ValueError(f"Existing outcomes must have shape (N,), got {outcomes.shape}.")

        if (
            states.shape[0] != actions.shape[0]
            or states.shape[0] != current_players.shape[0]
            or states.shape[0] != outcomes.shape[0]
        ):
            raise ValueError(
                "Existing dataset arrays mismatch: "
                f"states={states.shape[0]}, actions={actions.shape[0]}, "
                f"current_players={current_players.shape[0]}, "
                f"outcomes={outcomes.shape[0]}."
            )

        return states, actions, current_players, outcomes
