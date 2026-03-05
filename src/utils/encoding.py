"""Board encoding utilities for model input."""

from __future__ import annotations

import numpy as np


def encode_board(board: np.ndarray, current_player: int) -> np.ndarray:
    """
    Encode board from side-to-move perspective into 3 channels.

    Channels:
    1. Current player stones
    2. Opponent stones
    3. Turn plane: `1.0` if current player is black (`1`), else `0.0`

    Returns:
        np.ndarray of shape `(3, 9, 9)` and dtype `float32`.
    """
    if board.shape != (9, 9):
        raise ValueError(f"Expected board shape (9, 9), got {board.shape}.")
    if current_player not in (1, -1):
        raise ValueError(
            f"current_player must be 1 (black) or -1 (white), got {current_player}."
        )

    current_stones = (board == current_player).astype(np.float32)
    opponent_stones = (board == -current_player).astype(np.float32)
    turn_plane = np.full(
        (9, 9),
        1.0 if current_player == 1 else 0.0,
        dtype=np.float32,
    )
    return np.stack([current_stones, opponent_stones, turn_plane], axis=0)
