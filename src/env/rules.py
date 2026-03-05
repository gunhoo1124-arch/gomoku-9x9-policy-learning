"""Core Gomoku rule helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),   # Horizontal
    (1, 0),   # Vertical
    (1, 1),   # Main diagonal
    (1, -1),  # Anti-diagonal
)


def count_in_direction(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    dr: int,
    dc: int,
) -> int:
    """
    Count contiguous stones for `player` from `(row, col)` in direction `(dr, dc)`.

    The start cell `(row, col)` is excluded from the count.
    """
    size = board.shape[0]
    count = 0
    r = row + dr
    c = col + dc

    while 0 <= r < size and 0 <= c < size and board[r, c] == player:
        count += 1
        r += dr
        c += dc

    return count


def check_five_in_a_row(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    win_length: int = 5,
) -> bool:
    """
    Check whether a stone at `(row, col)` completes `win_length` in a row.
    """
    if board[row, col] != player:
        return False

    for dr, dc in DIRECTIONS:
        forward = count_in_direction(board, row, col, player, dr, dc)
        backward = count_in_direction(board, row, col, player, -dr, -dc)
        if 1 + forward + backward >= win_length:
            return True

    return False


def board_full(board: np.ndarray) -> bool:
    """Return `True` when no legal moves remain."""
    return not np.any(board == 0)


def iter_empty_cells(board: np.ndarray) -> Iterable[tuple[int, int]]:
    """Yield all empty cell coordinates."""
    empties = np.argwhere(board == 0)
    for r, c in empties:
        yield int(r), int(c)


def immediate_winning_actions(
    board: np.ndarray,
    player: int,
    win_length: int = 5,
    candidate_actions: Sequence[int] | None = None,
) -> list[int]:
    """
    Return legal actions that would win immediately for `player`.

    Each action is encoded as `row * board_size + col`.
    """
    size = int(board.shape[0])
    winning: list[int] = []

    if candidate_actions is None:
        candidates = [row * size + col for row, col in iter_empty_cells(board)]
    else:
        candidates = [int(a) for a in candidate_actions]

    for action in candidates:
        row, col = divmod(action, size)
        if board[row, col] != 0:
            continue
        board[row, col] = player
        is_win = check_five_in_a_row(board, row, col, player, win_length=win_length)
        board[row, col] = 0
        if is_win:
            winning.append(action)

    return winning


def pattern_score(length: int, open_ends: int, win_length: int = 5) -> int:
    """
    Map a local line pattern to a tactical score.

    The scale is intentionally coarse and monotonic:
    immediate wins > open fours > closed fours > open threes > ...
    """
    if length >= win_length:
        return 100_000
    if length == win_length - 1 and open_ends == 2:
        return 10_000
    if length == win_length - 1 and open_ends == 1:
        return 6_000
    if length == win_length - 2 and open_ends == 2:
        return 2_000
    if length == win_length - 2 and open_ends == 1:
        return 400
    if length == win_length - 3 and open_ends == 2:
        return 150
    if length == win_length - 3 and open_ends == 1:
        return 40
    if length == 1 and open_ends == 2:
        return 5
    return 1


def action_threat_score(
    board: np.ndarray,
    action: int,
    player: int,
    win_length: int = 5,
) -> int:
    """
    Estimate tactical threat strength created by `player` playing `action`.

    Returns a non-negative score on legal moves, and a large negative value
    on illegal moves.
    """
    size = int(board.shape[0])
    row, col = divmod(int(action), size)
    if board[row, col] != 0:
        return -1_000_000

    board[row, col] = player
    best = 0

    for dr, dc in DIRECTIONS:
        forward = count_in_direction(board, row, col, player, dr, dc)
        backward = count_in_direction(board, row, col, player, -dr, -dc)
        total = 1 + forward + backward

        open_ends = 0
        fr = row + (forward + 1) * dr
        fc = col + (forward + 1) * dc
        br = row - (backward + 1) * dr
        bc = col - (backward + 1) * dc

        if 0 <= fr < size and 0 <= fc < size and board[fr, fc] == 0:
            open_ends += 1
        if 0 <= br < size and 0 <= bc < size and board[br, bc] == 0:
            open_ends += 1

        score = pattern_score(total, open_ends, win_length=win_length)
        if score > best:
            best = score

    board[row, col] = 0
    return best


def max_player_threat_score(
    board: np.ndarray,
    player: int,
    win_length: int = 5,
    candidate_actions: Sequence[int] | None = None,
) -> int:
    """
    Return the strongest tactical threat the player can create in one move.
    """
    best = 0
    if candidate_actions is None:
        size = int(board.shape[0])
        candidates = [row * size + col for row, col in iter_empty_cells(board)]
    else:
        candidates = [int(a) for a in candidate_actions]

    for action in candidates:
        score = action_threat_score(board, action, player, win_length=win_length)
        if score > best:
            best = score
    return best


def local_candidate_actions(board: np.ndarray, radius: int = 2) -> list[int]:
    """
    Return legal actions within Chebyshev distance <= radius from any stone.

    Falls back to all legal actions when board is empty or filter yields none.
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}.")

    size = int(board.shape[0])
    legal = np.flatnonzero(board.ravel() == 0).astype(np.int64).tolist()
    occupied = np.argwhere(board != 0)

    if not legal or occupied.size == 0:
        return legal

    candidates: list[int] = []
    for action in legal:
        row, col = divmod(int(action), size)
        deltas = np.abs(occupied - np.array([row, col]))
        if np.any(np.max(deltas, axis=1) <= radius):
            candidates.append(int(action))

    return candidates if candidates else legal
