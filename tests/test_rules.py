"""Rule and environment tests."""

import unittest

import numpy as np

from src.env.gomoku_env import GomokuEnv
from src.env.rules import (
    action_threat_score,
    check_five_in_a_row,
    immediate_winning_actions,
    max_player_threat_score,
)


class TestGomokuRules(unittest.TestCase):
    def test_horizontal_win(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        row = 4
        for col in range(5):
            board[row, col] = 1
        self.assertTrue(check_five_in_a_row(board, row, 2, 1))

    def test_vertical_win(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        col = 3
        for row in range(5):
            board[row, col] = -1
        self.assertTrue(check_five_in_a_row(board, 2, col, -1))

    def test_diagonal_win(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        for i in range(5):
            board[i, i] = 1
        self.assertTrue(check_five_in_a_row(board, 2, 2, 1))

    def test_antidiagonal_win(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        for i in range(5):
            board[i, 8 - i] = -1
        self.assertTrue(check_five_in_a_row(board, 2, 6, -1))

    def test_legal_action_handling(self) -> None:
        env = GomokuEnv()
        self.assertEqual(len(env.legal_actions()), 81)

        first_action = env.rc_to_action(0, 0)
        env.step(first_action)
        legal = env.legal_actions()
        self.assertEqual(len(legal), 80)
        self.assertNotIn(first_action, legal)

        with self.assertRaises(ValueError):
            env.step(first_action)

    def test_immediate_winning_actions_detects_open_four(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        row = 4
        for col in (2, 3, 4, 5):
            board[row, col] = -1

        winning = set(immediate_winning_actions(board, player=-1))
        expected = {row * 9 + 1, row * 9 + 6}
        self.assertTrue(expected.issubset(winning))

    def test_immediate_winning_actions_empty_when_no_threat(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[4, 4] = 1
        board[4, 5] = -1
        self.assertEqual(immediate_winning_actions(board, player=-1), [])

    def test_action_threat_score_open_three_is_high(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[4, 3] = 1
        board[4, 4] = 1
        action = 4 * 9 + 5  # creates 3 in a row with both ends open.
        score = action_threat_score(board, action=action, player=1)
        self.assertGreaterEqual(score, 2_000)

    def test_max_player_threat_score_detects_strong_threat(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[2, 2] = -1
        board[2, 3] = -1
        board[2, 4] = -1
        score = max_player_threat_score(board, player=-1)
        self.assertGreaterEqual(score, 2_000)


if __name__ == "__main__":
    unittest.main()
