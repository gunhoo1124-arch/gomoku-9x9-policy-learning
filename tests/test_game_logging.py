"""Tests for match-data logging into supervised dataset format."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.game_logging import GameDataLogger


class TestGameDataLogger(unittest.TestCase):
    def test_human_policy_logs_only_human_moves(self) -> None:
        logger = GameDataLogger(log_policy="human")
        board = np.zeros((9, 9), dtype=np.int8)

        logger.record(board, current_player=1, action=40, actor="human")
        logger.record(board, current_player=-1, action=41, actor="model")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "human_games.npz"
            added, total = logger.save(output, append=False)

            self.assertEqual(added, 1)
            self.assertEqual(total, 1)

            with np.load(output) as data:
                self.assertEqual(data["states"].shape, (1, 3, 9, 9))
                self.assertEqual(data["actions"].shape, (1,))
                self.assertEqual(data["current_players"].shape, (1,))
                self.assertEqual(data["outcomes"].shape, (1,))
                self.assertEqual(int(data["actions"][0]), 40)
                self.assertEqual(int(data["outcomes"][0]), 0)

    def test_outcomes_set_by_game_result(self) -> None:
        logger = GameDataLogger(log_policy="all")
        board = np.zeros((9, 9), dtype=np.int8)

        logger.start_game()
        logger.record(board, current_player=1, action=10, actor="heuristic")
        logger.record(board, current_player=-1, action=20, actor="model")
        logger.finalize_game(winner=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "matches.npz"
            added, total = logger.save(output, append=False)
            self.assertEqual((added, total), (2, 2))

            with np.load(output) as data:
                self.assertEqual(list(data["outcomes"].tolist()), [1, -1])

    def test_append_merges_with_existing_dataset(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "matches.npz"

            first = GameDataLogger(log_policy="all")
            first.record(board, current_player=1, action=10, actor="heuristic")
            added_1, total_1 = first.save(output, append=False)
            self.assertEqual((added_1, total_1), (1, 1))

            second = GameDataLogger(log_policy="all")
            second.record(board, current_player=-1, action=20, actor="model")
            added_2, total_2 = second.save(output, append=True)
            self.assertEqual((added_2, total_2), (1, 2))

            with np.load(output) as data:
                self.assertEqual(data["states"].shape, (2, 3, 9, 9))
                self.assertEqual(data["actions"].tolist(), [10, 20])


if __name__ == "__main__":
    unittest.main()
