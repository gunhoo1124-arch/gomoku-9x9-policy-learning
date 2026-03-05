"""Tests for outcome-conditioned sample weighting."""

from __future__ import annotations

import unittest

import numpy as np

from src.training.train_supervised import apply_outcome_weights


class TestOutcomeWeights(unittest.TestCase):
    def test_outcome_scaling(self) -> None:
        weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        outcomes = np.array([1, -1, 0, -1], dtype=np.int8)

        out_weights, num_wins, num_losses, num_draws = apply_outcome_weights(
            weights=weights.copy(),
            outcomes=outcomes,
            win_weight=2.0,
            loss_weight=0.5,
            draw_weight=1.25,
        )

        np.testing.assert_allclose(out_weights, np.array([2.0, 0.5, 1.25, 0.5], dtype=np.float32))
        self.assertEqual(num_wins, 1)
        self.assertEqual(num_losses, 2)
        self.assertEqual(num_draws, 1)

    def test_invalid_outcome_weights(self) -> None:
        weights = np.ones((2,), dtype=np.float32)
        outcomes = np.array([1, -1], dtype=np.int8)

        with self.assertRaises(ValueError):
            apply_outcome_weights(weights, outcomes, win_weight=-1.0, loss_weight=1.0, draw_weight=1.0)


if __name__ == "__main__":
    unittest.main()

