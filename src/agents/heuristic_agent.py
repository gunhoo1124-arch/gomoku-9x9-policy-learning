"""Heuristic Gomoku agent with tactical priorities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.env.gomoku_env import GomokuEnv
from src.env.rules import DIRECTIONS, check_five_in_a_row, count_in_direction


class HeuristicAgent:
    """
    A stronger-than-random heuristic policy.

    Tactical priority ladder:
    1. Immediate winning move
    2. Block opponent immediate winning move
    3. Create strongest threat
    4. Block strongest opponent threat
    5. Prefer center
    6. Prefer moves near existing stones
    7. Otherwise choose sensibly among remaining legal moves

    By default this agent is deterministic. Set `noise > 0` to introduce
    small stochasticity (useful for data generation diversity).
    """

    def __init__(self, noise: float = 0.0, seed: Optional[int] = None) -> None:
        if noise < 0.0:
            raise ValueError("noise must be >= 0.")
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: GomokuEnv) -> int:
        scores = self.action_scores(env)
        if not scores:
            raise RuntimeError("No legal actions available.")

        best_action = min(scores.keys())
        best_score = float("-inf")
        for action in sorted(scores):
            score = float(scores[action])
            if self.noise > 0.0:
                score += float(self.rng.normal(0.0, self.noise))
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def action_scores(self, env: GomokuEnv) -> dict[int, float]:
        """
        Return dense heuristic scores for all legal actions.

        These scores preserve the tactical ladder used in `select_action` while
        remaining usable as teacher logits for soft-target generation.
        """
        legal_actions = sorted(env.legal_actions())
        if not legal_actions:
            return {}

        board = env.board
        player = env.current_player
        opponent = -player
        center = env.board_size // 2

        scores: dict[int, float] = {a: -4.0 for a in legal_actions}

        # Candidate filtering rule: on empty board, play center.
        if len(legal_actions) == env.board_size * env.board_size:
            center_action = env.rc_to_action(center, center)
            scores[center_action] = 8.0
            for action in legal_actions:
                if action == center_action:
                    continue
                row, col = env.action_to_rc(action)
                center_pref = -abs(row - center) - abs(col - center)
                scores[action] = 0.25 * center_pref
            return scores

        candidates = self._candidate_actions(env, legal_actions)

        winning_now = [
            action
            for action in legal_actions
            if self._is_immediate_win(board, action, player, env.board_size, env.win_length)
        ]
        if winning_now:
            for action in winning_now:
                row, col = env.action_to_rc(action)
                center_pref = -abs(row - center) - abs(col - center)
                nearby = self._nearby_stones_score(board, row, col)
                scores[action] = 9.0 + 0.05 * center_pref + 0.03 * nearby
            return scores

        opponent_winning_now = [
            action
            for action in legal_actions
            if self._is_immediate_win(board, action, opponent, env.board_size, env.win_length)
        ]
        if opponent_winning_now:
            for action in opponent_winning_now:
                row, col = env.action_to_rc(action)
                center_pref = -abs(row - center) - abs(col - center)
                nearby = self._nearby_stones_score(board, row, col)
                scores[action] = 8.0 + 0.05 * center_pref + 0.03 * nearby
            return scores

        my_threat_scores = {
            action: self._threat_score(board, action, player, env.board_size, env.win_length)
            for action in candidates
        }
        opp_threat_scores = {
            action: self._threat_score(
                board, action, opponent, env.board_size, env.win_length
            )
            for action in candidates
        }
        best_my = max(my_threat_scores.values())
        best_opp = max(opp_threat_scores.values())

        threat_threshold = 120
        for action in legal_actions:
            if action not in candidates:
                continue

            row, col = env.action_to_rc(action)
            center_pref = -abs(row - center) - abs(col - center)
            nearby = self._nearby_stones_score(board, row, col)
            my_threat = float(my_threat_scores[action])
            opp_threat = float(opp_threat_scores[action])

            # Composite tactical logit:
            # - own threat creation
            # - threat suppression
            # - local positional priors
            score = (
                0.0017 * my_threat
                + 0.0013 * opp_threat
                + 0.12 * center_pref
                + 0.09 * nearby
            )
            if my_threat == best_my and best_my >= threat_threshold:
                score += 1.5
            if opp_threat == best_opp and best_opp >= threat_threshold:
                score += 1.3

            scores[action] = score

        return scores

    def policy_distribution(self, env: GomokuEnv, temperature: float = 1.0) -> np.ndarray:
        """
        Build a teacher policy distribution over all 81 actions.

        Output:
            float32 array of shape `(81,)`, summing to 1, with zero mass on illegal moves.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        legal_actions = sorted(env.legal_actions())
        if not legal_actions:
            raise RuntimeError("No legal actions available.")

        scores = self.action_scores(env)
        logits = np.full((env.board_size * env.board_size,), -1e9, dtype=np.float64)
        for action in legal_actions:
            logits[action] = float(scores[action]) / float(temperature)

        # Stable softmax over legal actions.
        legal_logits = logits[legal_actions]
        max_logit = float(np.max(legal_logits))
        exp_vals = np.exp(np.clip(legal_logits - max_logit, -60.0, 0.0))
        probs_legal = exp_vals / np.sum(exp_vals)

        probs = np.zeros_like(logits, dtype=np.float64)
        probs[legal_actions] = probs_legal
        return probs.astype(np.float32)

    def _candidate_actions(self, env: GomokuEnv, legal_actions: list[int]) -> list[int]:
        """
        Keep only legal moves within Chebyshev distance <= 2 of any stone.
        """
        board = env.board
        occupied = np.argwhere(board != 0)
        if occupied.size == 0:
            center = env.board_size // 2
            return [env.rc_to_action(center, center)]

        candidates: list[int] = []
        for action in legal_actions:
            row, col = env.action_to_rc(action)
            deltas = np.abs(occupied - np.array([row, col]))
            close_to_stone = np.any(np.max(deltas, axis=1) <= 2)
            if close_to_stone:
                candidates.append(action)
        return candidates if candidates else legal_actions

    def _pick_best(
        self,
        env: GomokuEnv,
        actions: list[int],
        primary_scores: Optional[dict[int, int]] = None,
    ) -> int:
        """
        Pick best move with deterministic tie-break:
        primary score -> center -> nearby stones -> low action index.
        """
        center = env.board_size // 2
        board = env.board

        best_action = actions[0]
        best_value: tuple[float, int, int, int] | None = None

        for action in sorted(actions):
            row, col = env.action_to_rc(action)
            primary = float(primary_scores[action]) if primary_scores else 0.0
            if self.noise > 0.0:
                primary += float(self.rng.normal(0.0, self.noise))

            center_pref = -abs(row - center) - abs(col - center)
            nearby_pref = self._nearby_stones_score(board, row, col)
            value = (primary, center_pref, nearby_pref, -action)
            if best_value is None or value > best_value:
                best_value = value
                best_action = action

        return int(best_action)

    def _nearby_stones_score(self, board: np.ndarray, row: int, col: int) -> int:
        """Preference bonus for moves near existing stones (radius <= 2)."""
        score = 0
        size = board.shape[0]
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                rr = row + dr
                cc = col + dc
                if 0 <= rr < size and 0 <= cc < size and board[rr, cc] != 0:
                    distance = max(abs(dr), abs(dc))
                    score += 3 - distance  # distance 1 -> +2, distance 2 -> +1
        return score

    def _is_immediate_win(
        self,
        board: np.ndarray,
        action: int,
        player: int,
        board_size: int,
        win_length: int,
    ) -> bool:
        row, col = divmod(action, board_size)
        if board[row, col] != 0:
            return False
        board[row, col] = player
        is_win = check_five_in_a_row(board, row, col, player, win_length=win_length)
        board[row, col] = 0
        return is_win

    def _threat_score(
        self,
        board: np.ndarray,
        action: int,
        player: int,
        board_size: int,
        win_length: int,
    ) -> int:
        """Estimate tactical strength if `player` plays `action`."""
        row, col = divmod(action, board_size)
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

            if 0 <= fr < board_size and 0 <= fc < board_size and board[fr, fc] == 0:
                open_ends += 1
            if 0 <= br < board_size and 0 <= bc < board_size and board[br, bc] == 0:
                open_ends += 1

            score = self._pattern_score(total, open_ends, win_length)
            if score > best:
                best = score

        board[row, col] = 0
        return best

    @staticmethod
    def _pattern_score(length: int, open_ends: int, win_length: int) -> int:
        """Map line pattern to a tactical score."""
        if length >= win_length:
            return 100_000
        if length == 4 and open_ends == 2:
            return 10_000
        if length == 4 and open_ends == 1:
            return 6_000
        if length == 3 and open_ends == 2:
            return 2_000
        if length == 3 and open_ends == 1:
            return 400
        if length == 2 and open_ends == 2:
            return 150
        if length == 2 and open_ends == 1:
            return 40
        if length == 1 and open_ends == 2:
            return 5
        return 1
