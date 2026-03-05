"""Inference agent backed by a trained PyTorch policy network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.env.gomoku_env import GomokuEnv
from src.env.rules import action_threat_score, immediate_winning_actions
from src.models.policy_net import PolicyNet, mask_illegal_logits
from src.utils.encoding import encode_board


class ModelAgent:
    """Loads a checkpoint and selects legal moves by argmax."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        local_radius: int = 2,
        proximity_scale: float = 0.08,
        threat_block_threshold: int = 2_000,
        threat_bonus_scale: float = 0.00012,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path.resolve()}"
            )
        if local_radius < 1:
            raise ValueError(f"local_radius must be >= 1, got {local_radius}.")
        if proximity_scale < 0.0:
            raise ValueError(f"proximity_scale must be >= 0, got {proximity_scale}.")
        if threat_block_threshold < 1:
            raise ValueError(
                f"threat_block_threshold must be >= 1, got {threat_block_threshold}."
            )
        if threat_bonus_scale < 0.0:
            raise ValueError(f"threat_bonus_scale must be >= 0, got {threat_bonus_scale}.")

        self.device = torch.device(device)
        self.local_radius = local_radius
        self.proximity_scale = proximity_scale
        self.threat_block_threshold = threat_block_threshold
        self.threat_bonus_scale = threat_bonus_scale
        self.model = PolicyNet().to(self.device)
        # NOTE: weights_only=False is required here for checkpoints that include
        # non-tensor metadata (e.g., Path-like values) and to remain compatible
        # with PyTorch 2.6+ local loading semantics.
        state = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(state, dict):
            if "model_state_dict" in state:
                state_dict = state["model_state_dict"]
            elif "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state
        else:
            raise ValueError("Unsupported checkpoint format: expected a dict.")

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def select_action(self, env: GomokuEnv) -> int:
        legal_actions = env.legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions available.")

        if len(legal_actions) == env.board_size * env.board_size:
            center = env.board_size // 2
            return env.rc_to_action(center, center)

        encoded = encode_board(env.board, env.current_player)
        x = torch.from_numpy(encoded).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

            # Offensive guard: always take immediate winning move.
            my_wins = immediate_winning_actions(
                env.board,
                player=env.current_player,
                win_length=env.win_length,
            )
            if my_wins:
                masked_logits = mask_illegal_logits(logits, my_wins)
                return int(torch.argmax(masked_logits, dim=1).item())

            # Defensive guard:
            # If opponent has a one-move win threat, restrict policy choice to
            # blocking actions only.
            must_block = immediate_winning_actions(
                env.board,
                player=-env.current_player,
                win_length=env.win_length,
            )
            if must_block:
                masked_logits = mask_illegal_logits(logits, must_block)
                return int(torch.argmax(masked_logits, dim=1).item())

            # Secondary defensive guard:
            # If opponent can create a strong threat (e.g., open-three/open-four),
            # choose a move minimizing opponent's best next-turn threat.
            opp_threat = self._best_threat_score(
                env.board,
                legal_actions,
                player=-env.current_player,
                win_length=env.win_length,
            )
            if opp_threat >= self.threat_block_threshold:
                defensive = self._defensive_minimax_actions(env, legal_actions)
                if defensive:
                    masked_logits = mask_illegal_logits(logits, defensive)
                    masked_logits = masked_logits + self._proximity_bonus(
                        env,
                        defensive,
                        scale=self.proximity_scale * 1.3,
                        device=masked_logits.device,
                    )
                    masked_logits = masked_logits + self._counter_threat_bonus(
                        env,
                        defensive,
                        device=masked_logits.device,
                    )
                    return int(torch.argmax(masked_logits, dim=1).item())

            # Offensive pressure: if we can create a strong threat, prioritize it.
            my_best, my_strong_actions = self._best_threat_actions(
                env.board,
                legal_actions,
                player=env.current_player,
                win_length=env.win_length,
            )
            if my_best >= self.threat_block_threshold and my_strong_actions:
                masked_logits = mask_illegal_logits(logits, my_strong_actions)
                masked_logits = masked_logits + self._proximity_bonus(
                    env,
                    my_strong_actions,
                    scale=self.proximity_scale * 1.15,
                    device=masked_logits.device,
                )
                masked_logits = masked_logits + self._offense_threat_bonus(
                    env,
                    my_strong_actions,
                    device=masked_logits.device,
                )
                return int(torch.argmax(masked_logits, dim=1).item())

            # Positional prior: keep search local to active area of the board.
            candidate_actions = self._local_candidate_actions(env, legal_actions)

            # Inference:
            # logits_j are unnormalized log-scores for action j.
            # We restrict to candidate set A_cand and choose:
            #   a* = argmax_{a in A_cand} logits_a
            masked_logits = mask_illegal_logits(logits, candidate_actions)
            if self.proximity_scale > 0.0:
                masked_logits = masked_logits + self._proximity_bonus(
                    env,
                    candidate_actions,
                    scale=self.proximity_scale,
                    device=masked_logits.device,
                )
            masked_logits = masked_logits + self._offense_threat_bonus(
                env,
                candidate_actions,
                device=masked_logits.device,
            )
            action = int(torch.argmax(masked_logits, dim=1).item())
        return action

    def _local_candidate_actions(self, env: GomokuEnv, legal_actions: list[int]) -> list[int]:
        occupied = np.argwhere(env.board != 0)
        if occupied.size == 0:
            return legal_actions

        candidates: list[int] = []
        for action in legal_actions:
            row, col = env.action_to_rc(action)
            deltas = np.abs(occupied - np.array([row, col]))
            is_local = np.any(np.max(deltas, axis=1) <= self.local_radius)
            if is_local:
                candidates.append(action)
        return candidates if candidates else legal_actions

    def _best_threat_actions(
        self,
        board: np.ndarray,
        actions: list[int],
        player: int,
        win_length: int,
    ) -> tuple[int, list[int]]:
        best = -1_000_000
        best_actions: list[int] = []
        for action in actions:
            score = action_threat_score(board, action, player, win_length=win_length)
            if score > best:
                best = score
                best_actions = [action]
            elif score == best:
                best_actions.append(action)
        return best, best_actions

    def _best_threat_score(
        self,
        board: np.ndarray,
        actions: list[int],
        player: int,
        win_length: int,
    ) -> int:
        best, _ = self._best_threat_actions(board, actions, player, win_length=win_length)
        return best

    def _defensive_minimax_actions(self, env: GomokuEnv, legal_actions: list[int]) -> list[int]:
        """
        Choose actions that minimize opponent's best tactical reply threat.
        """
        my_player = env.current_player
        opp_player = -my_player

        best_risk: int | None = None
        best_actions: list[int] = []

        for action in legal_actions:
            row, col = env.action_to_rc(action)
            env.board[row, col] = my_player

            opp_legal = np.flatnonzero(env.board.ravel() == 0).astype(np.int64).tolist()
            if not opp_legal:
                risk = 0
            else:
                risk = self._best_threat_score(
                    env.board,
                    opp_legal,
                    player=opp_player,
                    win_length=env.win_length,
                )

            env.board[row, col] = 0

            if best_risk is None or risk < best_risk:
                best_risk = risk
                best_actions = [action]
            elif risk == best_risk:
                best_actions.append(action)

        return best_actions

    def _proximity_bonus(
        self,
        env: GomokuEnv,
        candidate_actions: list[int],
        scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Add small local-structure bias to logits.

        For each candidate action, score nearby own/opponent stones within radius 2.
        """
        bonus = torch.zeros((1, 81), dtype=torch.float32, device=device)
        occupied = np.argwhere(env.board != 0)
        if occupied.size == 0:
            return bonus

        last_rc: tuple[int, int] | None = None
        if env.last_action is not None:
            last_rc = env.action_to_rc(env.last_action)

        for action in candidate_actions:
            row, col = env.action_to_rc(action)
            score = 0.0
            for rr, cc in occupied:
                dr = abs(int(rr) - row)
                dc = abs(int(cc) - col)
                cheb = max(dr, dc)
                if cheb == 0 or cheb > 2:
                    continue
                stone = int(env.board[int(rr), int(cc)])
                base = 3 - cheb  # radius-1 gives 2, radius-2 gives 1.
                if stone == env.current_player:
                    score += 1.1 * base
                else:
                    score += 0.8 * base

            # Encourage tactical continuation around the most recent move.
            if last_rc is not None:
                lr, lc = last_rc
                cheb_last = max(abs(lr - row), abs(lc - col))
                if cheb_last == 1:
                    score += 2.2
                elif cheb_last == 2:
                    score += 1.1

            bonus[0, action] = float(scale * score)
        return bonus

    def _offense_threat_bonus(
        self,
        env: GomokuEnv,
        candidate_actions: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Prefer moves that increase our own tactical threat score (build 3/4).
        """
        bonus = torch.zeros((1, 81), dtype=torch.float32, device=device)
        for action in candidate_actions:
            my_score = action_threat_score(
                env.board,
                action,
                player=env.current_player,
                win_length=env.win_length,
            )
            bonus[0, action] = float(self.threat_bonus_scale * my_score)
        return bonus

    def _counter_threat_bonus(
        self,
        env: GomokuEnv,
        candidate_actions: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        In defensive mode, prefer blocks that also create our own pressure.

        For each candidate action:
        - add own post-move threat score
        - subtract opponent post-move max threat score
        """
        bonus = torch.zeros((1, 81), dtype=torch.float32, device=device)
        my_player = env.current_player
        opp_player = -my_player

        for action in candidate_actions:
            my_score = action_threat_score(
                env.board,
                action,
                player=my_player,
                win_length=env.win_length,
            )
            row, col = env.action_to_rc(action)
            env.board[row, col] = my_player

            opp_legal = np.flatnonzero(env.board.ravel() == 0).astype(np.int64).tolist()
            if opp_legal:
                opp_best = self._best_threat_score(
                    env.board,
                    opp_legal,
                    player=opp_player,
                    win_length=env.win_length,
                )
            else:
                opp_best = 0

            env.board[row, col] = 0

            # Heavier weight on reducing opponent tactical continuation.
            combined = 1.2 * my_score - 1.6 * opp_best
            bonus[0, action] = float(self.threat_bonus_scale * combined)

        return bonus
