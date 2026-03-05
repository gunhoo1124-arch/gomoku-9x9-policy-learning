"""Supervised training script for Gomoku policy model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.env.rules import (
    immediate_winning_actions,
    local_candidate_actions,
    max_player_threat_score,
)
from src.models.policy_net import PolicyNet


NUM_ACTIONS = 81
BOARD_SIZE = 9
TargetMode = Literal["auto", "hard", "soft"]


@dataclass
class EvalMetrics:
    loss: float
    top1: float
    top5: float


@dataclass
class SourceStats:
    path: Path
    samples: int
    has_soft_targets: bool
    has_outcomes: bool


class SymmetryAugmentedDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Dataset with optional random D4 symmetry augmentation for board games."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        policy_targets: np.ndarray,
        weights: np.ndarray,
        augment_symmetry: bool = False,
        seed: int = 42,
    ) -> None:
        self.states = states.astype(np.float32, copy=False)
        self.actions = actions.astype(np.int64, copy=False)
        self.policy_targets = policy_targets.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.augment_symmetry = augment_symmetry
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        state = self.states[idx]
        action = int(self.actions[idx])
        soft = self.policy_targets[idx]
        weight = float(self.weights[idx])

        if self.augment_symmetry:
            transform_id = int(self.rng.integers(0, 8))
            state = transform_state(state, transform_id)
            action = int(ACTION_TRANSFORM_MAP[transform_id, action])
            soft = transform_policy_targets(soft, transform_id)

        x = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32))
        y_hard = torch.tensor(action, dtype=torch.long)
        y_soft = torch.from_numpy(np.ascontiguousarray(soft, dtype=np.float32))
        w = torch.tensor(weight, dtype=torch.float32)
        return x, y_hard, y_soft, w


def transform_rc(row: int, col: int, transform_id: int, board_size: int) -> tuple[int, int]:
    """Apply one of 8 board symmetries to a coordinate."""
    n = board_size
    if transform_id == 0:  # identity
        return row, col
    if transform_id == 1:  # rot90
        return col, n - 1 - row
    if transform_id == 2:  # rot180
        return n - 1 - row, n - 1 - col
    if transform_id == 3:  # rot270
        return n - 1 - col, row
    if transform_id == 4:  # flip left-right
        return row, n - 1 - col
    if transform_id == 5:  # flip up-down
        return n - 1 - row, col
    if transform_id == 6:  # transpose (main diagonal)
        return col, row
    if transform_id == 7:  # anti-diagonal reflection
        return n - 1 - col, n - 1 - row
    raise ValueError(f"Invalid transform_id: {transform_id}")


def build_action_transform_map(board_size: int) -> np.ndarray:
    """Create map[transform_id, old_action] -> new_action."""
    mapping = np.zeros((8, board_size * board_size), dtype=np.int64)
    for t in range(8):
        for action in range(board_size * board_size):
            row, col = divmod(action, board_size)
            nr, nc = transform_rc(row, col, t, board_size)
            mapping[t, action] = nr * board_size + nc
    return mapping


ACTION_TRANSFORM_MAP = build_action_transform_map(BOARD_SIZE)


def transform_state(state: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply board symmetry to channels-first board tensor (C, H, W)."""
    if transform_id == 0:
        transformed = state
    elif transform_id == 1:
        transformed = np.rot90(state, k=1, axes=(1, 2))
    elif transform_id == 2:
        transformed = np.rot90(state, k=2, axes=(1, 2))
    elif transform_id == 3:
        transformed = np.rot90(state, k=3, axes=(1, 2))
    elif transform_id == 4:
        transformed = np.flip(state, axis=2)
    elif transform_id == 5:
        transformed = np.flip(state, axis=1)
    elif transform_id == 6:
        transformed = np.transpose(state, (0, 2, 1))
    elif transform_id == 7:
        transformed = np.flip(np.transpose(state, (0, 2, 1)), axis=(1, 2))
    else:
        raise ValueError(f"Invalid transform_id: {transform_id}")
    return np.ascontiguousarray(transformed, dtype=np.float32)


def transform_policy_targets(policy: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply action-index symmetry map to a policy target vector of shape (81,)."""
    mapped = np.zeros_like(policy, dtype=np.float32)
    mapped[ACTION_TRANSFORM_MAP[transform_id]] = policy
    return mapped


def one_hot_from_actions(actions: np.ndarray, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    one_hot = np.zeros((actions.shape[0], num_actions), dtype=np.float32)
    one_hot[np.arange(actions.shape[0]), actions] = 1.0
    return one_hot


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Per-sample cross entropy with soft labels."""
    log_probs = torch.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1)


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    # Top-k accuracy = mean_i[ target_i in TopK(logits_i) ].
    topk = torch.topk(logits, k=k, dim=1).indices
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_soft_targets: bool,
) -> EvalMetrics:
    model.eval()
    ce_hard = nn.CrossEntropyLoss(reduction="none")
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for xb, y_hard, y_soft, w in dataloader:
            xb = xb.to(device)
            y_hard = y_hard.to(device)
            y_soft = y_soft.to(device)
            w = w.to(device)

            logits = model(xb)
            if use_soft_targets:
                loss_per_sample = soft_cross_entropy(logits, y_soft)
            else:
                loss_per_sample = ce_hard(logits, y_hard)
            batch_weight_sum = w.sum().clamp_min(1.0)
            batch_loss = (loss_per_sample * w).sum() / batch_weight_sum

            batch_size = y_hard.size(0)
            total_samples += batch_size
            weighted_loss_sum += float(batch_loss.item() * batch_weight_sum.item())
            weight_sum += float(batch_weight_sum.item())
            total_top1 += top_k_accuracy(logits, y_hard, k=1) * batch_size
            total_top5 += top_k_accuracy(logits, y_hard, k=5) * batch_size

    if total_samples == 0 or weight_sum == 0.0:
        return EvalMetrics(loss=0.0, top1=0.0, top5=0.0)

    return EvalMetrics(
        loss=weighted_loss_sum / weight_sum,
        top1=total_top1 / total_samples,
        top5=total_top5 / total_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gomoku policy net from .npz data.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Primary .npz dataset path containing states/actions.",
    )
    parser.add_argument(
        "--extra-data",
        type=Path,
        nargs="*",
        default=[],
        help="Optional extra .npz dataset paths to concatenate.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["auto", "hard", "soft"],
        default="auto",
        help=(
            "Training target mode: hard one-hot labels, soft teacher distributions, "
            "or auto-detect soft when available."
        ),
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("checkpoints/train_val_split.npz"),
        help=(
            "Path to store/load fixed train/validation indices for reproducible "
            "run-to-run comparisons."
        ),
    )
    parser.add_argument("--epochs", type=int, default=60, help="Total training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("checkpoints/policy_net.pt"),
        help="Best-checkpoint save path.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=500,
        help="Minimum validation samples when dataset is large enough.",
    )
    parser.add_argument(
        "--symmetry-augment",
        action="store_true",
        help="Enable random board symmetry augmentation on training batches.",
    )
    parser.add_argument(
        "--no-symmetry-augment",
        action="store_true",
        help="Disable symmetry augmentation (overrides --symmetry-augment).",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau patience (epochs).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Lower bound for scheduler learning rate.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=12,
        help="Early stopping patience in epochs without val_loss improvement (<=0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_loss improvement to reset early-stop counter.",
    )
    parser.add_argument(
        "--defense-weight",
        type=float,
        default=3.0,
        help=(
            "Sample weight multiplier for must-block states where opponent has "
            "an immediate winning move."
        ),
    )
    parser.add_argument(
        "--attack-weight",
        type=float,
        default=3.0,
        help=(
            "Sample weight multiplier for must-win states where current player "
            "has an immediate winning move."
        ),
    )
    parser.add_argument(
        "--threat-weight",
        type=float,
        default=1.8,
        help=(
            "Extra multiplier when opponent has a strong one-move threat "
            "(e.g., open three/open four)."
        ),
    )
    parser.add_argument(
        "--build-threat-weight",
        type=float,
        default=2.0,
        help=(
            "Extra multiplier when current player can create a strong one-move threat "
            "(e.g., open three/open four)."
        ),
    )
    parser.add_argument(
        "--threat-threshold",
        type=int,
        default=2_000,
        help="Threat score threshold used for threat-weighting.",
    )
    parser.add_argument(
        "--weight-scan-radius",
        type=int,
        default=2,
        help=(
            "Local candidate radius for tactical-weight scans. "
            "Lower is faster; 2 is recommended."
        ),
    )
    parser.add_argument(
        "--weight-progress-every",
        type=int,
        default=5000,
        help="Print tactical-weight preprocessing progress every N samples.",
    )
    parser.add_argument(
        "--outcome-win-weight",
        type=float,
        default=1.35,
        help="Multiplier for moves played by the eventual game winner (can be >, <, or = 1.0).",
    )
    parser.add_argument(
        "--outcome-loss-weight",
        type=float,
        default=1.45,
        help="Multiplier for moves played by the eventual game loser (can be >, <, or = 1.0).",
    )
    parser.add_argument(
        "--outcome-draw-weight",
        type=float,
        default=1.0,
        help="Multiplier for moves from drawn games (can be >, <, or = 1.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, bool]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")

    with np.load(path) as data:
        if "states" not in data or "actions" not in data:
            raise ValueError("Dataset .npz must contain 'states' and 'actions' arrays.")

        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.int64)
        policy_targets = data["policy_targets"].astype(np.float32) if "policy_targets" in data else None
        has_outcomes = "outcomes" in data
        if has_outcomes:
            outcomes = data["outcomes"].astype(np.int8)
        else:
            outcomes = np.zeros((states.shape[0],), dtype=np.int8)

    if states.ndim != 4 or states.shape[1:] != (3, 9, 9):
        raise ValueError(f"states must have shape (N, 3, 9, 9), got {states.shape}.")
    if actions.ndim != 1:
        raise ValueError(f"actions must have shape (N,), got {actions.shape}.")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}."
        )
    if np.any((actions < 0) | (actions > 80)):
        raise ValueError("actions must be integer labels in [0, 80].")

    if policy_targets is not None:
        if policy_targets.shape != (states.shape[0], NUM_ACTIONS):
            raise ValueError(
                "policy_targets must have shape "
                f"({states.shape[0]}, {NUM_ACTIONS}), got {policy_targets.shape}."
            )
        row_sums = policy_targets.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("policy_targets rows must have positive mass.")
        policy_targets = policy_targets / row_sums
    if outcomes.ndim != 1 or outcomes.shape[0] != states.shape[0]:
        raise ValueError(
            f"outcomes must have shape (N,), got {outcomes.shape}."
        )
    if np.any(outcomes < -1) or np.any(outcomes > 1):
        raise ValueError("outcomes must be in {-1, 0, 1}.")

    return states, actions, policy_targets, outcomes, has_outcomes


def load_datasets(
    primary: Path,
    extras: Sequence[Path],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    list[SourceStats],
    int,
    int,
]:
    all_paths = [primary] + list(extras)
    states_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    policy_targets_list: list[np.ndarray | None] = []
    outcomes_list: list[np.ndarray] = []
    stats: list[SourceStats] = []
    soft_sources = 0
    outcome_sources = 0

    for path in all_paths:
        states, actions, policy_targets, outcomes, has_outcome_key = load_dataset(path)
        states_list.append(states)
        actions_list.append(actions)
        outcomes_list.append(outcomes)
        policy_targets_list.append(policy_targets)
        has_soft = policy_targets is not None
        has_outcomes = has_outcome_key
        if has_soft:
            soft_sources += 1
        if has_outcomes:
            outcome_sources += 1
        stats.append(
            SourceStats(
                path=path,
                samples=int(actions.shape[0]),
                has_soft_targets=has_soft,
                has_outcomes=has_outcomes,
            )
        )

    merged_states = np.concatenate(states_list, axis=0)
    merged_actions = np.concatenate(actions_list, axis=0)
    merged_outcomes = np.concatenate(outcomes_list, axis=0)

    if soft_sources == 0:
        merged_policy_targets = None
    else:
        normalized_parts: list[np.ndarray] = []
        for actions_part, policy_part in zip(actions_list, policy_targets_list):
            if policy_part is None:
                normalized_parts.append(one_hot_from_actions(actions_part))
            else:
                normalized_parts.append(policy_part.astype(np.float32, copy=False))
        merged_policy_targets = np.concatenate(normalized_parts, axis=0)

    return (
        merged_states,
        merged_actions,
        merged_policy_targets,
        merged_outcomes,
        stats,
        soft_sources,
        outcome_sources,
    )


def split_indices(
    num_samples: int,
    seed: int,
    val_ratio: float = 0.15,
    min_val_samples: int = 500,
    split_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_samples < 2:
        raise ValueError("Need at least 2 samples to create train/validation split.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
    if min_val_samples < 1:
        raise ValueError(f"min_val_samples must be >= 1, got {min_val_samples}.")

    if split_path is not None and split_path.exists():
        with np.load(split_path) as data:
            if "train_idx" not in data or "val_idx" not in data:
                raise ValueError(
                    f"Split file missing train_idx/val_idx: {split_path.resolve()}"
                )
            train_idx = data["train_idx"].astype(np.int64)
            val_idx = data["val_idx"].astype(np.int64)

        if len(train_idx) + len(val_idx) != num_samples:
            raise ValueError(
                "Existing split size does not match current dataset size. "
                f"Delete {split_path.resolve()} to regenerate split."
            )
        return train_idx, val_idx

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    target_val = max(int(num_samples * val_ratio), min_val_samples)
    val_size = min(max(1, target_val), num_samples - 1)

    val_idx = indices[:val_size].astype(np.int64)
    train_idx = indices[val_size:].astype(np.int64)

    if split_path is not None:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, train_idx=train_idx, val_idx=val_idx)

    return train_idx, val_idx


def decode_relative_board(encoded_state: np.ndarray) -> np.ndarray:
    """
    Decode one encoded state back to a relative board.

    Output convention:
    - `1`: current player stones
    - `-1`: opponent stones
    - `0`: empty
    """
    current = encoded_state[0] > 0.5
    opponent = encoded_state[1] > 0.5

    board = np.zeros((9, 9), dtype=np.int8)
    board[current] = 1
    board[opponent] = -1
    return board


def compute_tactical_weights(
    states: np.ndarray,
    defense_weight: float,
    attack_weight: float,
    threat_weight: float,
    build_threat_weight: float,
    threat_threshold: int,
    weight_scan_radius: int = 2,
    weight_progress_every: int = 5000,
    ) -> tuple[np.ndarray, int, int, int, int]:
    """
    Assign higher training weight to tactical states.

    Definitions (from side-to-move perspective):
    - Must-win: current player has >=1 immediate winning action.
    - Must-block: opponent has >=1 immediate winning action.
    - Strong threat: opponent can create a pattern at/above threat_threshold.
    - Build threat: current player can create a pattern at/above threat_threshold.
    """
    if defense_weight < 1.0:
        raise ValueError(f"defense_weight must be >= 1.0, got {defense_weight}.")
    if attack_weight < 1.0:
        raise ValueError(f"attack_weight must be >= 1.0, got {attack_weight}.")
    if threat_weight < 1.0:
        raise ValueError(f"threat_weight must be >= 1.0, got {threat_weight}.")
    if build_threat_weight < 1.0:
        raise ValueError(
            f"build_threat_weight must be >= 1.0, got {build_threat_weight}."
        )
    if threat_threshold < 1:
        raise ValueError(f"threat_threshold must be >= 1, got {threat_threshold}.")
    if weight_scan_radius < 1:
        raise ValueError(
            f"weight_scan_radius must be >= 1, got {weight_scan_radius}."
        )
    if weight_progress_every < 0:
        raise ValueError(
            f"weight_progress_every must be >= 0, got {weight_progress_every}."
        )

    weights = np.ones((states.shape[0],), dtype=np.float32)
    num_must_block = 0
    num_must_win = 0
    num_strong_threat = 0
    num_build_threat = 0

    for i in range(states.shape[0]):
        board = decode_relative_board(states[i])
        candidates = local_candidate_actions(board, radius=weight_scan_radius)

        my_winning_now = immediate_winning_actions(
            board,
            player=1,
            win_length=5,
            candidate_actions=candidates,
        )
        opponent_winning_now = immediate_winning_actions(
            board,
            player=-1,
            win_length=5,
            candidate_actions=candidates,
        )
        my_max_threat = max_player_threat_score(
            board,
            player=1,
            win_length=5,
            candidate_actions=candidates,
        )
        opponent_max_threat = max_player_threat_score(
            board,
            player=-1,
            win_length=5,
            candidate_actions=candidates,
        )

        if my_winning_now:
            weights[i] *= float(attack_weight)
            num_must_win += 1
        if opponent_winning_now:
            weights[i] *= float(defense_weight)
            num_must_block += 1
        if opponent_max_threat >= threat_threshold:
            weights[i] *= float(threat_weight)
            num_strong_threat += 1
        if my_max_threat >= threat_threshold:
            weights[i] *= float(build_threat_weight)
            num_build_threat += 1

        if weight_progress_every > 0 and (i + 1) % weight_progress_every == 0:
            print(f"Weight preprocessing: {i + 1}/{states.shape[0]} samples processed...")

    return (
        weights,
        num_must_block,
        num_must_win,
        num_strong_threat,
        num_build_threat,
    )


def apply_outcome_weights(
    weights: np.ndarray,
    outcomes: np.ndarray,
    win_weight: float,
    loss_weight: float,
    draw_weight: float,
) -> tuple[np.ndarray, int, int, int]:
    """
    Scale sample weights by game outcome labels:
    - `1` for moves by the eventual winner,
    - `-1` for moves by the eventual loser,
    - `0` for drawn games.
    """
    if win_weight < 0.0:
        raise ValueError(f"win_weight must be >= 0.0, got {win_weight}.")
    if loss_weight < 0.0:
        raise ValueError(f"loss_weight must be >= 0.0, got {loss_weight}.")
    if draw_weight < 0.0:
        raise ValueError(f"draw_weight must be >= 0.0, got {draw_weight}.")

    if outcomes.shape[0] != weights.shape[0]:
        raise ValueError("outcomes and weights must have same length.")

    num_wins = int(np.count_nonzero(outcomes == 1))
    num_losses = int(np.count_nonzero(outcomes == -1))
    num_draws = int(np.count_nonzero(outcomes == 0))

    if win_weight != 1.0:
        weights[outcomes == 1] *= float(win_weight)
    if loss_weight != 1.0:
        weights[outcomes == -1] *= float(loss_weight)
    if draw_weight != 1.0:
        weights[outcomes == 0] *= float(draw_weight)

    return weights, num_wins, num_losses, num_draws


def build_dataloaders(
    states: np.ndarray,
    actions: np.ndarray,
    policy_targets: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    augment_symmetry: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SymmetryAugmentedDataset(
        states=states[train_idx],
        actions=actions[train_idx],
        policy_targets=policy_targets[train_idx],
        weights=weights[train_idx],
        augment_symmetry=augment_symmetry,
        seed=seed,
    )
    val_ds = SymmetryAugmentedDataset(
        states=states[val_idx],
        actions=actions[val_idx],
        policy_targets=policy_targets[val_idx],
        weights=weights[val_idx],
        augment_symmetry=False,
        seed=seed + 1,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def maybe_resume(
    resume_from: Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float]:
    """
    Resume model/optimizer/scheduler states.

    Returns:
        (start_epoch, best_val_loss)
    """
    if resume_from is None:
        return 1, float("inf")
    if not resume_from.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from.resolve()}")

    # NOTE: weights_only=False is required for checkpoints that include
    # non-tensor metadata (e.g., Paths or other argparse args) and for
    # compatibility with PyTorch 2.6+ local checkpoints.
    ckpt = torch.load(
        resume_from,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(ckpt, dict):
        raise ValueError("Resume checkpoint format must be a dict.")

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("val_loss", float("inf")))
    print(
        f"Resumed from {resume_from.resolve()} | start_epoch={start_epoch} "
        f"| best_val_loss={best_val_loss:.4f}"
    )
    return start_epoch, best_val_loss


def resolve_target_mode(
    mode: TargetMode,
    merged_policy_targets: np.ndarray | None,
    soft_sources: int,
) -> tuple[bool, str]:
    """
    Decide whether to train with soft labels and return a short explanation.
    """
    if mode == "hard":
        return False, "using hard one-hot targets by explicit user choice"
    if mode == "soft":
        if merged_policy_targets is None:
            return True, "soft mode requested; falling back to one-hot soft labels"
        return True, "using soft teacher targets by explicit user choice"

    # auto
    if merged_policy_targets is not None and soft_sources > 0:
        return True, (
            "auto mode detected teacher soft targets "
            f"from {soft_sources} dataset(s)"
        )
    return False, "auto mode found no teacher soft targets; using hard labels"


def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    augment_symmetry = args.symmetry_augment and not args.no_symmetry_augment
    (
        states,
        actions,
        merged_policy_targets,
        merged_outcomes,
        source_stats,
        soft_sources,
        outcome_sources,
    ) = load_datasets(args.data, args.extra_data)

    for src in source_stats:
        print(
            f"Loaded dataset: {src.path} | samples={src.samples} "
            f"| soft_targets={'yes' if src.has_soft_targets else 'no'} "
            f"| outcomes={'yes' if src.has_outcomes else 'no'}"
        )
    print(f"Total merged samples: {len(actions)}")

    use_soft_targets, target_mode_note = resolve_target_mode(
        mode=args.target_mode,
        merged_policy_targets=merged_policy_targets,
        soft_sources=soft_sources,
    )
    if merged_policy_targets is None:
        merged_policy_targets = one_hot_from_actions(actions)

    (
        weights,
        num_must_block,
        num_must_win,
        num_strong_threat,
        num_build_threat,
    ) = compute_tactical_weights(
        states,
        defense_weight=args.defense_weight,
        attack_weight=args.attack_weight,
        threat_weight=args.threat_weight,
        build_threat_weight=args.build_threat_weight,
        threat_threshold=args.threat_threshold,
        weight_scan_radius=args.weight_scan_radius,
        weight_progress_every=args.weight_progress_every,
    )
    (
        weights,
        num_wins,
        num_losses,
        num_draws,
    ) = apply_outcome_weights(
        weights=weights,
        outcomes=merged_outcomes,
        win_weight=args.outcome_win_weight,
        loss_weight=args.outcome_loss_weight,
        draw_weight=args.outcome_draw_weight,
    )
    num_weighted = int(np.count_nonzero(weights > 1.0))

    train_idx, val_idx = split_indices(
        len(actions),
        seed=args.seed,
        val_ratio=args.val_ratio,
        min_val_samples=args.min_val_samples,
        split_path=args.split_path,
    )
    train_loader, val_loader = build_dataloaders(
        states=states,
        actions=actions,
        policy_targets=merged_policy_targets,
        weights=weights,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        augment_symmetry=augment_symmetry,
        seed=args.seed,
    )

    device = torch.device(args.device)
    model = PolicyNet().to(device)
    criterion_train_hard = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    start_epoch, best_val_loss = maybe_resume(
        resume_from=args.resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Train samples: {len(train_idx)}")
    print(f"Valid samples: {len(val_idx)}")
    print(
        f"Tactically weighted samples: {num_weighted}/{len(weights)} "
        f"(must_block={num_must_block}, must_win={num_must_win}, "
        f"strong_threat={num_strong_threat}, build_threat={num_build_threat}, "
        f"defense_weight={args.defense_weight}, attack_weight={args.attack_weight}, "
        f"threat_weight={args.threat_weight}, build_threat_weight={args.build_threat_weight}, "
        f"threat_threshold={args.threat_threshold})"
    )
    print(
        f"Outcome-weighted samples: wins={num_wins} losses={num_losses} "
        f"draws={num_draws} | "
        f"win_weight={args.outcome_win_weight}, loss_weight={args.outcome_loss_weight}, "
        f"draw_weight={args.outcome_draw_weight}, datasets_with_outcomes={outcome_sources}"
    )
    print(f"Target mode: {'soft' if use_soft_targets else 'hard'} | {target_mode_note}")
    print(f"Symmetry augmentation: {'on' if augment_symmetry else 'off'}")
    print(f"Device: {device}")
    print(
        "Reference random-policy CE (81-way): "
        f"{float(np.log(NUM_ACTIONS)):.4f} | near 0 is unrealistic with noisy labels."
    )

    if start_epoch > args.epochs:
        print(
            f"Nothing to do: start_epoch={start_epoch} is greater than --epochs={args.epochs}."
        )
        return

    no_improve_epochs = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_weighted_samples = 0.0
        total_samples = 0

        for xb, y_hard, y_soft, w in train_loader:
            xb = xb.to(device)
            y_hard = y_hard.to(device)
            y_soft = y_soft.to(device)
            w = w.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            if use_soft_targets:
                loss_per_sample = soft_cross_entropy(logits, y_soft)
            else:
                loss_per_sample = criterion_train_hard(logits, y_hard)
            batch_weight_sum = w.sum().clamp_min(1.0)
            loss = (loss_per_sample * w).sum() / batch_weight_sum

            loss.backward()
            optimizer.step()

            batch_size = y_hard.size(0)
            running_loss += float(loss.item() * batch_weight_sum.item())
            total_weighted_samples += float(batch_weight_sum.item())
            total_samples += batch_size

        train_loss = (
            running_loss / total_weighted_samples if total_weighted_samples > 0 else 0.0
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            use_soft_targets=use_soft_targets,
        )
        scheduler.step(val_metrics.loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_lr:.6g} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_top1={val_metrics.top1 * 100:.2f}% | "
            f"val_top5={val_metrics.top5 * 100:.2f}%"
        )

        improved = val_metrics.loss < (best_val_loss - args.early_stop_min_delta)
        if improved:
            best_val_loss = val_metrics.loss
            no_improve_epochs = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics.loss,
                "val_top1": val_metrics.top1,
                "val_top5": val_metrics.top5,
                "target_mode_used": "soft" if use_soft_targets else "hard",
                "config": vars(args),
            }
            torch.save(checkpoint, args.save_path)
            print(f"Saved new best checkpoint -> {args.save_path.resolve()}")
        else:
            no_improve_epochs += 1

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(
                "Early stopping triggered: "
                f"no val_loss improvement for {no_improve_epochs} epochs."
            )
            break

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
