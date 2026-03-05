"""Generate supervised training data from self-play."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.agents.heuristic_agent import HeuristicAgent
from src.env.gomoku_env import GomokuEnv
from src.utils.encoding import encode_board
from src.models.policy_net import PolicyNet, mask_illegal_logits


class _ModelTeacher:
    """Lightweight policy teacher that exposes action selection and soft targets."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        noise: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.device = self._resolve_device(device)
        self.noise = noise

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        self.model = PolicyNet().to(self.device)
        # NOTE: weights_only=False keeps compatibility with checkpoints
        # containing non-tensor metadata created by the trainer script.
        state = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(state, dict):
            raise ValueError(
                "Unsupported checkpoint format: expected a dictionary. "
                "Use `state_dict`, `model_state_dict`, or full dict with keys."
            )

        if "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        elif all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
        ):
            state_dict = state
        else:
            raise ValueError("Checkpoint does not contain a recognisable state dict.")

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print(
                f"CUDA requested for teacher but not available; "
                f"falling back to CPU. Requested device was: {requested}"
            )
            return torch.device("cpu")
        return requested

    def select_action(
        self,
        env: GomokuEnv,
        temperature: float,
    ) -> int:
        if self.noise > 0.0 and self.rng.random() < self.noise:
            legal = env.legal_actions()
            return int(self.rng.choice(legal))

        policy = self.policy_distribution(env, temperature=temperature)
        return int(np.argmax(policy))

    def policy_distribution(
        self,
        env: GomokuEnv,
        temperature: float,
    ) -> np.ndarray:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")

        legal = env.legal_actions()
        if not legal:
            return np.zeros((81,), dtype=np.float32)

        state = encode_board(env.board, env.current_player)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            masked_logits = mask_illegal_logits(logits, legal)
            scaled = masked_logits / float(temperature)
            probs = torch.softmax(scaled, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

        probs_sum = float(np.sum(probs))
        if not np.isfinite(probs_sum) or probs_sum <= 0.0:
            # Extremely unlikely, but keeps generation numerically robust.
            probs = np.zeros((81,), dtype=np.float32)
            probs[legal] = 1.0 / float(len(legal))
            return probs

        return probs / probs_sum


def generate_selfplay_data(
    num_games: int,
    noise: float,
    save_soft_targets: bool,
    teacher_temperature: float,
    teacher_hard_mix: float,
    teacher_checkpoint: Path | None = None,
    teacher_device: str = "cpu",
    teacher_noise: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Run self-play games and return `(states, actions, policy_targets, outcomes)`.
    """
    if num_games <= 0:
        raise ValueError("num_games must be > 0.")
    if teacher_temperature <= 0.0:
        raise ValueError("teacher_temperature must be > 0.")
    if not (0.0 <= teacher_hard_mix <= 1.0):
        raise ValueError("teacher_hard_mix must be in [0, 1].")
    if not (0.0 <= teacher_noise <= 1.0):
        raise ValueError("teacher_noise must be in [0, 1].")

    master_rng = np.random.default_rng(seed)
    black_seed = int(master_rng.integers(0, 2**31 - 1))
    white_seed = int(master_rng.integers(0, 2**31 - 1))

    model_teacher: _ModelTeacher | None = None
    fallback_to_heuristic = False
    if teacher_checkpoint is not None:
        try:
            model_teacher = _ModelTeacher(
                checkpoint_path=teacher_checkpoint,
                device=teacher_device,
                noise=teacher_noise,
                seed=seed,
            )
            print(
                "Using model teacher: "
                f"{teacher_checkpoint.resolve()} on {model_teacher.device}"
            )
        except Exception as exc:
            print(
                f"Model teacher load failed ({exc}); falling back to heuristic agents "
                f"with noise={teacher_noise}."
            )
            fallback_to_heuristic = True
    if model_teacher is None:
        fallback_to_heuristic = True

    black_agent = HeuristicAgent(noise=noise, seed=black_seed)
    white_agent = HeuristicAgent(noise=noise, seed=white_seed)
    if fallback_to_heuristic and teacher_checkpoint is not None:
        # Keep heuristic fallback noise aligned with teacher-noise intent.
        black_agent.noise = max(noise, teacher_noise)
        white_agent.noise = max(noise, teacher_noise)

    states: list[np.ndarray] = []
    actions: list[int] = []
    policy_targets: list[np.ndarray] = []
    outcomes: list[int] = []

    print(
        "Starting self-play generation: "
        f"games={num_games}, noise={noise}, seed={seed}, "
        f"teacher_checkpoint={teacher_checkpoint}"
    )
    progress_every = max(1, num_games // 10)

    for game_idx in range(num_games):
        env = GomokuEnv()
        env.reset()
        game_players: list[int] = []

        while not env.is_terminal():
            state = encode_board(env.board, env.current_player)
            actor = (
                model_teacher
                if (model_teacher is not None and not fallback_to_heuristic)
                else (black_agent if env.current_player == 1 else white_agent)
            )
            game_players.append(int(env.current_player))
            if isinstance(actor, _ModelTeacher):
                action = actor.select_action(
                    env,
                    temperature=teacher_temperature,
                )
            else:
                # Heuristic teacher uses a deterministic policy when temperature=1.
                action = actor.select_action(env)

            if save_soft_targets:
                if isinstance(actor, _ModelTeacher):
                    teacher_policy = actor.policy_distribution(
                        env,
                        temperature=teacher_temperature,
                    )
                else:
                    teacher_policy = actor.policy_distribution(
                        env,
                        temperature=teacher_temperature,
                    )
                if teacher_hard_mix > 0.0:
                    hard = np.zeros((81,), dtype=np.float32)
                    hard[action] = 1.0
                    teacher_policy = (
                        (1.0 - teacher_hard_mix) * teacher_policy
                        + teacher_hard_mix * hard
                    ).astype(np.float32)
                    teacher_policy /= float(np.sum(teacher_policy))
                policy_targets.append(teacher_policy)

            # Supervised sample (s_t, a_t):
            # s_t is the 3x9x9 encoded state, a_t in {0,...,80} is the
            # heuristic policy target for this position.
            states.append(state)
            actions.append(action)
            env.step(action)

        winner = env.winner
        if winner is None:
            winner = 0
        if winner not in (1, -1, 0):
            raise ValueError(f"Unexpected winner value: {winner!r}.")

        if winner == 0:
            outcomes.extend([0] * len(game_players))
        else:
            outcomes.extend([1 if p == winner else -1 for p in game_players])

        if (game_idx + 1) % progress_every == 0 or (game_idx + 1) == num_games:
            print(
                f"Completed {game_idx + 1}/{num_games} games | "
                f"samples={len(actions)}"
            )

    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.int64)
    policy_arr = (
        np.asarray(policy_targets, dtype=np.float32) if save_soft_targets else None
    )
    outcomes_arr = np.asarray(outcomes, dtype=np.int8)
    return states_arr, actions_arr, policy_arr, outcomes_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gomoku supervised labels from heuristic self-play."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=200,
        help="Number of self-play games to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selfplay_dataset.npz"),
        help="Output .npz path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Small heuristic noise (e.g. 0.05) for data diversity.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional policy checkpoint used as teacher. If set, generates policy from model.",
    )
    parser.add_argument(
        "--teacher-device",
        type=str,
        default="cpu",
        help="Device for teacher model inference (cpu or cuda).",
    )
    parser.add_argument(
        "--teacher-noise",
        type=float,
        default=0.0,
        help=(
            "Optional teacher exploration probability in [0,1] (epsilon-greedy fallback). "
            "Also applied to heuristic fallback when teacher fails to load."
        ),
    )
    parser.add_argument(
        "--save-soft-targets",
        action="store_true",
        help="Save teacher policy distributions as `policy_targets` in the dataset.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for teacher policy generation.",
    )
    parser.add_argument(
        "--teacher-hard-mix",
        type=float,
        default=0.15,
        help=(
            "Mixture weight in [0,1] for blending one-hot chosen action into "
            "teacher soft targets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states, actions, policy_targets, outcomes = generate_selfplay_data(
        num_games=args.num_games,
        noise=args.noise,
        save_soft_targets=args.save_soft_targets,
        teacher_temperature=args.teacher_temperature,
        teacher_hard_mix=args.teacher_hard_mix,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_device=args.teacher_device,
        teacher_noise=args.teacher_noise,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_payload: dict[str, np.ndarray] = {"states": states, "actions": actions, "outcomes": outcomes}
    if policy_targets is not None:
        save_payload["policy_targets"] = policy_targets
    np.savez_compressed(args.output, **save_payload)
    print(f"Saved dataset to: {args.output.resolve()}")
    print(f"Final dataset size: {len(actions)} moves")
    if policy_targets is not None:
        print("Saved soft teacher targets: policy_targets shape="
              f"{tuple(policy_targets.shape)}")


if __name__ == "__main__":
    main()
