"""Terminal entrypoint for Gomoku matches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from src.agents.heuristic_agent import HeuristicAgent
from src.data.game_logging import GameDataLogger
from src.env.gomoku_env import GomokuEnv


MoveSelector = Callable[[GomokuEnv], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play 9x9 Gomoku in terminal or popup GUI."
    )
    parser.add_argument(
        "--ui",
        type=str,
        choices=["terminal", "gui"],
        default="terminal",
        help="Interface mode: terminal text UI or Tkinter popup GUI.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["human-vs-heuristic", "human-vs-model", "heuristic-vs-model"],
        help="Game mode.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("checkpoints/policy_net.pt"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--human-player",
        type=str,
        choices=["black", "white"],
        default="black",
        help="Which side the human controls in human modes.",
    )
    parser.add_argument(
        "--model-player",
        type=str,
        choices=["black", "white"],
        default="white",
        help="Which side the model controls in heuristic-vs-model mode.",
    )
    parser.add_argument(
        "--heuristic-noise",
        type=float,
        default=0.0,
        help="Optional heuristic randomness for non-human agents.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for model mode (cpu/cuda/cuda:0).",
    )
    parser.add_argument(
        "--log-data",
        type=Path,
        default=None,
        help=(
            "Optional .npz path for logging played samples "
            "(states/actions for training)."
        ),
    )
    parser.add_argument(
        "--log-policy",
        type=str,
        choices=["human", "all"],
        default="human",
        help="Log only human moves or all moves.",
    )
    parser.add_argument(
        "--overwrite-log",
        action="store_true",
        help="Overwrite log file instead of appending.",
    )
    return parser.parse_args()


def player_name(player: int) -> str:
    return "Black (X)" if player == 1 else "White (O)"


def parse_human_action(env: GomokuEnv) -> int:
    while True:
        raw = input(f"{player_name(env.current_player)} move as row,col: ").strip()
        if raw.lower() in {"quit", "exit"}:
            raise KeyboardInterrupt("User exited game.")

        parts = raw.split(",")
        if len(parts) != 2:
            print("Invalid format. Please use row,col (example: 4,4).")
            continue

        try:
            row = int(parts[0].strip())
            col = int(parts[1].strip())
            action = env.rc_to_action(row, col)
        except ValueError as exc:
            print(f"Invalid move: {exc}")
            continue

        if action not in env.legal_actions():
            print(f"Illegal move at ({row}, {col}): cell already occupied.")
            continue

        return action


def run_game(
    black_selector: MoveSelector,
    white_selector: MoveSelector,
    black_actor: str,
    white_actor: str,
    logger: GameDataLogger | None = None,
) -> None:
    env = GomokuEnv()
    env.reset()
    if logger is not None:
        logger.start_game()

    print("Starting game. Type 'quit' to exit.")
    env.render()

    while not env.is_terminal():
        selector = black_selector if env.current_player == 1 else white_selector
        actor = black_actor if env.current_player == 1 else white_actor
        action = selector(env)
        if logger is not None:
            # Logged sample is from pre-move state s_t and selected label a_t.
            logger.record(env.board, env.current_player, action, actor)
        row, col = env.action_to_rc(action)
        print(f"{player_name(env.current_player)} plays: {row},{col}")
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            break

    if logger is not None and env.winner is not None:
        logger.finalize_game(env.winner)

    if env.winner == 0:
        print("Game result: Draw.")
    elif env.winner == 1:
        print("Game result: Black (X) wins.")
    elif env.winner == -1:
        print("Game result: White (O) wins.")
    else:
        print("Game ended unexpectedly without a result.")


def main() -> None:
    args = parse_args()

    if args.ui == "gui":
        from src.ui.gomoku_gui import run_gui

        run_gui(args)
        return

    heuristic_agent = HeuristicAgent(noise=args.heuristic_noise)
    logger = GameDataLogger(log_policy=args.log_policy) if args.log_data else None

    if args.mode == "human-vs-heuristic":
        if args.human_player == "black":
            black_selector = parse_human_action
            white_selector = heuristic_agent.select_action
            black_actor, white_actor = "human", "heuristic"
        else:
            black_selector = heuristic_agent.select_action
            white_selector = parse_human_action
            black_actor, white_actor = "heuristic", "human"
        try:
            run_game(
                black_selector,
                white_selector,
                black_actor=black_actor,
                white_actor=white_actor,
                logger=logger,
            )
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
        finally:
            maybe_save_logs(logger, args.log_data, overwrite=args.overwrite_log)
        return

    if args.mode == "human-vs-model":
        from src.agents.model_agent import ModelAgent

        model_agent = ModelAgent(args.model_path, device=args.device)
        if args.human_player == "black":
            black_selector = parse_human_action
            white_selector = model_agent.select_action
            black_actor, white_actor = "human", "model"
        else:
            black_selector = model_agent.select_action
            white_selector = parse_human_action
            black_actor, white_actor = "model", "human"
        try:
            run_game(
                black_selector,
                white_selector,
                black_actor=black_actor,
                white_actor=white_actor,
                logger=logger,
            )
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
        finally:
            maybe_save_logs(logger, args.log_data, overwrite=args.overwrite_log)
        return

    if args.mode == "heuristic-vs-model":
        from src.agents.model_agent import ModelAgent

        model_agent = ModelAgent(args.model_path, device=args.device)
        if args.model_player == "black":
            black_selector = model_agent.select_action
            white_selector = heuristic_agent.select_action
            black_actor, white_actor = "model", "heuristic"
        else:
            black_selector = heuristic_agent.select_action
            white_selector = model_agent.select_action
            black_actor, white_actor = "heuristic", "model"
        try:
            run_game(
                black_selector,
                white_selector,
                black_actor=black_actor,
                white_actor=white_actor,
                logger=logger,
            )
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
        finally:
            maybe_save_logs(logger, args.log_data, overwrite=args.overwrite_log)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


def maybe_save_logs(
    logger: GameDataLogger | None,
    output_path: Path | None,
    overwrite: bool,
) -> None:
    if logger is None or output_path is None:
        return

    num_added, total = logger.save(output_path, append=not overwrite)
    if num_added == 0:
        print(
            "No samples were logged for this run. "
            "If you are in non-human mode, use --log-policy all."
        )
        return

    mode = "overwrote" if overwrite else "appended"
    print(
        f"Saved {num_added} samples ({mode}) -> {output_path.resolve()} "
        f"| total_samples={total}"
    )


if __name__ == "__main__":
    main()
