"""Tkinter popup GUI for 9x9 Gomoku."""

from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Optional

from src.agents.heuristic_agent import HeuristicAgent
from src.data.game_logging import GameDataLogger
from src.env.gomoku_env import GomokuEnv


MoveSelector = Callable[[GomokuEnv], int]


@dataclass
class PlayerController:
    """Controller for one side of play."""

    actor: str
    selector: Optional[MoveSelector] = None

    @property
    def is_human(self) -> bool:
        return self.selector is None


class GomokuGUI:
    """Interactive popup GUI with styled board and click-to-play controls."""

    def __init__(
        self,
        black: PlayerController,
        white: PlayerController,
        mode_name: str,
        initial_human_player: str,
        alternate_controllers: tuple[PlayerController, PlayerController] | None = None,
        logger: GameDataLogger | None = None,
        log_path: Path | None = None,
        overwrite_log: bool = False,
    ) -> None:
        self.env = GomokuEnv()
        self.mode_name = mode_name
        self.black = black
        self.white = white
        self.human_player = initial_human_player
        self.controller_pairs = {"black": (black, white)}
        if alternate_controllers is not None:
            self.controller_pairs["white"] = alternate_controllers
        if self.human_player not in self.controller_pairs:
            self.human_player = "black"
        self.side_var: tk.StringVar | None = None

        self.logger = logger
        self.log_path = log_path
        self.overwrite_log = overwrite_log

        self.move_history: list[int] = []
        self.ai_thinking = False
        self.completed_games = 0

        self.board_size = self.env.board_size
        self.cell = 64
        self.margin = 72
        self.board_pixels = (self.board_size - 1) * self.cell
        self.canvas_size = self.board_pixels + 2 * self.margin
        self.sidebar_width = 290
        self.window_width = self.canvas_size + self.sidebar_width
        self.window_height = self.canvas_size + 24

        self.root = tk.Tk()
        self.root.title("Gomoku 9x9")
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.configure(bg="#1f2126")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._new_game()

    def _build_layout(self) -> None:
        left = tk.Frame(self.root, bg="#1f2126")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(14, 8), pady=12)

        right = tk.Frame(self.root, bg="#272b31")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 14), pady=12)

        self.canvas = tk.Canvas(
            left,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#d9d9d9",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        title = tk.Label(
            right,
            text="Gomoku",
            fg="#f5f5f5",
            bg="#272b31",
            font=("Segoe UI Semibold", 21),
        )
        title.pack(anchor="w", pady=(8, 4), padx=12)

        subtitle = tk.Label(
            right,
            text=f"Mode: {self.mode_name}",
            fg="#b7c6d6",
            bg="#272b31",
            font=("Segoe UI", 11),
        )
        subtitle.pack(anchor="w", padx=12, pady=(0, 14))

        self.turn_var = tk.StringVar(value="Turn: Black")
        self.status_var = tk.StringVar(value="Click an intersection to play.")
        self.result_var = tk.StringVar(value="Result: In progress")
        self.data_var = tk.StringVar(value="Data: logging disabled")

        if self.log_path is not None:
            self.data_var.set(f"Data file: {self.log_path}")

        self._make_info_row(right, self.turn_var, "#f4f0c9")
        self._make_info_row(right, self.status_var, "#c5d4e3")
        self._make_info_row(right, self.result_var, "#f4f0c9")
        self._make_info_row(right, self.data_var, "#9fb2c3")

        if len(self.controller_pairs) > 1:
            controls_frame = tk.LabelFrame(
                right,
                text="Human Side",
                bg="#2f353d",
                fg="#cfd7e0",
                font=("Segoe UI Semibold", 10),
                padx=8,
                pady=8,
            )
            controls_frame.pack(anchor="w", padx=12, pady=(0, 10), fill=tk.X)

            self.side_var = tk.StringVar(value=self.human_player)
            tk.Radiobutton(
                controls_frame,
                text="Play as Black",
                variable=self.side_var,
                value="black",
                command=self._on_side_change,
                bg="#2f353d",
                fg="#f4f0c9",
                selectcolor="#1f2126",
                activebackground="#404857",
                activeforeground="#ffffff",
                font=("Segoe UI", 10),
            ).pack(anchor="w")
            tk.Radiobutton(
                controls_frame,
                text="Play as White",
                variable=self.side_var,
                value="white",
                command=self._on_side_change,
                bg="#2f353d",
                fg="#f4f0c9",
                selectcolor="#1f2126",
                activebackground="#404857",
                activeforeground="#ffffff",
                font=("Segoe UI", 10),
            ).pack(anchor="w")
            tk.Label(
                controls_frame,
                text="Changing side restarts the current game.",
                fg="#9fb2c3",
                bg="#2f353d",
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(6, 0))

        button_row = tk.Frame(right, bg="#272b31")
        button_row.pack(anchor="w", pady=(14, 8), padx=12)

        new_btn = tk.Button(
            button_row,
            text="New Game",
            command=self._new_game,
            bg="#f2cc3d",
            fg="#1c1c1c",
            activebackground="#ffdc61",
            activeforeground="#111111",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            font=("Segoe UI Semibold", 10),
        )
        new_btn.pack(side=tk.LEFT, padx=(0, 8))

        save_btn = tk.Button(
            button_row,
            text="Save Logs",
            command=self._persist_logs,
            bg="#4a85b8",
            fg="#ffffff",
            activebackground="#5d99cc",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            font=("Segoe UI Semibold", 10),
        )
        save_btn.pack(side=tk.LEFT, padx=(0, 8))

        quit_btn = tk.Button(
            button_row,
            text="Quit",
            command=self._on_close,
            bg="#3f4146",
            fg="#f5f5f5",
            activebackground="#50535a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            font=("Segoe UI Semibold", 10),
        )
        quit_btn.pack(side=tk.LEFT)

        hint_lines = [
            "Black = solid dark stones",
            "White = light stones",
            "Red ring marks last move",
            "Move numbers shown on stones",
        ]
        hint = tk.Label(
            right,
            text="\n".join(hint_lines),
            justify=tk.LEFT,
            fg="#9fb2c3",
            bg="#272b31",
            font=("Segoe UI", 10),
        )
        hint.pack(anchor="w", padx=12, pady=(6, 0))

    def _make_info_row(self, parent: tk.Widget, text_var: tk.StringVar, color: str) -> None:
        lbl = tk.Label(
            parent,
            textvariable=text_var,
            fg=color,
            bg="#272b31",
            justify=tk.LEFT,
            wraplength=250,
            font=("Segoe UI", 11),
        )
        lbl.pack(anchor="w", padx=12, pady=3)

    def _on_side_change(self) -> None:
        if self.side_var is None:
            return

        requested_side = self.side_var.get()
        if requested_side not in self.controller_pairs or requested_side == self.human_player:
            return

        self.human_player = requested_side
        self.black, self.white = self.controller_pairs[requested_side]
        self._new_game()

    def _star_points(self) -> list[tuple[int, int]]:
        # Standard star point pattern adapted to 9x9 board.
        return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]

    def _rc_to_xy(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.margin + col * self.cell,
            self.margin + row * self.cell,
        )

    def _closest_rc(self, x: float, y: float) -> tuple[int, int] | None:
        col = round((x - self.margin) / self.cell)
        row = round((y - self.margin) / self.cell)
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return None

        cx, cy = self._rc_to_xy(row, col)
        max_dist = self.cell * 0.43
        if (x - cx) ** 2 + (y - cy) ** 2 > max_dist**2:
            return None
        return row, col

    def _controller_for_current_player(self) -> PlayerController:
        return self.black if self.env.current_player == 1 else self.white

    def _player_label(self, player: int) -> str:
        return "Black" if player == 1 else "White"

    def _draw_board(self) -> None:
        c = self.canvas
        c.delete("all")

        board_x0 = self.margin
        board_y0 = self.margin
        board_x1 = self.margin + self.board_pixels
        board_y1 = self.margin + self.board_pixels

        # Background slab and playable board.
        c.create_rectangle(
            0,
            0,
            self.canvas_size,
            self.canvas_size,
            fill="#d4d4d4",
            outline="",
        )
        c.create_rectangle(
            board_x0 - 24,
            board_y0 - 24,
            board_x1 + 24,
            board_y1 + 24,
            fill="#e6c84f",
            outline="",
        )
        c.create_rectangle(
            board_x0,
            board_y0,
            board_x1,
            board_y1,
            fill="#e2bf45",
            outline="#14110f",
            width=3,
        )

        for idx in range(self.board_size):
            x = self.margin + idx * self.cell
            y = self.margin + idx * self.cell
            line_w = 2 if idx in (0, self.board_size - 1) else 1
            c.create_line(x, board_y0, x, board_y1, fill="#1e160e", width=line_w)
            c.create_line(board_x0, y, board_x1, y, fill="#1e160e", width=line_w)

        # Coordinates on all sides.
        letters = [chr(ord("A") + i) for i in range(self.board_size)]
        for i, letter in enumerate(letters):
            x = self.margin + i * self.cell
            c.create_text(
                x,
                board_y1 + 34,
                text=letter,
                fill="#161616",
                font=("Segoe UI", 16),
            )
            c.create_text(
                x,
                board_y0 - 34,
                text=letter,
                fill="#161616",
                font=("Segoe UI", 14),
            )

        for i in range(self.board_size):
            y = self.margin + i * self.cell
            label = str(self.board_size - i)
            c.create_text(
                board_x0 - 34,
                y,
                text=label,
                fill="#161616",
                font=("Segoe UI", 16),
            )
            c.create_text(
                board_x1 + 34,
                y,
                text=label,
                fill="#161616",
                font=("Segoe UI", 16),
            )

        for r, ccol in self._star_points():
            x, y = self._rc_to_xy(r, ccol)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#100f0d", outline="")

        self._draw_stones()

    def _draw_stones(self) -> None:
        index_by_action = {action: idx + 1 for idx, action in enumerate(self.move_history)}
        last_action = self.move_history[-1] if self.move_history else None

        radius = self.cell * 0.41

        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = int(self.env.board[row, col])
                if stone == 0:
                    continue

                action = self.env.rc_to_action(row, col)
                move_number = index_by_action.get(action)
                x, y = self._rc_to_xy(row, col)

                self.canvas.create_oval(
                    x - radius + 2,
                    y - radius + 2,
                    x + radius + 2,
                    y + radius + 2,
                    fill="#737373",
                    outline="",
                )

                if stone == 1:
                    fill = "#111111"
                    outline = "#050505"
                    highlight = "#3c3c3c"
                    text_color = "#f5f5f5"
                else:
                    fill = "#efefef"
                    outline = "#1f1f1f"
                    highlight = "#ffffff"
                    text_color = "#111111"

                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline=outline,
                    width=1.2,
                )
                self.canvas.create_oval(
                    x - radius * 0.48,
                    y - radius * 0.62,
                    x - radius * 0.05,
                    y - radius * 0.20,
                    fill=highlight,
                    outline="",
                )

                if move_number is not None:
                    self.canvas.create_text(
                        x,
                        y,
                        text=str(move_number),
                        fill=text_color,
                        font=("Segoe UI Semibold", 15),
                    )

                if action == last_action:
                    ring = radius + 4
                    self.canvas.create_oval(
                        x - ring,
                        y - ring,
                        x + ring,
                        y + ring,
                        outline="#e64235",
                        width=2.2,
                    )

    def _refresh_labels(self) -> None:
        if self.env.is_terminal():
            self.turn_var.set("Turn: -")
            if self.env.winner == 0:
                self.result_var.set("Result: Draw")
            elif self.env.winner == 1:
                self.result_var.set("Result: Black wins")
            else:
                self.result_var.set("Result: White wins")
            return

        self.turn_var.set(f"Turn: {self._player_label(self.env.current_player)}")
        self.result_var.set("Result: In progress")

    def _new_game(self) -> None:
        if self.logger is not None:
            self.logger.finalize_game(None)
        self.env.reset()
        self.move_history.clear()
        self.ai_thinking = False
        if self.logger is not None:
            self.logger.start_game()
        self.status_var.set(
            f"New game started (Game #{self.completed_games + 1})."
        )
        self._refresh_labels()
        self._draw_board()
        self._schedule_turn_if_ai()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.env.is_terminal() or self.ai_thinking:
            return

        controller = self._controller_for_current_player()
        if not controller.is_human:
            return

        rc = self._closest_rc(event.x, event.y)
        if rc is None:
            self.status_var.set("Click closer to a board intersection.")
            return

        row, col = rc
        action = self.env.rc_to_action(row, col)
        if action not in self.env.legal_actions():
            self.status_var.set(f"Illegal move at ({row}, {col}); cell occupied.")
            return

        self._apply_action(action=action, actor=controller.actor)

    def _apply_action(self, action: int, actor: str) -> None:
        if self.logger is not None:
            self.logger.record(self.env.board, self.env.current_player, action, actor)

        self.env.step(action)
        self.move_history.append(action)
        row, col = self.env.action_to_rc(action)
        self.status_var.set(f"{actor.title()} played at ({row}, {col}).")

        self._refresh_labels()
        self._draw_board()

        if self.env.is_terminal():
            self.completed_games += 1
            if self.logger is not None:
                self.logger.finalize_game(self.env.winner)
            if self.logger is not None and self.log_path is not None:
                self.status_var.set(
                    f"Game #{self.completed_games} finished. "
                    "Click Save Logs to store training data."
                )
            else:
                self.status_var.set(
                    f"Game #{self.completed_games} finished. Start a new game or quit."
                )
            return
        self._schedule_turn_if_ai()

    def _schedule_turn_if_ai(self) -> None:
        if self.env.is_terminal():
            return

        controller = self._controller_for_current_player()
        if controller.is_human:
            self.ai_thinking = False
            self.status_var.set(
                f"{self._player_label(self.env.current_player)} to move. Click a point."
            )
            return

        self.ai_thinking = True
        self.status_var.set(f"{controller.actor.title()} is thinking...")
        self.root.after(170, self._run_ai_turn)

    def _run_ai_turn(self) -> None:
        if self.env.is_terminal():
            self.ai_thinking = False
            return

        controller = self._controller_for_current_player()
        if controller.is_human or controller.selector is None:
            self.ai_thinking = False
            return

        try:
            action = controller.selector(self.env)
        except Exception as exc:  # pragma: no cover - GUI runtime error path
            self.ai_thinking = False
            messagebox.showerror("AI Error", str(exc))
            return

        self.ai_thinking = False
        self._apply_action(action=action, actor=controller.actor)

    def _persist_logs(self, from_close: bool = False) -> None:
        if self.logger is None or self.log_path is None:
            if not from_close:
                messagebox.showinfo(
                    "Logging Disabled",
                    "No log file is configured. Relaunch with --log-data to enable.",
                )
            return

        append = not self.overwrite_log
        try:
            num_added, total = self.logger.save(self.log_path, append=append)
        except Exception as exc:  # pragma: no cover - GUI runtime error path
            messagebox.showerror("Save Error", str(exc))
            return

        if num_added == 0:
            self.data_var.set("Data: no new unsaved samples.")
            if not from_close and self.env.is_terminal():
                self._prompt_replay_or_quit()
            return

        mode = "overwrote" if self.overwrite_log else "appended"
        self.data_var.set(f"Data: {mode} +{num_added} samples, total={total}")

        # Clear in-memory cache after save to avoid duplicate re-appends.
        self.logger.clear_buffer()

        if not from_close and self.env.is_terminal():
            messagebox.showinfo(
                "Logs Saved",
                f"game:{self.completed_games} training game has been saved.",
            )
            self._prompt_replay_or_quit()

    def _prompt_replay_or_quit(self) -> None:
        replay = messagebox.askyesno(
            "Next Action",
            "Do you want to replay?\n\nYes = Replay\nNo = Quit",
        )
        if replay:
            self._new_game()
        else:
            self._on_close()

    def _on_close(self) -> None:
        self._persist_logs(from_close=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _build_controllers(
    args: argparse.Namespace,
    human_player: str = "black",
    heuristic_agent: HeuristicAgent | None = None,
    model_agent: object | None = None,
) -> tuple[PlayerController, PlayerController]:
    if heuristic_agent is None:
        heuristic_agent = HeuristicAgent(noise=args.heuristic_noise)

    if args.mode == "human-vs-heuristic":
        if human_player == "black":
            return (
                PlayerController(actor="human", selector=None),
                PlayerController(actor="heuristic", selector=heuristic_agent.select_action),
            )
        return (
            PlayerController(actor="heuristic", selector=heuristic_agent.select_action),
            PlayerController(actor="human", selector=None),
        )

    if args.mode == "human-vs-model":
        if model_agent is None:
            from src.agents.model_agent import ModelAgent

            model_agent = ModelAgent(args.model_path, device=args.device)
        if human_player == "black":
            return (
                PlayerController(actor="human", selector=None),
                PlayerController(actor="model", selector=model_agent.select_action),
            )
        return (
            PlayerController(actor="model", selector=model_agent.select_action),
            PlayerController(actor="human", selector=None),
        )

    if args.mode == "heuristic-vs-model":
        if model_agent is None:
            from src.agents.model_agent import ModelAgent

            model_agent = ModelAgent(args.model_path, device=args.device)
        if args.model_player == "black":
            return (
                PlayerController(actor="model", selector=model_agent.select_action),
                PlayerController(actor="heuristic", selector=heuristic_agent.select_action),
            )
        return (
            PlayerController(actor="heuristic", selector=heuristic_agent.select_action),
            PlayerController(actor="model", selector=model_agent.select_action),
        )

    raise ValueError(f"Unsupported mode: {args.mode}")


def run_gui(args: argparse.Namespace) -> None:
    """Entry point used by `main.py --ui gui`."""
    logger = GameDataLogger(log_policy=args.log_policy) if args.log_data else None

    heuristic_agent = HeuristicAgent(noise=args.heuristic_noise)
    model_agent = None
    if args.mode in {"human-vs-model", "heuristic-vs-model"}:
        from src.agents.model_agent import ModelAgent

        model_agent = ModelAgent(args.model_path, device=args.device)

    black, white = _build_controllers(
        args=args,
        human_player=args.human_player,
        heuristic_agent=heuristic_agent,
        model_agent=model_agent,
    )

    alternate = None
    if args.mode in {"human-vs-model", "human-vs-heuristic"}:
        alt_human = "white" if args.human_player == "black" else "black"
        alt_black, alt_white = _build_controllers(
            args=args,
            human_player=alt_human,
            heuristic_agent=heuristic_agent,
            model_agent=model_agent,
        )
        alternate = (alt_black, alt_white)

    app = GomokuGUI(
        black=black,
        white=white,
        mode_name=args.mode,
        initial_human_player=args.human_player,
        alternate_controllers=alternate,
        logger=logger,
        log_path=args.log_data,
        overwrite_log=args.overwrite_log,
    )
    app.run()
