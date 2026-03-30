"""UCI (Universal Chess Interface) wrapper for the ChessGNN engine.

The engine communicates over stdin/stdout using the UCI protocol.  Supported
commands:

    uci            → id + option lines + uciok
    isready        → readyok
    ucinewgame     → reset hidden state
    position ...   → set current position (startpos or FEN, optional move list)
    go ...         → bestmove (searchless, argmax Q-head)
    stop           → ignored (no background thread; go is synchronous)
    quit           → exit

Usage
-----
    python uci_engine.py
    python uci_engine.py --model output/gateau_distilled.pt --calib output/gateau_distilled.pt.calib.json

The engine loads a GATEAUChessModel by default.  Pass --model to override the
checkpoint path; pass --calib to enable temperature-scaled win-probability
reporting in the info line.
"""

import argparse
import logging
import math
import sys

import chess
import chess.polyglot
import torch

from chessgnn.calibration import TemperatureScaler
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel
from tutor import CaseTutor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "output/gateau_distilled.pt"
DEFAULT_BOOK_PATH = "input/chessgnn.bin"
ENGINE_NAME = "ChessGNN"
ENGINE_AUTHOR = "chessgnn"

logging.basicConfig(
    filename="output/uci_engine.log",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_STARTPOS_FEN = chess.STARTING_FEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(model_path: str, device: torch.device) -> GATEAUChessModel:
    """Load a GATEAUChessModel from checkpoint, inferring hidden_channels."""
    ckpt = torch.load(model_path, map_location=device)

    # Infer hidden_channels from GRU weight shape: [3*H, input] → H = shape[0]//3
    if "global_gru.weight_ih_l0" in ckpt:
        hidden_channels = ckpt["global_gru.weight_ih_l0"].shape[0] // 3
    else:
        # Fallback: k_lin projection weight [H, in_features]
        hidden_channels = next(
            v.shape[0]
            for k, v in ckpt.items()
            if k.startswith("convs.0.k_lin.")
        )

    num_layers = max(int(k.split(".")[1]) for k in ckpt if k.startswith("convs.")) + 1

    builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
    model = GATEAUChessModel(
        builder.get_metadata(),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        temporal_mode="global_gru",
    )
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    logger.info(
        "Loaded GATEAUChessModel from %s (hidden=%d, layers=%d)",
        model_path,
        hidden_channels,
        num_layers,
    )
    return model


def _win_prob_to_cp(win_prob_pct: float) -> int:
    """Convert win probability (0–100 %) to centipawns (Stockfish convention).

    Uses the inverse of the logistic win-probability formula:
        cp = round(111.714 * tanh(1.5620 * (p - 0.5)))
    Clamped to ±30000 to stay within 16-bit signed integer range.
    """
    p = max(0.001, min(0.999, win_prob_pct / 100.0))
    cp = round(111.714 * math.tanh(1.5620 * (p - 0.5)))
    return max(-30000, min(30000, cp))


def _parse_position(tokens: list[str]):
    """Parse a `position` command token list (excluding the leading 'position').

    Returns
    -------
    board : chess.Board
        Board after all moves have been applied.
    history_fens : list[str]
        FEN after each incremental move (used to advance GRU state).
    """
    if not tokens:
        return chess.Board(), []

    if tokens[0] == "startpos":
        board = chess.Board()
        move_start = 1
    elif tokens[0] == "fen":
        # FEN is next 4–6 tokens; "moves" keyword marks the end
        try:
            moves_idx = tokens.index("moves")
            fen_tokens = tokens[1:moves_idx]
            move_start = moves_idx
        except ValueError:
            fen_tokens = tokens[1:]
            move_start = len(tokens)
        board = chess.Board(" ".join(fen_tokens))
    else:
        logger.warning("Unknown position format: %s", tokens)
        return chess.Board(), []

    # Skip "moves" keyword
    if move_start < len(tokens) and tokens[move_start] == "moves":
        move_start += 1

    history_fens: list[str] = []
    for uci_str in tokens[move_start:]:
        try:
            move = chess.Move.from_uci(uci_str)
            board.push(move)
            history_fens.append(board.fen())
        except (ValueError, chess.InvalidMoveError):
            logger.warning("Skipping invalid move: %s", uci_str)

    return board, history_fens


# ---------------------------------------------------------------------------
# UCI Engine
# ---------------------------------------------------------------------------


class UCIEngine:
    """Reads UCI commands from stdin and dispatches to the GNN engine."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device_str: str = "cpu",
        temperature_path: str | None = None,
        book_path: str | None = DEFAULT_BOOK_PATH,
    ) -> None:
        self._device = torch.device(device_str)
        model = _load_model(model_path, self._device)
        self._tutor = CaseTutor(model, self._device)

        self._scaler: TemperatureScaler | None = None
        if temperature_path is not None:
            self._scaler = TemperatureScaler()
            self._scaler.load(temperature_path)
            self._tutor.set_calibration(self._scaler)
            logger.info("Calibration loaded from %s (T=%.4f)", temperature_path, self._scaler.T)

        self._book_path: str | None = book_path
        if book_path is not None:
            import os
            if os.path.isfile(book_path):
                logger.info("Opening book: %s", book_path)
            else:
                logger.warning("Opening book not found at %s — book disabled", book_path)
                self._book_path = None

        self._current_board: chess.Board = chess.Board()
        self._current_fen: str = _STARTPOS_FEN

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_uci(self) -> None:
        print(f"id name {ENGINE_NAME}")
        print(f"id author {ENGINE_AUTHOR}")
        print("option name Hash type spin default 1 min 1 max 1")
        print("uciok")
        sys.stdout.flush()

    def _handle_isready(self) -> None:
        print("readyok")
        sys.stdout.flush()

    def _handle_ucinewgame(self) -> None:
        self._tutor.reset()
        self._current_board = chess.Board()
        self._current_fen = _STARTPOS_FEN
        logger.debug("New game: hidden state reset")

    def _handle_position(self, tokens: list[str]) -> None:
        board, history_fens = _parse_position(tokens)
        # Advance GRU state for each committed move in history
        for fen in history_fens:
            self._tutor.update_state(fen)
        self._current_board = board
        self._current_fen = board.fen()
        logger.debug("Position set: %s", self._current_fen)

    def _book_move(self) -> chess.Move | None:
        if self._book_path is None:
            return None
        try:
            with chess.polyglot.open_reader(self._book_path) as reader:
                entries = list(reader.find_all(self._current_board))
            if not entries:
                return None
            # Weighted random selection proportional to book weight
            total = sum(e.weight for e in entries)
            import random
            r = random.randint(0, total - 1)
            cumulative = 0
            for entry in entries:
                cumulative += entry.weight
                if r < cumulative:
                    return entry.move
            return entries[-1].move
        except Exception as exc:
            logger.warning("Book lookup failed: %s", exc)
            return None

    def _handle_go(self) -> None:
        book_move = self._book_move()
        if book_move is not None:
            logger.debug("go → book move %s", book_move.uci())
            print(f"info string book move")
            print(f"bestmove {book_move.uci()}")
            sys.stdout.flush()
            return

        result = self._tutor.recommend_move(self._current_fen)
        best_move, best_prob, _ranking, uncertainty = result

        if best_move is None:
            print("bestmove 0000")
            sys.stdout.flush()
            return

        cp = _win_prob_to_cp(best_prob)
        move_uci = best_move.uci()

        logger.debug(
            "go → bestmove %s  cp=%d  win_prob=%.1f%%  uncertainty=%.3f",
            move_uci, cp, best_prob, uncertainty,
        )
        print(f"info score cp {cp} pv {move_uci}")
        print(f"bestmove {move_uci}")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Read UCI commands from stdin until 'quit'."""
        logger.info("UCI engine started")
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            logger.debug(">>> %s", line)
            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                self._handle_uci()
            elif cmd == "isready":
                self._handle_isready()
            elif cmd == "ucinewgame":
                self._handle_ucinewgame()
            elif cmd == "position":
                self._handle_position(tokens[1:])
            elif cmd == "go":
                self._handle_go()
            elif cmd in ("stop", "ponderhit"):
                pass  # No background search; nothing to stop
            elif cmd == "quit":
                logger.info("UCI engine quitting")
                break
            else:
                logger.debug("Unknown command: %s", line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChessGNN UCI engine")
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to GATEAUChessModel checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (default: %(default)s)",
    )
    parser.add_argument(
        "--calib",
        default=None,
        metavar="CALIB_JSON",
        help="Path to temperature calibration JSON sidecar (optional)",
    )
    parser.add_argument(
        "--book",
        default=DEFAULT_BOOK_PATH,
        metavar="BOOK_BIN",
        help="Path to polyglot opening book .bin file (default: %(default)s)",
    )
    parser.add_argument(
        "--no-book",
        action="store_true",
        help="Disable opening book; use GNN from move 1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    UCIEngine(
        model_path=args.model,
        device_str=args.device,
        temperature_path=args.calib,
        book_path=None if args.no_book else args.book,
    ).run()
