import argparse
import json
import logging
import math
import os
from typing import Iterator

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_MATE_CP = 15_000


def cp_to_winprob(cp: float) -> float:
    """Convert centipawn score to win probability in [0, 1].

    Uses the Lichess approximation: wp = 1 / (1 + exp(-cp / 400)).
    Mate scores (|cp| >= _MATE_CP) are clamped to 1.0 / 0.0.
    """
    if cp >= _MATE_CP:
        return 1.0
    if cp <= -_MATE_CP:
        return 0.0
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


# ---------------------------------------------------------------------------
# PGN sampler
# ---------------------------------------------------------------------------


def sample_positions_from_pgn(
    pgn_path: str,
    max_positions: int,
    min_move: int = 10,
    max_move: int = 100,
) -> Iterator[str]:
    """Yield up to *max_positions* FEN strings from *pgn_path*.

    Only positions where the full-move number is in [min_move, max_move] are
    emitted.  The iterator stops once *max_positions* FENs have been yielded
    or the PGN file is exhausted.
    """
    count = 0
    with open(pgn_path) as f:
        while count < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number < min_move:
                    continue
                if board.fullmove_number > max_move:
                    break
                yield board.fen()
                count += 1
                if count >= max_positions:
                    return


# ---------------------------------------------------------------------------
# Stockfish evaluator
# ---------------------------------------------------------------------------


def evaluate_positions(
    fens: Iterator[str],
    stockfish_path: str,
    depth: int = 12,
    multipv_k: int = 5,
) -> Iterator[dict]:
    """Evaluate each FEN with Stockfish and yield labelled dicts.

    Each yielded dict has the shape::

        {
            "fen": str,
            "eval_wp": float,          # win-prob for side to move
            "top_k_moves": [
                {"uci": str, "cp": int, "wp": float},
                ...
            ],
        }

    Win probabilities are always from the **side-to-move's** perspective.
    """
    engine: chess.engine.SimpleEngine | None = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        for fen in fens:
            board = chess.Board(fen)
            try:
                infos = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=multipv_k,
                )
            except chess.engine.EngineTerminatedError:
                logger.warning("Engine terminated; restarting for remaining positions")
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                continue

            top_k: list[dict] = []
            for info in infos:
                pov_score = info["score"].relative
                if pov_score.is_mate():
                    cp = _MATE_CP if pov_score.mate() > 0 else -_MATE_CP
                else:
                    cp = pov_score.score()
                wp = cp_to_winprob(cp)
                uci_str = info.get("pv", [None])[0]
                if uci_str is not None:
                    uci_str = uci_str.uci()
                top_k.append({"uci": uci_str, "cp": cp, "wp": wp})

            # Overall eval is the best line's score
            best_cp = top_k[0]["cp"] if top_k else 0
            eval_wp = cp_to_winprob(best_cp)

            yield {
                "fen": fen,
                "eval_wp": eval_wp,
                "top_k_moves": top_k,
            }
    finally:
        if engine is not None:
            engine.quit()


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def save_jsonl(labels: Iterator[dict], out_path: str) -> int:
    """Write *labels* as one JSON object per line. Returns the count written."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    count = 0
    with open(out_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")
            count += 1
    return count


def load_jsonl(path: str) -> Iterator[dict]:
    """Stream-read a JSONL file, yielding one dict per line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------


class DistillationDatasetBuilder:
    """Orchestrates PGN sampling → Stockfish evaluation → JSONL output."""

    def __init__(
        self,
        pgn_path: str,
        stockfish_path: str,
        out_path: str,
        max_positions: int = 10_000,
        depth: int = 12,
        multipv_k: int = 5,
        min_move: int = 10,
        max_move: int = 100,
    ):
        self.pgn_path = pgn_path
        self.stockfish_path = stockfish_path
        self.out_path = out_path
        self.max_positions = max_positions
        self.depth = depth
        self.multipv_k = multipv_k
        self.min_move = min_move
        self.max_move = max_move

    def build(self) -> int:
        """Run the full pipeline. Returns the number of positions written."""
        logger.info(
            "Sampling up to %d positions (moves %d–%d) from %s",
            self.max_positions, self.min_move, self.max_move, self.pgn_path,
        )

        fens = sample_positions_from_pgn(
            self.pgn_path,
            self.max_positions,
            self.min_move,
            self.max_move,
        )

        logger.info(
            "Evaluating with Stockfish (depth=%d, multipv=%d) at %s",
            self.depth, self.multipv_k, self.stockfish_path,
        )

        labels = evaluate_positions(
            tqdm(fens, total=self.max_positions, desc="Evaluating"),
            self.stockfish_path,
            self.depth,
            self.multipv_k,
        )

        count = save_jsonl(labels, self.out_path)
        logger.info("Wrote %d labelled positions to %s", count, self.out_path)
        return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Stockfish-labelled distillation dataset from PGN games."
    )
    parser.add_argument("--pgn", required=True, help="Path to PGN file")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish binary")
    parser.add_argument("--positions", type=int, default=10_000, help="Max positions to label")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth")
    parser.add_argument("--multipv", type=int, default=5, help="Number of top moves per position")
    parser.add_argument("--min-move", type=int, default=10, help="Earliest full-move number to sample")
    parser.add_argument("--max-move", type=int, default=100, help="Latest full-move number to sample")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("output/distillation.log"),
            logging.StreamHandler(),
        ],
    )

    builder = DistillationDatasetBuilder(
        pgn_path=args.pgn,
        stockfish_path=args.stockfish,
        out_path=args.out,
        max_positions=args.positions,
        depth=args.depth,
        multipv_k=args.multipv,
        min_move=args.min_move,
        max_move=args.max_move,
    )
    builder.build()


if __name__ == "__main__":
    main()
