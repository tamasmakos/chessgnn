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


def sample_games_from_pgn(
    pgn_path: str,
    max_games: int,
    min_move: int = 5,
    max_move: int = 120,
) -> Iterator[dict]:
    """Yield game dicts from *pgn_path*.

    Each yielded dict has the shape::

        {
            "fens":       list[str],   # ordered FENs from min_move to end
            "white_elo":  int,         # 1500 if header absent
            "black_elo":  int,         # 1500 if header absent
            "result":     str,         # "1-0" | "0-1" | "1/2-1/2" | "*"
        }

    Games with fewer than 2 qualifying positions or unknown result ("*")
    are skipped.
    """
    _RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
    count = 0
    with open(pgn_path) as f:
        while count < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in _RESULT_MAP:
                continue
            try:
                white_elo = int(game.headers.get("WhiteElo", 1500))
            except ValueError:
                white_elo = 1500
            try:
                black_elo = int(game.headers.get("BlackElo", 1500))
            except ValueError:
                black_elo = 1500

            fens: list[str] = []
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number < min_move:
                    continue
                if board.fullmove_number > max_move:
                    break
                fens.append(board.fen())

            if len(fens) < 2:
                continue

            yield {
                "fens": fens,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "result": result,
            }
            count += 1


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


def evaluate_positions_engine(
    fens: Iterator[str],
    engine: "chess.engine.SimpleEngine",
    depth: int = 12,
    multipv_k: int = 5,
) -> Iterator[dict]:
    """Like :func:`evaluate_positions` but accepts an already-open engine.

    The caller is responsible for opening and closing the engine.  This
    allows multiple games to be evaluated on the same engine instance without
    the overhead of subprocess startup per game.
    """
    for fen in fens:
        board = chess.Board(fen)
        try:
            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=multipv_k,
            )
        except chess.engine.EngineTerminatedError:
            logger.warning("Engine terminated during evaluate_positions_engine; skipping FEN")
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

        best_cp = top_k[0]["cp"] if top_k else 0
        eval_wp = cp_to_winprob(best_cp)

        yield {"fen": fen, "eval_wp": eval_wp, "top_k_moves": top_k}


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
# Offline game-sequence pre-generation
# ---------------------------------------------------------------------------


def _evaluate_game_worker(args: tuple) -> dict | None:
    """Top-level function for ProcessPoolExecutor.

    Args tuple: (game_dict, stockfish_path, depth, multipv_k)

    Returns a JSONL-ready dict or None if the game should be skipped.
    Each dict::

        {
            "fens":       [str, ...],
            "sf_labels":  [{"eval_wp": float, "top_k_moves": [...]}, ...],
            "white_elo":  int,
            "black_elo":  int,
            "result":     str,
        }
    """
    game_dict, stockfish_path, depth, multipv_k = args
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        sf_labels: list[dict] = []
        for rec in evaluate_positions_engine(
            iter(game_dict["fens"]),
            engine,
            depth=depth,
            multipv_k=multipv_k,
        ):
            sf_labels.append({"eval_wp": rec["eval_wp"], "top_k_moves": rec["top_k_moves"]})
        engine.quit()
    except Exception as exc:
        logger.warning("Worker failed for a game: %s", exc)
        return None

    if len(sf_labels) < 2:
        return None

    return {
        "fens": game_dict["fens"][:len(sf_labels)],
        "sf_labels": sf_labels,
        "white_elo": game_dict["white_elo"],
        "black_elo": game_dict["black_elo"],
        "result": game_dict["result"],
    }


def save_game_labels_jsonl(
    pgn_path: str,
    stockfish_path: str,
    out_path: str,
    max_games: int,
    depth: int = 8,
    multipv_k: int = 5,
    num_workers: int = 4,
    min_move: int = 5,
    max_move: int = 120,
) -> int:
    """Pre-generate Stockfish labels for full game sequences → JSONL.

    Each worker runs its own Stockfish process (truly parallel — no GIL on
    subprocess I/O).  Labels are written as one JSON object per line.

    Parameters
    ----------
    pgn_path, stockfish_path, out_path : str
    max_games : int
        Number of qualifying games to label.
    depth : int
        Stockfish search depth per position.
    multipv_k : int
        Number of top moves returned per position.
    num_workers : int
        Number of parallel Stockfish workers (ProcessPoolExecutor).
    min_move, max_move : int
        Move-number window; positions outside this range are skipped.

    Returns
    -------
    int
        Number of games written.
    """
    import concurrent.futures
    from tqdm import tqdm

    games = list(sample_games_from_pgn(pgn_path, max_games, min_move=min_move, max_move=max_move))
    logger.info("Generating labels for %d games with %d workers (depth=%d)", len(games), num_workers, depth)

    work_args = [(g, stockfish_path, depth, multipv_k) for g in games]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w") as f, concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        for result in tqdm(
            pool.map(_evaluate_game_worker, work_args, chunksize=1),
            total=len(work_args),
            desc="Labelling games",
        ):
            if result is not None:
                f.write(json.dumps(result) + "\n")
                written += 1

    logger.info("Saved %d labelled games to %s", written, out_path)
    return written




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
