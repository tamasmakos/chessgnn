"""Evaluation harness for chess GNN models.

Metrics
-------
* Engine agreement  — top-1 / top-k move match rate vs Stockfish labels.
* Puzzle accuracy   — first-move accuracy on Lichess puzzle CSV.
* Value correlation — Pearson R and Spearman ρ vs Stockfish win probability.
* Reliability diagram — calibration plot (predicted prob vs Stockfish eval).

CLI
---
python -m chessgnn.eval \\
    --model output/gateau_distilled.pt \\
    --positions output/distillation_labels.jsonl \\
    --puzzles input/lichess_puzzles.csv \\
    --out output/eval_results.json
"""

import argparse
import csv
import json
import logging
import os
from typing import Sequence

import chess
import chess.pgn
import numpy as np
import torch

from .calibration import TemperatureScaler
from .distillation_dataset import infer_played_move_uci
from .distillation_pipeline import load_jsonl
from .graph_builder import ChessGraphBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_scalar(step_output) -> float:
    """Extract a value scalar from a forward_step return.

    GATEAUChessModel returns a Tensor [1, 1].
    STHGATLikeModel returns (win_logits [3], mat [1], dom [1]); convert to
    scalar via softmax(white) - softmax(black).
    """
    if isinstance(step_output, torch.Tensor):
        return step_output.item()
    win_logits = step_output[0]
    probs = torch.softmax(win_logits, dim=0)
    return (probs[0] - probs[2]).item()


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.sqrt((x_c**2).sum() * (y_c**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((x_c * y_c).sum() / denom)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata  # scipy is available in this environment

    return _pearson_r(rankdata(x), rankdata(y))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Predicted win probabilities in [0, 1].
    outcomes : np.ndarray, shape (N,)
        Observed outcomes in {0.0, 0.5, 1.0}.

    Returns
    -------
    float
        Brier score ∈ [0, 1]; lower is better.
    """
    p = np.asarray(probs, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    return float(np.mean((p - o) ** 2))


def log_loss_metric(
    probs: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Binary cross-entropy between predicted probabilities and outcomes.

    Draws (outcome=0.5) are handled as soft targets.

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Predicted win probabilities in [0, 1].
    outcomes : np.ndarray, shape (N,)
        Observed outcomes in {0.0, 0.5, 1.0}.
    eps : float
        Clipping threshold to avoid log(0).

    Returns
    -------
    float
        Log-loss ≥ 0; lower is better.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    o = np.asarray(outcomes, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    return float(-np.mean(o * np.log(p) + (1.0 - o) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Runs evaluation metrics for a single chess GNN model.

    Parameters
    ----------
    model : nn.Module
        Must implement either ``forward_with_q(graph)`` (GATEAUChessModel) or
        ``forward_step(graph)`` (STHGATLikeModel).
    device : torch.device, optional
        Inference device.  Defaults to CPU.
    use_global_node : bool
        Whether to build graphs with a global node.  Should match the
        training configuration of the model.
    """

    def __init__(
        self,
        model,
        device: torch.device | None = None,
        use_global_node: bool = True,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self._has_q_head = hasattr(model, "forward_with_q")
        self.builder = ChessGraphBuilder(
            use_global_node=use_global_node,
            use_move_edges=self._has_q_head,
        )

    # ------------------------------------------------------------------
    # Internal: single-position move picker
    # ------------------------------------------------------------------

    def _pick_best_move_uci(self, fen: str) -> str | None:
        """Return the UCI string of the model's top-1 move, or None."""
        topk = self._pick_topk_moves_uci(fen, k=1)
        return topk[0] if topk else None

    def _pick_topk_moves_uci(self, fen: str, k: int = 3) -> list[str]:
        """Return up to k model-ranked legal moves for a position."""
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        graph = self.builder.fen_to_graph(fen).to(self.device)

        with torch.no_grad():
            if self._has_q_head:
                _, q_scores, _ = self.model.forward_with_q(graph)
                q_list = q_scores.cpu().tolist()
                if len(q_list) != len(legal_moves):
                    return []
                topk_idx = q_scores.topk(min(k, len(q_scores))).indices.tolist()
                return [legal_moves[idx].uci() for idx in topk_idx]

            # --- rollout path (STHGATLikeModel) ---
            is_white = board.turn == chess.WHITE
            scored_moves: list[tuple[float, str]] = []
            for move in legal_moves:
                board.push(move)
                g = self.builder.fen_to_graph(board.fen()).to(self.device)
                step_out, _ = self.model.forward_step(g)
                scalar = _extract_scalar(step_out)
                board.pop()
                score = scalar if is_white else -scalar
                scored_moves.append((score, move.uci()))
            scored_moves.sort(key=lambda item: item[0], reverse=True)
            return [uci for _, uci in scored_moves[:k]]

    # ------------------------------------------------------------------
    # Internal: value prediction for a single position
    # ------------------------------------------------------------------

    def _predict_value(self, fen: str) -> float:
        """Return predicted win probability in [0, 1] for the side to move."""
        graph = self.builder.fen_to_graph(fen).to(self.device)
        with torch.no_grad():
            if self._has_q_head:
                v = self.model(graph)  # [1, 1] in [-1, 1]
                return float((v.item() + 1.0) / 2.0)
            # STHGATLikeModel: forward_step returns multi-head tuple
            step_out, _ = self.model.forward_step(graph)
            scalar = _extract_scalar(step_out)  # [-1, 1] white perspective
            # Convert: if it's white's turn, scalar is already white win prob;
            # if it's black's turn we return from side-to-move perspective.
            board = chess.Board(fen)
            if board.turn == chess.WHITE:
                return float((scalar + 1.0) / 2.0)
            else:
                return float((-scalar + 1.0) / 2.0)

    # ------------------------------------------------------------------
    # Public metrics
    # ------------------------------------------------------------------

    def evaluate_engine_agreement(
        self,
        positions_jsonl: str,
        k: int = 1,
    ) -> dict[str, float]:
        """Measure move agreement between the model and Stockfish labels.

        Parameters
        ----------
        positions_jsonl : str
            Path to a JSONL file produced by the distillation pipeline.
            Each record must have ``fen`` and ``top_k_moves`` fields.
        k : int
            Number of Stockfish top moves to consider "correct" for the
            topk accuracy metric.

        Returns
        -------
        dict with keys ``top1_acc``, ``topk_acc``, ``count``.
        """
        top1_correct = 0
        topk_correct = 0
        count = 0

        for rec in load_jsonl(positions_jsonl):
            fen = rec["fen"]
            sf_moves: list[dict] = rec.get("top_k_moves", [])
            if not sf_moves:
                continue

            sf_top1_uci = sf_moves[0].get("uci")
            sf_topk_ucis = {m.get("uci") for m in sf_moves[:k]}

            model_uci = self._pick_best_move_uci(fen)
            if model_uci is None:
                logger.debug("No legal moves for FEN: %s", fen)
                continue

            if model_uci == sf_top1_uci:
                top1_correct += 1
            if model_uci in sf_topk_ucis:
                topk_correct += 1
            count += 1

        n = max(count, 1)
        result = {
            "top1_acc": top1_correct / n,
            f"top{k}_acc": topk_correct / n,
            "count": count,
        }
        logger.info(
            "Engine agreement — top1: %.3f  top%d: %.3f  (n=%d)",
            result["top1_acc"],
            k,
            result[f"top{k}_acc"],
            count,
        )
        return result

    def evaluate_human_move_prediction(
        self,
        games_jsonl: str,
        k: int = 3,
        max_games: int = 0,
        max_positions: int = 0,
    ) -> dict[str, float]:
        """Measure whether the model predicts the move actually played by humans."""
        top1_correct = 0
        topk_correct = 0
        count = 0

        games_seen = 0
        for rec in load_jsonl(games_jsonl):
            if max_games > 0 and games_seen >= max_games:
                break
            fens: list[str] = rec.get("fens", [])
            for idx in range(len(fens) - 1):
                if max_positions > 0 and count >= max_positions:
                    break
                fen = fens[idx]
                played_uci = infer_played_move_uci(fen, fens[idx + 1])
                if played_uci is None:
                    continue
                predicted = self._pick_topk_moves_uci(fen, k=k)
                if not predicted:
                    continue
                if predicted[0] == played_uci:
                    top1_correct += 1
                if played_uci in predicted:
                    topk_correct += 1
                count += 1
            games_seen += 1
            if max_positions > 0 and count >= max_positions:
                break

        n = max(count, 1)
        result = {
            "top1_acc": top1_correct / n,
            f"top{k}_acc": topk_correct / n,
            "count": count,
        }
        logger.info(
            "Human move prediction — top1: %.3f  top%d: %.3f  (n=%d)",
            result["top1_acc"],
            k,
            result[f"top{k}_acc"],
            count,
        )
        return result

    def evaluate_puzzles(
        self,
        puzzle_csv: str,
        n: int = 10_000,
    ) -> dict[str, float]:
        """Evaluate first-move accuracy on the Lichess puzzle CSV.

        The CSV columns are:
            PuzzleId, FEN, Moves, Rating, …

        ``FEN`` is the position just before the opponent's tactic-creating
        move.  ``Moves`` is space-separated UCI moves where ``Moves[0]`` is
        the opponent's move and ``Moves[1]`` is the solution the model must
        find.

        Parameters
        ----------
        puzzle_csv : str
            Path to the Lichess puzzle CSV (no header, or with header row).
        n : int
            Maximum number of puzzles to evaluate.

        Returns
        -------
        dict with keys ``accuracy``, ``count``, ``solved``.
        """
        import itertools

        correct = 0
        count = 0

        with open(puzzle_csv, newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                return {"accuracy": 0.0, "count": 0, "solved": 0}

            # Prepend the first row back unless it is the CSV header
            if first_row[0].lower() == "puzzleid":
                row_iter = reader
            else:
                row_iter = itertools.chain([first_row], reader)

            for row in row_iter:
                if count >= n:
                    break
                if len(row) < 3:
                    continue

                fen_col, moves_col = row[1], row[2]
                moves = moves_col.split()
                if len(moves) < 2:
                    continue

                board = chess.Board(fen_col)
                try:
                    board.push_uci(moves[0])  # opponent's setup move
                except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
                    continue

                puzzle_fen = board.fen()
                solution_uci = moves[1]

                model_uci = self._pick_best_move_uci(puzzle_fen)
                if model_uci == solution_uci:
                    correct += 1
                count += 1

        result = {
            "accuracy": correct / max(count, 1),
            "count": count,
            "solved": correct,
        }
        logger.info(
            "Puzzle accuracy — %.3f  (%d/%d)",
            result["accuracy"],
            correct,
            count,
        )
        return result

    def evaluate_value_correlation(
        self,
        positions_jsonl: str,
    ) -> dict[str, float]:
        """Pearson R and Spearman ρ between predicted V(s) and Stockfish eval.

        Parameters
        ----------
        positions_jsonl : str
            Path to distillation JSONL file with ``fen`` and ``eval_wp``.

        Returns
        -------
        dict with keys ``pearson_r``, ``spearman_rho``, ``count``.
        """
        preds: list[float] = []
        targets: list[float] = []

        for rec in load_jsonl(positions_jsonl):
            fen = rec.get("fen")
            eval_wp = rec.get("eval_wp")
            if fen is None or eval_wp is None:
                continue
            pred = self._predict_value(fen)
            preds.append(pred)
            targets.append(float(eval_wp))

        count = len(preds)
        if count < 2:
            return {"pearson_r": 0.0, "spearman_rho": 0.0, "count": count}

        x = np.array(preds, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)
        r = _pearson_r(x, y)
        rho = _spearman_rho(x, y)

        result = {"pearson_r": r, "spearman_rho": rho, "count": count}
        logger.info(
            "Value correlation — Pearson R=%.4f  Spearman ρ=%.4f  (n=%d)",
            r,
            rho,
            count,
        )
        return result

    def reliability_diagram(
        self,
        positions_jsonl: str,
        out_path: str,
        n_bins: int = 10,
    ) -> list[dict]:
        """Generate a calibration reliability diagram.

        Buckets predicted win probabilities into *n_bins* equal-width bins
        and plots the average Stockfish win probability per bin.  A perfectly
        calibrated model would have each bin's mean prediction equal to its
        mean Stockfish eval (points on the diagonal).

        Parameters
        ----------
        positions_jsonl : str
            Distillation JSONL with ``fen`` and ``eval_wp``.
        out_path : str
            PNG path to save the reliability diagram.
        n_bins : int
            Number of equal-width bins.

        Returns
        -------
        list of dicts with keys ``bin_center``, ``mean_pred``, ``mean_target``,
        ``count`` for each bin.
        """
        import matplotlib.pyplot as plt

        preds: list[float] = []
        targets: list[float] = []

        for rec in load_jsonl(positions_jsonl):
            fen = rec.get("fen")
            eval_wp = rec.get("eval_wp")
            if fen is None or eval_wp is None:
                continue
            preds.append(self._predict_value(fen))
            targets.append(float(eval_wp))

        if not preds:
            logger.warning("No data for reliability diagram.")
            return []

        pred_arr = np.array(preds)
        tgt_arr = np.array(targets)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        bin_stats: list[dict] = []
        mean_preds_plot: list[float] = []
        mean_tgts_plot: list[float] = []
        counts_plot: list[int] = []

        for lo, hi, center in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
            mask = (pred_arr >= lo) & (pred_arr < hi)
            cnt = int(mask.sum())
            if cnt == 0:
                mean_preds_plot.append(float(center))
                mean_tgts_plot.append(float(center))
            else:
                mean_preds_plot.append(float(pred_arr[mask].mean()))
                mean_tgts_plot.append(float(tgt_arr[mask].mean()))
            counts_plot.append(cnt)
            bin_stats.append(
                {
                    "bin_center": float(center),
                    "mean_pred": mean_preds_plot[-1],
                    "mean_target": mean_tgts_plot[-1],
                    "count": cnt,
                }
            )

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_preds_plot, mean_tgts_plot, "o-", label="Model")
        ax.set_xlabel("Mean predicted win probability")
        ax.set_ylabel("Mean Stockfish win probability")
        ax.set_title("Reliability Diagram")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Reliability diagram saved to %s", out_path)
        return bin_stats

    def evaluate_pgn_outcomes(
        self,
        pgn_path: str,
        max_games: int = 0,
    ) -> dict[str, float | int]:
        """Evaluate win-probability quality against real game outcomes from a PGN.

        Parses every position in each game, predicts win probability via the
        value head, and compares against the game result (white's perspective).
        Draws are treated as outcome 0.5.  Games with unknown result (``*``) are
        skipped.

        Parameters
        ----------
        pgn_path : str
            Path to a PGN file.  May contain multiple games.
        max_games : int
            Maximum games to read.  0 means read all games in the file.

        Returns
        -------
        dict with keys:

        * ``brier_score``  — mean squared error between predicted prob and outcome.
        * ``log_loss``     — binary cross-entropy (soft targets for draws).
        * ``ece``          — Expected Calibration Error.
        * ``n_positions``  — total positions evaluated.
        * ``n_games``      — total games used.
        """
        _RESULT_MAP = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

        all_preds: list[float] = []
        all_outcomes: list[float] = []
        n_games = 0

        with open(pgn_path) as pgn_fh:
            while True:
                if max_games > 0 and n_games >= max_games:
                    break
                game = chess.pgn.read_game(pgn_fh)
                if game is None:
                    break
                result_str = game.headers.get("Result", "*")
                outcome = _RESULT_MAP.get(result_str)
                if outcome is None:
                    continue

                board = game.board()
                for move in game.mainline_moves():
                    fen = board.fen()
                    pred = self._predict_value(fen)
                    all_preds.append(pred)
                    all_outcomes.append(outcome)
                    board.push(move)

                n_games += 1

        preds_arr = np.array(all_preds, dtype=np.float64)
        outcomes_arr = np.array(all_outcomes, dtype=np.float64)

        if len(preds_arr) == 0:
            result = {
                "brier_score": 0.0,
                "log_loss": 0.0,
                "ece": 0.0,
                "n_positions": 0,
                "n_games": 0,
            }
        else:
            _scaler = TemperatureScaler()
            result = {
                "brier_score": brier_score(preds_arr, outcomes_arr),
                "log_loss": log_loss_metric(preds_arr, outcomes_arr),
                "ece": _scaler.ece(preds_arr, outcomes_arr),
                "n_positions": len(preds_arr),
                "n_games": n_games,
            }

        logger.info(
            "PGN outcomes — Brier=%.4f  LogLoss=%.4f  ECE=%.4f  "
            "(positions=%d  games=%d)",
            result["brier_score"],
            result["log_loss"],
            result["ece"],
            result["n_positions"],
            result["n_games"],
        )
        return result


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------


def compare_models(
    models_dict: dict,
    positions_jsonl: str | None = None,
    puzzle_csv: str | None = None,
    device: torch.device | None = None,
    k: int = 1,
    n_puzzles: int = 10_000,
    use_global_node: bool = True,
) -> dict:
    """Run all available metrics for each model and return a results dict.

    Parameters
    ----------
    models_dict : dict
        Mapping of ``{name: model}`` for every model to evaluate.
    positions_jsonl : str, optional
        Path to distillation JSONL (for engine agreement and value correlation).
    puzzle_csv : str, optional
        Path to Lichess puzzle CSV (for puzzle accuracy).
    device : torch.device, optional
        Inference device.
    k : int
        Top-k for engine agreement.
    n_puzzles : int
        Maximum puzzles to solve.
    use_global_node : bool
        Builder option; should match training configuration.

    Returns
    -------
    dict
        ``{model_name: {metric: value}}``
    """
    results: dict[str, dict] = {}

    for name, model in models_dict.items():
        logger.info("Evaluating model: %s", name)
        ev = Evaluator(model, device=device, use_global_node=use_global_node)
        model_results: dict = {}

        if positions_jsonl is not None:
            model_results.update(
                ev.evaluate_engine_agreement(positions_jsonl, k=k)
            )
            model_results.update(ev.evaluate_value_correlation(positions_jsonl))

        if puzzle_csv is not None:
            model_results.update(ev.evaluate_puzzles(puzzle_csv, n=n_puzzles))

        results[name] = model_results
        logger.info("Results for %s: %s", name, model_results)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a chess GNN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pt).")
    p.add_argument(
        "--positions",
        default=None,
        help="Distillation JSONL for engine agreement and value correlation.",
    )
    p.add_argument(
        "--puzzles",
        default=None,
        help="Lichess puzzle CSV for puzzle accuracy.",
    )
    p.add_argument(
        "--out",
        default="output/eval_results.json",
        help="Path to write JSON results.",
    )
    p.add_argument(
        "--reliability-out",
        default=None,
        help="PNG path for reliability diagram.  Skipped if not provided.",
    )
    p.add_argument("--k", type=int, default=1, help="Top-k for engine agreement.")
    p.add_argument(
        "--n-puzzles", type=int, default=10_000, help="Max puzzles to evaluate."
    )
    p.add_argument(
        "--no-global-node",
        action="store_true",
        help="Build graphs without global node (for STHGATLikeModel).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string, e.g. 'cpu' or 'cuda'.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    os.makedirs("output", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("output/eval.log"),
            logging.StreamHandler(),
        ],
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    logger.info("Loading model from %s", args.model)
    checkpoint = torch.load(args.model, map_location=device)
    # Support checkpoints saved as state_dict or as the full model
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        from .graph_builder import ChessGraphBuilder as _CGB
        from .model import GATEAUChessModel

        builder = _CGB(
            use_global_node=not args.no_global_node,
            use_move_edges=True,
        )
        model = GATEAUChessModel(
            builder.get_metadata(),
            hidden_channels=checkpoint.get("hidden_channels", 128),
            num_layers=checkpoint.get("num_layers", 4),
            temporal_mode=checkpoint.get("temporal_mode", "none"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = checkpoint

    ev = Evaluator(model, device=device, use_global_node=not args.no_global_node)
    all_results: dict = {}

    if args.positions:
        all_results.update(ev.evaluate_engine_agreement(args.positions, k=args.k))
        all_results.update(ev.evaluate_value_correlation(args.positions))
        if args.reliability_out:
            ev.reliability_diagram(args.positions, args.reliability_out)

    if args.puzzles:
        all_results.update(ev.evaluate_puzzles(args.puzzles, n=args.n_puzzles))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results written to %s", args.out)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
