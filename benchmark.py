"""Regression benchmark for chess GNN win-probability quality.

Evaluates a checkpoint against real game outcomes from a PGN file and appends
results to an append-only JSONL and CSV log.  Designed to be run after each
training checkpoint to track whether win-probability quality is improving or
degrading over time.

Metrics reported
----------------
* brier_score  — mean squared error between predicted win probability and outcome.
* log_loss     — binary cross-entropy (draws treated as 0.5 soft target).
* ece          — Expected Calibration Error.

Usage
-----
    python benchmark.py \\
        --checkpoint output/gateau_distilled.pt \\
        --pgn input/lichess_db_standard_rated_2013-01.pgn \\
        --run-id gateau_distilled_v1 \\
        --max-games 200

    python benchmark.py \\
        --checkpoint output/gateau_medium.pt \\
        --pgn input/lichess_db_standard_rated_2013-01.pgn \\
        --run-id gateau_medium_epoch20 \\
        --calib output/gateau_medium.pt.calib.json \\
        --output-jsonl output/benchmark_log.jsonl \\
        --output-csv output/benchmark_log.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Sequence

import torch

from chessgnn.calibration import TemperatureScaler
from chessgnn.eval import Evaluator
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JSONL = "output/benchmark_log.jsonl"
DEFAULT_CSV = "output/benchmark_log.csv"

_CSV_FIELDS = [
    "run_id",
    "timestamp",
    "checkpoint",
    "pgn",
    "n_games",
    "n_positions",
    "brier_score",
    "log_loss",
    "ece",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("output/benchmark.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loader (mirrors uci_engine._load_model; not imported to avoid coupling
# to a top-level script)
# ---------------------------------------------------------------------------


def _load_model(model_path: str, device: torch.device) -> GATEAUChessModel:
    """Load a GATEAUChessModel from a checkpoint, inferring architecture."""
    ckpt = torch.load(model_path, map_location=device)

    if "global_gru.weight_ih_l0" in ckpt:
        hidden_channels = ckpt["global_gru.weight_ih_l0"].shape[0] // 3
    else:
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("Appended result to %s", path)


def _append_csv(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k) for k in _CSV_FIELDS})
    logger.info("Appended result to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark a chess GNN checkpoint against real game outcomes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint file.",
    )
    p.add_argument(
        "--pgn",
        required=True,
        help="Path to PGN file containing real games.",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Unique identifier for this benchmark run (e.g. 'gateau_medium_epoch20').",
    )
    p.add_argument(
        "--output-jsonl",
        default=DEFAULT_JSONL,
        help="Path to append-only JSONL log.",
    )
    p.add_argument(
        "--output-csv",
        default=DEFAULT_CSV,
        help="Path to append-only CSV log.",
    )
    p.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Maximum games to read from PGN.  0 = all.",
    )
    p.add_argument(
        "--calib",
        default=None,
        help="Path to .calib.json sidecar for temperature scaling (optional).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string, e.g. 'cpu' or 'cuda'.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    os.makedirs("output", exist_ok=True)
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    model = _load_model(args.checkpoint, device)

    evaluator = Evaluator(model, device=device, use_global_node=True)

    if args.calib is not None:
        scaler = TemperatureScaler()
        scaler.load(args.calib)
        logger.info("Calibration loaded: T=%.4f", scaler.T)
    else:
        scaler = None

    logger.info(
        "Benchmarking %s against %s (max_games=%d)",
        args.checkpoint,
        args.pgn,
        args.max_games,
    )

    metrics = evaluator.evaluate_pgn_outcomes(args.pgn, max_games=args.max_games)

    if scaler is not None:
        logger.info(
            "Note: --calib was provided but calibration is applied to individual "
            "predictions by the Evaluator if integrated.  Temperature T=%.4f logged.",
            scaler.T,
        )

    record = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "pgn": args.pgn,
        "n_games": metrics["n_games"],
        "n_positions": metrics["n_positions"],
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "ece": metrics["ece"],
    }

    logger.info(
        "Run '%s' — Brier=%.4f  LogLoss=%.4f  ECE=%.4f  "
        "(positions=%d  games=%d)",
        record["run_id"],
        record["brier_score"],
        record["log_loss"],
        record["ece"],
        record["n_positions"],
        record["n_games"],
    )

    _append_jsonl(args.output_jsonl, record)
    _append_csv(args.output_csv, record)


if __name__ == "__main__":
    main()
