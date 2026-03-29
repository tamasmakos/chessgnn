"""Calibration fit script.

Fits a post-hoc temperature scalar T to a held-out calibration set of
Stockfish-evaluated positions, saves the scalar as a JSON sidecar alongside
the model checkpoint, and generates before/after reliability diagram PNGs.

Usage
-----
    python calibrate.py \\
        --model output/gateau_distilled.pt \\
        --positions output/distillation_labels.jsonl

Outputs (next to the model checkpoint by default):
    <model>.calib.json          – {"temperature": T}
    <model>.calib_before.png    – reliability diagram before calibration
    <model>.calib_after.png     – reliability diagram after calibration
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

from chessgnn.calibration import TemperatureScaler, reliability_diagram
from chessgnn.distillation_pipeline import load_jsonl
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading (mirrors run_eval.py)
# ---------------------------------------------------------------------------


def _load_gateau(model_path: str, device: torch.device) -> GATEAUChessModel:
    ckpt = torch.load(model_path, map_location=device)

    if "global_gru.weight_ih_l0" in ckpt:
        hidden_channels = ckpt["global_gru.weight_ih_l0"].shape[0] // 3
    else:
        hidden_channels = next(
            v.shape[0] for k, v in ckpt.items() if k.startswith("convs.0.k_lin.")
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
        "Loaded GATEAUChessModel (hidden=%d, layers=%d) from %s",
        hidden_channels,
        num_layers,
        model_path,
    )
    return model


# ---------------------------------------------------------------------------
# Collect predictions and targets
# ---------------------------------------------------------------------------


def _collect_calibration_data(
    model: GATEAUChessModel,
    positions_jsonl: str,
    device: torch.device,
    max_positions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over the calibration set and gather predicted/target win probs.

    Returns
    -------
    preds : np.ndarray, shape (N,)
        Predicted win probabilities in [0, 1] from the value head.
    targets : np.ndarray, shape (N,)
        Stockfish win probabilities (eval_wp) in [0, 1].
    """
    builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
    preds: list[float] = []
    targets: list[float] = []

    for i, rec in enumerate(load_jsonl(positions_jsonl)):
        if max_positions is not None and i >= max_positions:
            break

        fen = rec.get("fen")
        eval_wp = rec.get("eval_wp")
        if fen is None or eval_wp is None:
            continue

        try:
            graph = builder.fen_to_graph(fen).to(device)
        except Exception as exc:
            logger.debug("Skipping FEN %s: %s", fen, exc)
            continue

        with torch.no_grad():
            v = model(graph)  # [1, 1] tanh in [-1, 1]
            pred_wp = float((v.item() + 1.0) / 2.0)

        preds.append(pred_wp)
        targets.append(float(eval_wp))

        if (i + 1) % 500 == 0:
            logger.info("Collected %d positions …", i + 1)

    logger.info("Calibration dataset: %d positions", len(preds))
    return np.array(preds, dtype=np.float64), np.array(targets, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = _load_gateau(args.model, device)

    preds, targets = _collect_calibration_data(
        model,
        args.positions,
        device,
        max_positions=args.max_positions,
    )

    if len(preds) < 2:
        logger.error("Not enough calibration samples (got %d). Aborting.", len(preds))
        sys.exit(1)

    scaler = TemperatureScaler()
    _EPSILON = 1e-7
    logits = np.log(
        np.clip(preds, _EPSILON, 1 - _EPSILON) / np.clip(1 - preds, _EPSILON, 1.0)
    )

    ece_before = scaler.ece(preds, targets)
    logger.info("ECE before calibration: %.5f", ece_before)

    # Reliability diagram before calibration
    before_fig = reliability_diagram(preds, targets, n_bins=args.n_bins)
    before_fig.suptitle(f"Before calibration  (ECE={ece_before:.4f})")
    before_path = args.model + ".calib_before.png"
    before_fig.savefig(before_path, dpi=120)
    logger.info("Saved before-diagram: %s", before_path)

    scaler.fit(logits, targets)
    logger.info("Fitted temperature: T=%.4f", scaler.T)

    calibrated_preds = np.array([scaler.calibrate(p) for p in preds])
    ece_after = scaler.ece(calibrated_preds, targets)
    logger.info("ECE after calibration: %.5f  (delta=%.5f)", ece_after, ece_before - ece_after)

    # Reliability diagram after calibration
    after_fig = reliability_diagram(calibrated_preds, targets, n_bins=args.n_bins)
    after_fig.suptitle(f"After calibration  T={scaler.T:.3f}  (ECE={ece_after:.4f})")
    after_path = args.model + ".calib_after.png"
    after_fig.savefig(after_path, dpi=120)
    logger.info("Saved after-diagram: %s", after_path)

    calib_json_path = args.model + ".calib.json"
    scaler.save(calib_json_path)
    logger.info("Calibration sidecar saved: %s", calib_json_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit temperature scaling calibration for ChessGNN")
    parser.add_argument(
        "--model",
        default="output/gateau_distilled.pt",
        help="Path to GATEAUChessModel checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--positions",
        default="output/distillation_labels.jsonl",
        help="Path to calibration JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (default: %(default)s)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of calibration positions (default: all)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for reliability diagrams (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
