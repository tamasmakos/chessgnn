"""Single-experiment runner for the GATEAUChessModel scaling sweep.

Reads a JSON config, trains GATEAUChessModel from scratch, evaluates it on
the standard benchmark suite, writes one row to ``output/scaling_results.csv``,
and saves the best checkpoint to ``output/<run_name>.pt``.

Config schema
-------------
{
    "run_name":        str,          // unique id, e.g. "gateau_h128_l4_d500"
    "hidden_channels": int,          // e.g. 64, 128, 256, 512
    "num_layers":      int,          // e.g. 3, 4, 6, 8
    "temporal_mode":   str,          // "global_gru" | "none" | "node_gru"
    "data_jsonl":      str,          // path to training/eval JSONL
    "epochs":          int,          // default 10
    "lr":              float,        // default 1e-3
    "accumulation_steps": int,       // default 32
    "lambda_v":        float,        // value loss weight (default 1.0)
    "lambda_q":        float,        // policy loss weight (default 1.0)
    "temperature":     float,        // KL temperature (default 1.0)
    "val_split":       float,        // fraction for validation (default 0.1)
    "puzzle_csv":      str | null,   // path to Lichess puzzle CSV (optional)
    "n_puzzles":       int,          // max puzzles to evaluate (default 25)
    "eval_k":          int,          // top-k for engine agreement (default 3)
    "results_csv":     str           // output CSV path (default output/scaling_results.csv)
}

Usage
-----
    python run_experiment.py --config configs/gateau_h128_l4.json
    python run_experiment.py --config configs/sweep_all.json   # runs all entries
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from chessgnn.calibration import TemperatureScaler
from chessgnn.distillation_dataset import DistillationDataset, distillation_collate
from chessgnn.distillation_pipeline import load_jsonl
from chessgnn.eval import Evaluator
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel
from chessgnn.online_distillation import OnlineDistillationDataset

os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("output/run_experiment.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_CSV_DEFAULT = "output/scaling_results.csv"
RESULTS_CSV_FIELDS = [
    "run_name", "hidden_channels", "num_layers", "temporal_mode",
    "params", "data_size", "epochs", "lr",
    "train_time_s",
    "best_val_top1", "best_val_top3",
    "best_val_v_mse", "best_val_q_kl",
    "eval_top1_acc", "eval_top3_acc",
    "puzzle_acc", "puzzle_n",
    "pearson_r", "spearman_rho",
    "ece_before", "ece_after", "temperature_t",
    "checkpoint",
]


# ---------------------------------------------------------------------------
# Training helpers (mirrors distill_train.py but takes explicit config dict)
# ---------------------------------------------------------------------------


def _validate(model, val_loader, device, temperature):
    model.eval()
    total_v, total_q = 0.0, 0.0
    top1, top3, count = 0, 0, 0
    with torch.no_grad():
        for batch_list in val_loader:
            for sample in batch_list:
                pt = sample["policy_target"]
                if pt.numel() == 0:
                    continue
                graph = sample["graph"].to(device)
                vt = sample["value_target"].to(device)
                pt = pt.to(device)
                value, q_scores, _ = model.forward_with_q(graph)
                total_v += F.mse_loss(value.squeeze(), vt.squeeze()).item()
                log_q = F.log_softmax(q_scores / temperature, dim=0)
                total_q += F.kl_div(log_q, pt, reduction="sum").item()
                pred_top = q_scores.topk(min(3, len(q_scores))).indices
                best = pt.argmax()
                if pred_top[0] == best:
                    top1 += 1
                if best in pred_top:
                    top3 += 1
                count += 1
    n = max(count, 1)
    return {
        "val_v_mse": total_v / n,
        "val_q_kl": total_q / n,
        "val_top1_acc": top1 / n,
        "val_top3_acc": top3 / n,
    }


def train_model(cfg: dict, device: torch.device) -> tuple[GATEAUChessModel, dict]:
    """Train GATEAUChessModel from *cfg*.  Returns (model, best_metrics)."""
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 1e-3)
    accum = cfg.get("accumulation_steps", 32)
    lv = cfg.get("lambda_v", 1.0)
    lq = cfg.get("lambda_q", 1.0)
    temp = cfg.get("temperature", 1.0)
    val_split = cfg.get("val_split", 0.1)
    run_name = cfg["run_name"]

    builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)

    if cfg.get("online"):
        # ---- Online mode: Stockfish labeling runs in background thread ----
        # DataLoader must use num_workers=0 (Stockfish subprocess is not fork-safe).
        data_size = cfg["total_positions"]
        logger.info("[%s] Online dataset: %d positions (depth=%d)",
                    run_name, data_size, cfg.get("sf_depth", 8))
        train_loader = DataLoader(
            OnlineDistillationDataset(
                pgn_path=cfg["pgn_path"],
                stockfish_path=cfg.get("stockfish_path", "stockfish/src/stockfish"),
                total_positions=data_size,
                depth=cfg.get("sf_depth", 8),
                multipv_k=cfg.get("multipv_k", 5),
                buffer_size=cfg.get("buffer_size", 128),
                temperature=temp,
                num_sf_workers=cfg.get("num_sf_workers", 4),
            ),
            batch_size=1,
            collate_fn=distillation_collate,
            num_workers=0,
        )
        # Validation comes from a fixed JSONL (val_jsonl key, or data_jsonl fallback)
        val_jsonl = cfg.get("val_jsonl") or cfg.get("data_jsonl")
        val_full = DistillationDataset(val_jsonl, temperature=temp)
        n_val = max(1, int(len(val_full) * val_split))
        val_loader = DataLoader(
            Subset(val_full, list(range(n_val))),
            batch_size=1, collate_fn=distillation_collate,
        )
        logger.info("[%s] Val set: %d positions from %s", run_name, n_val, val_jsonl)
    else:
        # ---- Offline mode: load pre-generated JSONL ----
        dataset = DistillationDataset(cfg["data_jsonl"], temperature=temp)
        data_size = len(dataset)
        logger.info("[%s] Dataset: %d positions", run_name, data_size)
        indices = list(range(data_size))
        random.shuffle(indices)
        split = max(1, int(data_size * (1 - val_split)))
        train_loader = DataLoader(
            Subset(dataset, indices[:split]),
            batch_size=1, collate_fn=distillation_collate, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(dataset, indices[split:]),
            batch_size=1, collate_fn=distillation_collate,
        )

    model = GATEAUChessModel(
        builder.get_metadata(),
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        temporal_mode=cfg.get("temporal_mode", "global_gru"),
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(
        "[%s] GATEAUChessModel: hidden=%d  layers=%d  params=%s",
        run_name, cfg["hidden_channels"], cfg["num_layers"], f"{params:,}",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_path = f"output/{run_name}.pt"
    best_top1 = -1.0
    best_metrics: dict = {}

    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = total_v_loss = total_q_loss = 0.0
        step = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[{run_name}] Epoch {epoch+1}/{epochs}", leave=False)
        for i, batch_list in enumerate(pbar):
            for sample in batch_list:
                pt = sample["policy_target"]
                if pt.numel() == 0:
                    continue
                graph = sample["graph"].to(device)
                vt = sample["value_target"].to(device)
                pt = pt.to(device)
                value, q_scores, _ = model.forward_with_q(graph)
                l_v = F.mse_loss(value.squeeze(), vt.squeeze())
                log_q = F.log_softmax(q_scores / temp, dim=0)
                l_q = F.kl_div(log_q, pt, reduction="sum")
                loss = (lv * l_v + lq * l_q) / accum
                loss.backward()
                if not math.isnan(loss.item()):
                    total_loss += loss.item() * accum
                    total_v_loss += l_v.item()
                    total_q_loss += l_q.item()
                step += 1
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if step % 50 == 0 and step > 0:
                pbar.set_postfix(loss=f"{total_loss/step:.4f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        m = _validate(model, val_loader, device, temp)
        logger.info(
            "[%s] Epoch %d/%d  loss=%.4f  val_top1=%.1f%%  val_top3=%.1f%%",
            run_name, epoch + 1, epochs,
            total_loss / max(step, 1),
            m["val_top1_acc"] * 100, m["val_top3_acc"] * 100,
        )
        if m["val_top1_acc"] > best_top1:
            best_top1 = m["val_top1_acc"]
            best_metrics = {
                "best_val_top1": m["val_top1_acc"],
                "best_val_top3": m["val_top3_acc"],
                "best_val_v_mse": m["val_v_mse"],
                "best_val_q_kl": m["val_q_kl"],
                "params": params, "data_size": data_size,
                "train_time_s": time.time() - t_start,
                "checkpoint": checkpoint_path,
            }
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("[%s]   → new best top-1 %.1f%%, checkpoint saved", run_name, best_top1 * 100)

    if not best_metrics:
        m = _validate(model, val_loader, device, temp)
        best_metrics = {
            "best_val_top1": m["val_top1_acc"],
            "best_val_top3": m["val_top3_acc"],
            "best_val_v_mse": m["val_v_mse"],
            "best_val_q_kl": m["val_q_kl"],
            "params": params, "data_size": data_size,
            "train_time_s": time.time() - t_start,
            "checkpoint": checkpoint_path,
        }
        torch.save(model.state_dict(), checkpoint_path)

    logger.info("[%s] Training done in %.0fs.  Best val top-1: %.1f%%",
                run_name, best_metrics["train_time_s"], best_top1 * 100)
    # Load the best checkpoint back for evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, best_metrics


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_model(
    model: GATEAUChessModel,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Run Evaluator metrics + calibration on the trained model."""
    # In online mode data_jsonl may be absent; fall back to val_jsonl.
    eval_jsonl: str | None = cfg.get("data_jsonl") or cfg.get("val_jsonl")

    ev = Evaluator(model, device=device, use_global_node=True)

    eval_metrics: dict = {}

    if eval_jsonl and os.path.exists(eval_jsonl):
        # Engine agreement
        ea = ev.evaluate_engine_agreement(eval_jsonl, k=cfg.get("eval_k", 3))
        eval_metrics["eval_top1_acc"] = ea["top1_acc"]
        eval_metrics["eval_top3_acc"] = ea.get(f"top{cfg.get('eval_k', 3)}_acc", 0.0)

        # Value correlation
        vc = ev.evaluate_value_correlation(eval_jsonl)
        eval_metrics["pearson_r"] = vc["pearson_r"]
        eval_metrics["spearman_rho"] = vc["spearman_rho"]
    else:
        eval_metrics.update({"eval_top1_acc": 0.0, "eval_top3_acc": 0.0,
                             "pearson_r": 0.0, "spearman_rho": 0.0})

    # Puzzle accuracy (optional)
    puzzle_csv = cfg.get("puzzle_csv")
    if puzzle_csv and os.path.exists(puzzle_csv):
        n_puz = cfg.get("n_puzzles", 25)
        puz = ev.evaluate_puzzles(puzzle_csv, n=n_puz)
        eval_metrics["puzzle_acc"] = puz["accuracy"]
        eval_metrics["puzzle_n"] = puz["count"]
    else:
        eval_metrics["puzzle_acc"] = None
        eval_metrics["puzzle_n"] = 0

    # Calibration: fit temperature on the eval JSONL
    import numpy as np
    preds, targets = [], []
    if eval_jsonl and os.path.exists(eval_jsonl):
        builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
        for rec in load_jsonl(eval_jsonl):
            fen = rec.get("fen")
            ewp = rec.get("eval_wp")
            if fen is None or ewp is None:
                continue
            g = builder.fen_to_graph(fen).to(device)
            with torch.no_grad():
                v = model(g)
                pred = float((v.item() + 1.0) / 2.0)
            preds.append(pred)
            targets.append(float(ewp))

    preds_np = np.array(preds)
    targets_np = np.array(targets)
    scaler = TemperatureScaler()
    ece_before = scaler.ece(preds_np, targets_np)
    eps = 1e-7
    logits = np.log(np.clip(preds_np, eps, 1 - eps) / np.clip(1 - preds_np, eps, 1.0))
    if len(logits) >= 2:
        scaler.fit(logits, targets_np)
    ece_after = scaler.ece(np.array([scaler.calibrate(p) for p in preds_np]), targets_np)
    eval_metrics["ece_before"] = ece_before
    eval_metrics["ece_after"] = ece_after
    eval_metrics["temperature_t"] = scaler.T

    # Save calibration sidecar
    calib_path = cfg.get("checkpoint", f"output/{cfg['run_name']}.pt") + ".calib.json"
    scaler.save(calib_path)

    return eval_metrics


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def _append_csv(row: dict, csv_path: str) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Results appended to %s", csv_path)


# ---------------------------------------------------------------------------
# Single-experiment entry point
# ---------------------------------------------------------------------------


def run_one(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== Experiment: %s | device=%s ===", cfg["run_name"], device)

    model, train_metrics = train_model(cfg, device)
    eval_metrics = evaluate_model(model, cfg, device)

    row = {
        "run_name": cfg["run_name"],
        "hidden_channels": cfg["hidden_channels"],
        "num_layers": cfg["num_layers"],
        "temporal_mode": cfg.get("temporal_mode", "global_gru"),
        "epochs": cfg.get("epochs", 10),
        "lr": cfg.get("lr", 1e-3),
        **train_metrics,
        **eval_metrics,
    }

    results_csv = cfg.get("results_csv", RESULTS_CSV_DEFAULT)
    _append_csv(row, results_csv)

    logger.info(
        "=== %s done | top1=%.1f%% | puzzle=%.1f%% | R=%.3f | ECE %.3f→%.3f ===",
        cfg["run_name"],
        eval_metrics["eval_top1_acc"] * 100,
        (eval_metrics["puzzle_acc"] or 0) * 100,
        eval_metrics["pearson_r"],
        eval_metrics["ece_before"],
        eval_metrics["ece_after"],
    )
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one or more GATEAUChessModel scaling experiments")
    p.add_argument("--config", required=True,
                   help="Path to a JSON config file.  If it contains a top-level "
                        "'experiments' list, all entries are run in sequence.")
    p.add_argument("--results-csv", default=RESULTS_CSV_DEFAULT,
                   help="Path to the results CSV (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        config_data = json.load(f)

    # Support both a single config dict and {"experiments": [...]}
    if "experiments" in config_data:
        experiments = config_data["experiments"]
    else:
        experiments = [config_data]

    for exp_cfg in experiments:
        exp_cfg.setdefault("results_csv", args.results_csv)
        run_one(exp_cfg)

    logger.info("All experiments complete. Results → %s", args.results_csv)
