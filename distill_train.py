
import logging
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from chessgnn.distillation_dataset import DistillationDataset, distillation_collate
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JSONL_PATH = "output/distillation_labels.jsonl"
HIDDEN_DIM = 128
NUM_LAYERS = 4
TEMPORAL_MODE = "global_gru"
LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 1
ACCUMULATION_STEPS = 32
LAMBDA_V = 1.0
LAMBDA_Q = 1.0
TEMPERATURE = 1.0
VALIDATION_SPLIT = 0.1
CHECKPOINT_PATH = "output/gateau_distilled.pt"
PRETRAIN_CHECKPOINT: str | None = None  # Set to load weights before distillation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("output/distill_train.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    model: GATEAUChessModel,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_v_mse = 0.0
    total_q_kl = 0.0
    top1_correct = 0
    top3_correct = 0
    count = 0

    with torch.no_grad():
        for batch_list in val_loader:
            for sample in batch_list:
                graph = sample["graph"].to(device)
                value_target = sample["value_target"].to(device)
                policy_target = sample["policy_target"].to(device)

                value, q_scores, _ = model.forward_with_q(graph)

                v_mse = F.mse_loss(value.squeeze(), value_target.squeeze())
                log_q = F.log_softmax(q_scores / TEMPERATURE, dim=0)
                q_kl = F.kl_div(log_q, policy_target, reduction="sum")

                total_v_mse += v_mse.item()
                total_q_kl += q_kl.item()

                # Top-1 / Top-3 accuracy
                pred_top = q_scores.topk(min(3, len(q_scores))).indices
                target_best = policy_target.argmax()

                if pred_top[0] == target_best:
                    top1_correct += 1
                if target_best in pred_top:
                    top3_correct += 1

                count += 1

    n = max(count, 1)
    return {
        "val_v_mse": total_v_mse / n,
        "val_q_kl": total_q_kl / n,
        "val_top1_acc": top1_correct / n,
        "val_top3_acc": top3_correct / n,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Dataset
    logger.info("Loading distillation labels from %s", JSONL_PATH)
    dataset = DistillationDataset(JSONL_PATH, temperature=TEMPERATURE)
    logger.info("Loaded %d positions", len(dataset))

    # Train / val split
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - VALIDATION_SPLIT))
    train_set = Subset(dataset, indices[:split])
    val_set = Subset(dataset, indices[split:])
    logger.info("Train: %d  |  Val: %d", len(train_set), len(val_set))

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, collate_fn=distillation_collate, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, collate_fn=distillation_collate
    )

    # Model
    builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
    metadata = builder.get_metadata()
    model = GATEAUChessModel(
        metadata, hidden_channels=HIDDEN_DIM, num_layers=NUM_LAYERS, temporal_mode=TEMPORAL_MODE
    ).to(device)

    if PRETRAIN_CHECKPOINT and os.path.exists(PRETRAIN_CHECKPOINT):
        logger.info("Loading pretrained weights from %s", PRETRAIN_CHECKPOINT)
        model.load_state_dict(torch.load(PRETRAIN_CHECKPOINT, map_location=device))

    logger.info(
        "GATEAUChessModel: hidden=%d, layers=%d, temporal=%s",
        HIDDEN_DIM, NUM_LAYERS, TEMPORAL_MODE,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_top1 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_v = 0.0
        total_q = 0.0
        step = 0

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, batch_list in enumerate(pbar):
            for sample in batch_list:
                graph = sample["graph"].to(device)
                value_target = sample["value_target"].to(device)
                policy_target = sample["policy_target"].to(device)

                value, q_scores, _ = model.forward_with_q(graph)

                l_value = F.mse_loss(value.squeeze(), value_target.squeeze())
                log_q = F.log_softmax(q_scores / TEMPERATURE, dim=0)
                l_policy = F.kl_div(log_q, policy_target, reduction="sum")

                loss = LAMBDA_V * l_value + LAMBDA_Q * l_policy
                loss_scaled = loss / ACCUMULATION_STEPS
                loss_scaled.backward()

                if not math.isnan(loss.item()):
                    total_loss += loss.item()
                    total_v += l_value.item()
                    total_q += l_policy.item()
                step += 1

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0 and step > 0:
                pbar.set_postfix(
                    loss=f"{total_loss / step:.4f}",
                    v_mse=f"{total_v / step:.4f}",
                    q_kl=f"{total_q / step:.4f}",
                )

        # Flush remaining gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / max(step, 1)
        logger.info(
            "Epoch %d/%d | Loss: %.4f | V_MSE: %.4f | Q_KL: %.4f",
            epoch + 1, EPOCHS, avg_loss, total_v / max(step, 1), total_q / max(step, 1),
        )

        # Validation
        metrics = validate(model, val_loader, device)
        logger.info(
            "  val_v_mse=%.4f  val_q_kl=%.4f  top1=%.2f%%  top3=%.2f%%",
            metrics["val_v_mse"],
            metrics["val_q_kl"],
            metrics["val_top1_acc"] * 100,
            metrics["val_top3_acc"] * 100,
        )

        if metrics["val_top1_acc"] > best_top1:
            best_top1 = metrics["val_top1_acc"]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            logger.info("  Saved best checkpoint (top1=%.2f%%) → %s", best_top1 * 100, CHECKPOINT_PATH)

    logger.info("Training complete. Best top-1 accuracy: %.2f%%", best_top1 * 100)


if __name__ == "__main__":
    train()
