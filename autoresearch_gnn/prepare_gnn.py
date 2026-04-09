"""
autoresearch_gnn/prepare_gnn.py  —  READ-ONLY HARNESS, DO NOT MODIFY.

Fixed constants, data loading, graph building, and the ground-truth
evaluation metric used by every autoresearch_gnn experiment.

The sole metric is top-1 Stockfish agreement:
    correct / len(val_data)   where correct = argmax(Q-head) == sf_top_move

Val set is fixed at module-import time from EVAL_JSONL.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import chess
import torch

# ---------------------------------------------------------------------------
# Paths — relative to the workspace root (one level up from this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_WORKSPACE = _HERE.parent

TRAIN_JSONL = str(_WORKSPACE / "output" / "game_labels_elo1600_probe1024.jsonl")
EVAL_JSONL  = str(_WORKSPACE / "output" / "distillation_labels_val1k.jsonl")
PUZZLE_CSV  = str(_WORKSPACE / "input"  / "lichess_db_puzzle.csv")

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
TIME_BUDGET   = 600      # wall-clock training seconds (fixed — do not change)
EVAL_N        = 1_000    # number of val positions used for the metric
MAX_TRAIN_N   = 5_000    # cap training positions — agent uses ~3k samples/run, 5k is ample
ELO_NORM_SF   = 1.0      # Stockfish "skill level" ELO normalisation fed to model

# ---------------------------------------------------------------------------
# Lazy-import chessgnn (lives one level up); add workspace root to sys.path
# ---------------------------------------------------------------------------
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from chessgnn.graph_builder import ChessGraphBuilder  # noqa: E402

# Singleton builder — use_move_edges=True required for Q-head inference
_BUILDER = ChessGraphBuilder(use_global_node=True, use_move_edges=True)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _iter_flat_jsonl(path: str, n: int | None = None):
    """Yield flat {fen, eval_wp, top_k_moves} dicts from either JSONL format.

    Two on-disk formats are supported:
      • Flat   — one dict per line: {fen, eval_wp, top_k_moves}
      • Game   — one dict per game: {fens, sf_labels, white_elo, black_elo, result}

    Both are normalised to the flat format before returning.
    """
    count = 0
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            if "fens" in d:
                # Game-level record — expand to per-position records
                for fen, sf in zip(d["fens"], d["sf_labels"]):
                    yield {"fen": fen, "eval_wp": sf["eval_wp"],
                           "top_k_moves": sf.get("top_k_moves", [])}
                    count += 1
                    if n is not None and count >= n:
                        return
            else:
                yield d
                count += 1
                if n is not None and count >= n:
                    return


def _legal_move_uci_list(fen: str) -> list[str]:
    """Return legal moves in the same order ChessGraphBuilder emits move edges."""
    board = chess.Board(fen)
    return [
        move.uci()
        for move in board.legal_moves
        if board.piece_at(move.from_square) is not None
    ]


# ---------------------------------------------------------------------------
# Public data loaders
# ---------------------------------------------------------------------------

_CACHE_DIR = _WORKSPACE / ".cache" / "autoresearch_gnn"


def _cache_path(jsonl_path: str, suffix: str = "") -> Path:
    """Return a deterministic cache path based on the JSONL file's mtime+size and any suffix."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    st = os.stat(jsonl_path)
    key = f"{Path(jsonl_path).name}-{st.st_size}-{st.st_mtime_ns}{suffix}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{Path(jsonl_path).stem}_{digest}.pt"


def _save_batch_cache(records: list[dict], path: Path) -> None:
    """Save records as a Batch + metadata (fast compact format).

    Storing 66k HeteroData as a Python list takes 1.6 GB of pickle that
    deserialises slowly.  Packing them into a single Batch first reduces
    the file to a set of stacked tensors that torch.load reads in seconds.
    """
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([r["graph"] for r in records])
    meta = [
        {
            "value_target":    r["value_target"],
            "top_move_uci":    r["top_move_uci"],
            "legal_moves":     r["legal_moves"],
            "num_legal_moves": r["num_legal_moves"],
        }
        for r in records
    ]
    torch.save({"batch": batch, "meta": meta}, path)


def _load_batch_cache(path: Path) -> list[dict]:
    """Load records saved with _save_batch_cache."""
    data = torch.load(path, weights_only=False)
    batch = data["batch"]
    meta  = data["meta"]
    records = []
    for i, m in enumerate(meta):
        records.append({"graph": batch.get_example(i), **m})
    return records


def load_train_data(jsonl_path: str = TRAIN_JSONL) -> list[dict]:
    """Load training position records from the JSONL, capped at MAX_TRAIN_N.

    Graphs are built once and cached to disk (.cache/autoresearch_gnn/) as a
    compact Batch tensor (not a list of Python objects).  Subsequent calls —
    including across separate agent iterations — load in a few seconds.

    Returns a list of dicts, each with:
        graph           — HeteroData (with move edges)
        value_target    — float in [-1, 1]  (2*eval_wp - 1)
        top_move_uci    — str  (Stockfish best move)
        legal_moves     — list[str]  ordered UCI moves
        num_legal_moves — int
    """
    cache = _cache_path(jsonl_path, suffix=f"-n{MAX_TRAIN_N}")
    if cache.exists():
        print(f"Loading train data from cache: {cache.name}")
        return _load_batch_cache(cache)

    print(f"Building train graphs (first time, caching to {cache.name})…")
    records = []
    for pos in _iter_flat_jsonl(jsonl_path, n=MAX_TRAIN_N * 2):  # over-read to allow filtering
        if len(records) >= MAX_TRAIN_N:
            break
        fen = pos["fen"]
        top_k = pos.get("top_k_moves", [])
        if not top_k:
            continue
        top_move = top_k[0]["uci"]
        legal_moves = _legal_move_uci_list(fen)
        if top_move not in legal_moves:
            continue
        graph = _BUILDER.fen_to_graph(fen)
        records.append({
            "graph":           graph,
            "value_target":    2.0 * pos["eval_wp"] - 1.0,
            "top_move_uci":    top_move,
            "legal_moves":     legal_moves,
            "num_legal_moves": len(legal_moves),
        })
    _save_batch_cache(records, cache)
    print(f"Cached {len(records)} train positions to {cache.name}")
    return records


# ---------------------------------------------------------------------------
# Fixed validation set — built once at import time
# ---------------------------------------------------------------------------

def _build_val_set() -> list[dict]:
    cache = _cache_path(EVAL_JSONL, suffix=f"-n{EVAL_N}")
    if cache.exists():
        print(f"Loading val data from cache: {cache.name}")
        return _load_batch_cache(cache)

    print(f"Building val graphs (first time, caching to {cache.name})…")
    records = []
    for pos in _iter_flat_jsonl(EVAL_JSONL, n=EVAL_N):
        fen = pos["fen"]
        top_k = pos.get("top_k_moves", [])
        if not top_k:
            continue
        top_move = top_k[0]["uci"]
        legal_moves = _legal_move_uci_list(fen)
        if top_move not in legal_moves:
            continue
        graph = _BUILDER.fen_to_graph(fen)
        records.append({
            "graph":        graph,
            "top_move_uci": top_move,
            "legal_moves":  legal_moves,
            "num_legal_moves": len(legal_moves),
        })
    _save_batch_cache(records, cache)
    print(f"Cached {len(records)} val positions to {cache.name}")
    return records


# Build val set once when module is first imported
_VAL_DATA: list[dict] | None = None


def get_val_data() -> list[dict]:
    global _VAL_DATA
    if _VAL_DATA is None:
        _VAL_DATA = _build_val_set()
    return _VAL_DATA


# ---------------------------------------------------------------------------
# Ground-truth evaluation metric — DO NOT MODIFY
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_top1_agreement(model, device: torch.device) -> float:
    """Compute top-1 Stockfish agreement on the fixed val set.

    For each position the model runs ``forward_with_q`` to get Q-scores over
    legal move edges.  The argmax move (in UCI) is compared to the first entry
    in top_k_moves (Stockfish's best move).

    Parameters
    ----------
    model : GATEAUChessModel (or any object with forward_with_q)
        Must be in eval mode and already on *device*.
    device : torch.device

    Returns
    -------
    float  — fraction of positions where argmax Q == sf_top_move  (in [0, 1])
    """
    val_data = get_val_data()
    correct = 0
    model.eval()
    for item in val_data:
        graph = item["graph"].to(device)
        legal_moves = item["legal_moves"]
        top_move = item["top_move_uci"]
        if not legal_moves:
            continue
        _value, q_scores, _mei = model.forward_with_q(graph, elo_norm=ELO_NORM_SF)
        if q_scores.numel() != len(legal_moves):
            continue
        best_idx = q_scores.argmax().item()
        if legal_moves[best_idx] == top_move:
            correct += 1
    return correct / max(len(val_data), 1)


# ---------------------------------------------------------------------------
# Smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading val set from: {EVAL_JSONL}")
    val = get_val_data()
    print(f"Val positions loaded: {len(val)}")
    print(f"Train JSONL        : {TRAIN_JSONL}")
    recs = load_train_data()
    print(f"Train positions loaded: {len(recs)}")
    print("prepare_gnn.py OK")
