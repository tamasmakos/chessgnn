import torch
import torch.nn.functional as F
import chess
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from .distillation_pipeline import load_jsonl
from .graph_builder import ChessGraphBuilder


def legal_move_index_map(fen: str) -> dict[str, int]:
    """Return the UCI-to-index mapping matching graph_builder move-edge order."""
    board = chess.Board(fen)
    uci_to_idx: dict[str, int] = {}
    idx = 0
    for move in board.legal_moves:
        if board.piece_at(move.from_square) is None:
            continue
        uci_to_idx[move.uci()] = idx
        idx += 1
    return uci_to_idx


def hard_policy_target(
    move_uci: str | None,
    fen: str,
    num_legal_moves: int,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Build a one-hot or lightly smoothed target over legal moves."""
    if num_legal_moves <= 0:
        return torch.empty(0, dtype=torch.float32)

    move_idx = legal_move_index_map(fen).get(move_uci or "")
    if move_idx is None:
        return torch.empty(0, dtype=torch.float32)

    probs = torch.zeros(num_legal_moves, dtype=torch.float32)
    if smoothing > 0.0 and num_legal_moves > 1:
        off_value = smoothing / float(num_legal_moves - 1)
        probs.fill_(off_value)
        probs[move_idx] = 1.0 - smoothing
    else:
        probs[move_idx] = 1.0
    return probs


def infer_played_move_uci(current_fen: str, next_fen: str) -> str | None:
    """Recover the played move from two consecutive FENs, if legal and unique."""
    board = chess.Board(current_fen)
    for move in board.legal_moves:
        board.push(move)
        matches = board.fen() == next_fen
        board.pop()
        if matches:
            return move.uci()
    return None


def soft_policy_target(
    top_k_moves: list[dict],
    fen: str,
    num_legal_moves: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build a soft probability distribution over legal moves.

    Parameters
    ----------
    top_k_moves : list of dict
        Each entry has ``"uci"`` (str) and ``"cp"`` (int) from Stockfish.
    fen : str
        Board FEN used to reconstruct the legal-move ordering (must match
        the order used by ``ChessGraphBuilder``).
    num_legal_moves : int
        Total number of legal moves (= number of move edges in the graph).
    temperature : float
        Softmax temperature.  Lower → sharper distribution.

    Returns
    -------
    Tensor [M]
        Probability distribution over the M legal moves.
    """
    # Build UCI → index mapping matching graph_builder iteration order
    uci_to_idx = legal_move_index_map(fen)

    # Collect logits for matched moves; use large negative for unmatched
    logits = torch.full((num_legal_moves,), -1e9)
    for entry in top_k_moves:
        uci = entry.get("uci")
        cp = entry.get("cp", 0)
        if uci in uci_to_idx:
            logits[uci_to_idx[uci]] = float(cp)

    # If no moves matched (shouldn't happen), return uniform
    if (logits > -1e8).sum() == 0:
        return torch.ones(num_legal_moves) / max(num_legal_moves, 1)

    probs = F.softmax(logits / max(temperature, 1e-8), dim=0)
    return probs


class DistillationDataset(Dataset):
    """Map-style dataset that reads JSONL labels and builds graphs on-the-fly.

    Each item is a dict with keys:
        ``graph``          – HeteroData (with move edges)
        ``value_target``   – Tensor [1] in [-1, 1]
        ``policy_target``  – Tensor [M] soft probability over legal moves
        ``num_legal_moves``– int
    """

    def __init__(
        self,
        jsonl_path: str,
        graph_builder: ChessGraphBuilder | None = None,
        temperature: float = 1.0,
    ):
        self.labels: list[dict] = list(load_jsonl(jsonl_path))
        self.builder = graph_builder or ChessGraphBuilder(
            use_global_node=True, use_move_edges=True
        )
        self.temperature = temperature

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        fen = label["fen"]
        eval_wp = label["eval_wp"]

        graph: HeteroData = self.builder.fen_to_graph(fen)
        num_legal = graph['piece', 'move', 'square'].edge_index.shape[1]

        value_target = torch.tensor([2.0 * eval_wp - 1.0])
        policy_target = soft_policy_target(
            label["top_k_moves"], fen, num_legal, self.temperature
        )

        return {
            "graph": graph,
            "value_target": value_target,
            "policy_target": policy_target,
            "num_legal_moves": num_legal,
        }


def distillation_collate(batch: list[dict]) -> list[dict]:
    """Passthrough collate — graphs have different move counts."""
    return batch
