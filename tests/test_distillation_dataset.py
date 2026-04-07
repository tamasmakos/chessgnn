import json
import os
import tempfile

import chess
import pytest
import torch
from torch.utils.data import DataLoader

from chessgnn.distillation_dataset import (
    DistillationDataset,
    distillation_collate,
    hard_policy_target,
    infer_played_move_uci,
    legal_move_index_map,
    soft_policy_target,
)
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel
from chessgnn.online_distillation import GameSequenceOfflineDataset

# A few positions with known Stockfish-style labels for testing.
SAMPLE_LABELS = [
    {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "eval_wp": 0.52,
        "top_k_moves": [
            {"uci": "e7e5", "cp": 30, "wp": 0.52},
            {"uci": "c7c5", "cp": 20, "wp": 0.51},
            {"uci": "d7d5", "cp": 10, "wp": 0.50},
        ],
    },
    {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "eval_wp": 0.55,
        "top_k_moves": [
            {"uci": "g1f3", "cp": 40, "wp": 0.55},
            {"uci": "f1c4", "cp": 30, "wp": 0.53},
        ],
    },
    {
        "fen": "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "eval_wp": 0.54,
        "top_k_moves": [
            {"uci": "d2d4", "cp": 35, "wp": 0.54},
        ],
    },
]


def _write_sample_jsonl(path: str) -> None:
    with open(path, "w") as f:
        for rec in SAMPLE_LABELS:
            f.write(json.dumps(rec) + "\n")


def _write_sample_game_jsonl(path: str) -> None:
    records = [
        {
            "fens": [
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            ],
            "sf_labels": [
                {
                    "eval_wp": 0.52,
                    "top_k_moves": [
                        {"uci": "e7e5", "cp": 30, "wp": 0.52},
                        {"uci": "c7c5", "cp": 20, "wp": 0.51},
                    ],
                },
                {
                    "eval_wp": 0.55,
                    "top_k_moves": [
                        {"uci": "g1f3", "cp": 40, "wp": 0.55},
                        {"uci": "f1c4", "cp": 30, "wp": 0.53},
                    ],
                },
            ],
            "white_elo": 1850,
            "black_elo": 1750,
            "result": "1-0",
        },
        {
            "fens": [
                "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
                "r1bqkbnr/pppppppp/2n5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
            ],
            "sf_labels": [
                {
                    "eval_wp": 0.54,
                    "top_k_moves": [
                        {"uci": "d2d4", "cp": 35, "wp": 0.54},
                    ],
                },
                {
                    "eval_wp": 0.49,
                    "top_k_moves": [
                        {"uci": "g8f6", "cp": 10, "wp": 0.49},
                    ],
                },
            ],
            "white_elo": 1350,
            "black_elo": 1450,
            "result": "0-1",
        },
    ]
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------
# soft_policy_target
# ------------------------------------------------------------------


class TestSoftPolicyTarget:
    def test_sums_to_one(self):
        fen = SAMPLE_LABELS[0]["fen"]
        board = chess.Board(fen)
        num_legal = len(list(board.legal_moves))
        probs = soft_policy_target(SAMPLE_LABELS[0]["top_k_moves"], fen, num_legal)
        assert probs.shape == (num_legal,)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_nonnegative(self):
        fen = SAMPLE_LABELS[0]["fen"]
        board = chess.Board(fen)
        num_legal = len(list(board.legal_moves))
        probs = soft_policy_target(SAMPLE_LABELS[0]["top_k_moves"], fen, num_legal)
        assert (probs >= 0).all()

    def test_matched_moves_have_mass(self):
        fen = SAMPLE_LABELS[0]["fen"]
        board = chess.Board(fen)
        legal_ucis = [m.uci() for m in board.legal_moves]
        num_legal = len(legal_ucis)
        probs = soft_policy_target(SAMPLE_LABELS[0]["top_k_moves"], fen, num_legal)
        # The top-k moves should carry almost all probability mass
        top_k_ucis = {m["uci"] for m in SAMPLE_LABELS[0]["top_k_moves"]}
        matched_mass = sum(
            probs[i].item()
            for i, u in enumerate(legal_ucis)
            if u in top_k_ucis
        )
        assert matched_mass > 0.99

    def test_lower_temperature_sharper(self):
        fen = SAMPLE_LABELS[0]["fen"]
        board = chess.Board(fen)
        num_legal = len(list(board.legal_moves))
        probs_warm = soft_policy_target(SAMPLE_LABELS[0]["top_k_moves"], fen, num_legal, temperature=10.0)
        probs_cold = soft_policy_target(SAMPLE_LABELS[0]["top_k_moves"], fen, num_legal, temperature=0.1)
        # Cold distribution should have higher max
        assert probs_cold.max() > probs_warm.max()


class TestHumanPolicyUtilities:
    def test_legal_move_index_map_covers_legal_moves(self):
        fen = SAMPLE_LABELS[0]["fen"]
        board = chess.Board(fen)
        mapping = legal_move_index_map(fen)
        assert len(mapping) == len(list(board.legal_moves))

    def test_hard_policy_target_is_one_hot(self):
        fen = SAMPLE_LABELS[0]["fen"]
        board = chess.Board(fen)
        num_legal = len(list(board.legal_moves))
        target = hard_policy_target("e7e5", fen, num_legal)
        assert target.shape == (num_legal,)
        assert target.sum().item() == pytest.approx(1.0)
        assert target.max().item() == pytest.approx(1.0)

    def test_infer_played_move_uci_from_consecutive_fens(self):
        current_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        next_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        assert infer_played_move_uci(current_fen, next_fen) == "e7e5"


# ------------------------------------------------------------------
# DistillationDataset
# ------------------------------------------------------------------


class TestDistillationDataset:
    def test_len_and_item_keys(self, tmp_path):
        path = str(tmp_path / "labels.jsonl")
        _write_sample_jsonl(path)
        ds = DistillationDataset(path)
        assert len(ds) == 3
        item = ds[0]
        assert "graph" in item
        assert "value_target" in item
        assert "policy_target" in item
        assert "num_legal_moves" in item

    def test_graph_has_move_edges(self, tmp_path):
        path = str(tmp_path / "labels.jsonl")
        _write_sample_jsonl(path)
        ds = DistillationDataset(path)
        item = ds[0]
        graph = item["graph"]
        assert ("piece", "move", "square") in graph.edge_index_dict
        M = graph["piece", "move", "square"].edge_index.shape[1]
        assert M > 0
        assert item["num_legal_moves"] == M

    def test_value_target_range(self, tmp_path):
        path = str(tmp_path / "labels.jsonl")
        _write_sample_jsonl(path)
        ds = DistillationDataset(path)
        for i in range(len(ds)):
            vt = ds[i]["value_target"]
            assert vt.shape == (1,)
            assert -1.0 <= vt.item() <= 1.0

    def test_policy_target_shape_matches_moves(self, tmp_path):
        path = str(tmp_path / "labels.jsonl")
        _write_sample_jsonl(path)
        ds = DistillationDataset(path)
        for i in range(len(ds)):
            item = ds[i]
            assert item["policy_target"].shape == (item["num_legal_moves"],)
            assert item["policy_target"].sum().item() == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------


class TestDistillationCollate:
    def test_passthrough(self, tmp_path):
        path = str(tmp_path / "labels.jsonl")
        _write_sample_jsonl(path)
        ds = DistillationDataset(path)
        batch = [ds[0], ds[1]]
        out = distillation_collate(batch)
        assert len(out) == 2
        assert out[0]["graph"].fen == SAMPLE_LABELS[0]["fen"]


# ------------------------------------------------------------------
# Forward pass with GATEAUChessModel
# ------------------------------------------------------------------


class TestForwardWithQ:
    def test_shapes(self, tmp_path):
        path = str(tmp_path / "labels.jsonl")
        _write_sample_jsonl(path)
        builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
        ds = DistillationDataset(path, graph_builder=builder)
        metadata = builder.get_metadata()
        model = GATEAUChessModel(metadata, hidden_channels=32, num_layers=2)

        item = ds[0]
        graph = item["graph"]
        value, q_scores, edge_idx = model.forward_with_q(graph)

        M = item["num_legal_moves"]
        assert value.shape == (1, 1)
        assert q_scores.shape == (M,)
        assert edge_idx.shape == (2, M)
        assert torch.isfinite(q_scores).all()


class TestGameSequenceOfflineDataset:
    def test_filters_games_by_average_elo(self, tmp_path):
        path = str(tmp_path / "games.jsonl")
        _write_sample_game_jsonl(path)

        ds = GameSequenceOfflineDataset(path, elo_min=1600)
        games = list(ds)

        assert len(ds) == 1
        assert len(games) == 1
        assert games[0]["elo_norm"] == pytest.approx((1850 + 1750) / 2.0 / 3000.0)
        assert len(games[0]["human_policy_targets"]) == 2
        assert games[0]["human_policy_targets"][0].sum().item() == pytest.approx(1.0)
        assert games[0]["human_policy_targets"][1].numel() == 0

    def test_dataloader_yields_filtered_sequence(self, tmp_path):
        path = str(tmp_path / "games_loader.jsonl")
        _write_sample_game_jsonl(path)

        ds = GameSequenceOfflineDataset(path, elo_min=1600)
        loader = DataLoader(ds, batch_size=1, collate_fn=lambda batch: batch[0], num_workers=0)
        sample = next(iter(loader))

        assert len(sample["graphs"]) == 2
        assert sample["result_value"] == 1.0
        assert sample["human_policy_targets"][0].sum().item() == pytest.approx(1.0)
