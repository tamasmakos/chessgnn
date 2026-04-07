"""Tests for chessgnn/eval.py — evaluation harness."""

import csv
import json
import os
import tempfile

import chess
import pytest
import torch

from chessgnn.eval import (
    Evaluator,
    _extract_scalar,
    _pearson_r,
    _spearman_rho,
    compare_models,
)
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STARTING_FEN = chess.STARTING_FEN
SICILIAN_FEN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

SAMPLE_LABELS = [
    {
        "fen": STARTING_FEN,
        "eval_wp": 0.50,
        "top_k_moves": [
            {"uci": "e2e4", "cp": 20, "wp": 0.52},
            {"uci": "d2d4", "cp": 15, "wp": 0.51},
            {"uci": "g1f3", "cp": 10, "wp": 0.50},
        ],
    },
    {
        "fen": SICILIAN_FEN,
        "eval_wp": 0.55,
        "top_k_moves": [
            {"uci": "g1f3", "cp": 40, "wp": 0.55},
            {"uci": "d2d4", "cp": 30, "wp": 0.53},
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

SAMPLE_GAMES = [
    {
        "fens": [
            STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ],
        "white_elo": 1800,
        "black_elo": 1700,
        "result": "1-0",
    }
]

# A minimal Lichess puzzle (FEN before opponent blunder + solution moves)
# From board after 1.e4 e5 - a simple king pawn game position
# Here we just construct a puzzle-like tuple; correctness of chess logic is
# validated separately.
_PUZZLE_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
# opponent move: black plays d7d6; then white finds Ng5 (g1g5? no...) anything - we just test structure


def _write_sample_jsonl(path: str) -> None:
    with open(path, "w") as f:
        for rec in SAMPLE_LABELS:
            f.write(json.dumps(rec) + "\n")


def _write_sample_games_jsonl(path: str) -> None:
    with open(path, "w") as f:
        for rec in SAMPLE_GAMES:
            f.write(json.dumps(rec) + "\n")


def _write_puzzle_csv(path: str, with_header: bool = True) -> None:
    """Write a tiny puzzle CSV (moves[0]=opponent move, moves[1]=solution)."""
    # Use starting position: apply e2e4, then model should find d7d5.
    # PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags
    puzzle_fen = chess.STARTING_FEN
    moves = "e2e4 d7d5"  # opponent plays e2e4 (setup), model must find d7d5
    row = ["puzzle001", puzzle_fen, moves, "1500", "80", "95", "10000", "opening", "url", ""]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if with_header:
            writer.writerow(
                ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
                 "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"]
            )
        writer.writerow(row)


@pytest.fixture(scope="module")
def builder():
    return ChessGraphBuilder(use_global_node=True, use_move_edges=True)


@pytest.fixture(scope="module")
def gateau_model(builder):
    metadata = builder.get_metadata()
    m = GATEAUChessModel(metadata, hidden_channels=32, num_layers=2)
    m.eval()
    return m


@pytest.fixture(scope="module")
def evaluator(gateau_model):
    return Evaluator(gateau_model, device=torch.device("cpu"), use_global_node=True)


# ---------------------------------------------------------------------------
# _extract_scalar
# ---------------------------------------------------------------------------


class TestExtractScalar:
    def test_tensor_input(self):
        t = torch.tensor([[0.5]])
        assert _extract_scalar(t) == pytest.approx(0.5)

    def test_tuple_input_white_winning(self):
        # High white logit → positive scalar
        logits = torch.tensor([5.0, 0.0, -5.0])
        result = _extract_scalar((logits, None, None))
        assert result > 0

    def test_tuple_input_near_zero(self):
        logits = torch.tensor([0.0, 0.0, 0.0])
        result = _extract_scalar((logits, None, None))
        assert result == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------


class TestCorrelationHelpers:
    def test_pearson_perfect(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        import numpy as np

        assert _pearson_r(np.array(x), np.array(x)) == pytest.approx(1.0)

    def test_pearson_anti(self):
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _pearson_r(x, -x) == pytest.approx(-1.0)

    def test_pearson_constant_returns_zero(self):
        import numpy as np

        x = np.zeros(5)
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        assert _pearson_r(x, y) == pytest.approx(0.0)

    def test_spearman_perfect(self):
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _spearman_rho(x, x) == pytest.approx(1.0, abs=1e-5)

    def test_spearman_monotone_nonlinear(self):
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x**2  # monotone but nonlinear
        # Pearson < 1, Spearman = 1
        assert _spearman_rho(x, y) == pytest.approx(1.0, abs=1e-5)
        assert _pearson_r(x, y) < 1.0


# ---------------------------------------------------------------------------
# Evaluator.__init__
# ---------------------------------------------------------------------------


class TestEvaluatorInit:
    def test_has_q_head_flag_set(self, gateau_model):
        ev = Evaluator(gateau_model)
        assert ev._has_q_head is True

    def test_builder_has_move_edges(self, gateau_model):
        ev = Evaluator(gateau_model)
        # Builder should have use_move_edges=True for models with Q-head
        assert ev.builder.use_move_edges is True

    def test_model_on_device(self, gateau_model):
        ev = Evaluator(gateau_model, device=torch.device("cpu"))
        for p in ev.model.parameters():
            assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# Evaluator._pick_best_move_uci
# ---------------------------------------------------------------------------


class TestPickBestMoveUci:
    def test_returns_legal_move(self, evaluator):
        uci = evaluator._pick_best_move_uci(STARTING_FEN)
        assert uci is not None
        board = chess.Board(STARTING_FEN)
        legal_ucis = {m.uci() for m in board.legal_moves}
        assert uci in legal_ucis

    def test_returns_none_for_no_legal_moves(self, evaluator):
        # Stalemate position: black king on a1, white king on b3, white queen on c2 — stalemate
        stalemate_fen = "8/8/8/8/8/1K6/2Q5/k7 b - - 0 1"
        board = chess.Board(stalemate_fen)
        if not list(board.legal_moves):
            result = evaluator._pick_best_move_uci(stalemate_fen)
            assert result is None

    def test_deterministic_given_same_weights(self, evaluator):
        uci1 = evaluator._pick_best_move_uci(STARTING_FEN)
        uci2 = evaluator._pick_best_move_uci(STARTING_FEN)
        assert uci1 == uci2


# ---------------------------------------------------------------------------
# Evaluator._predict_value
# ---------------------------------------------------------------------------


class TestPredictValue:
    def test_returns_probability_in_unit_interval(self, evaluator):
        v = evaluator._predict_value(STARTING_FEN)
        assert 0.0 <= v <= 1.0

    def test_valid_for_different_positions(self, evaluator):
        for rec in SAMPLE_LABELS:
            v = evaluator._predict_value(rec["fen"])
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# evaluate_engine_agreement
# ---------------------------------------------------------------------------


class TestEvaluateEngineAgreement:
    def test_returns_expected_keys(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            result = evaluator.evaluate_engine_agreement(tf.name, k=3)
        os.unlink(tf.name)
        assert "top1_acc" in result
        assert "top3_acc" in result
        assert "count" in result

    def test_count_matches_input(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            result = evaluator.evaluate_engine_agreement(tf.name, k=1)
        os.unlink(tf.name)
        assert result["count"] == len(SAMPLE_LABELS)

    def test_accuracy_in_unit_interval(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            result = evaluator.evaluate_engine_agreement(tf.name, k=1)
        os.unlink(tf.name)
        assert 0.0 <= result["top1_acc"] <= 1.0

    def test_topk_ge_top1(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            result = evaluator.evaluate_engine_agreement(tf.name, k=3)
        os.unlink(tf.name)
        assert result["top3_acc"] >= result["top1_acc"]

    def test_empty_jsonl_returns_zero_count(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            tf.write("")
            result = evaluator.evaluate_engine_agreement(tf.name, k=1)
        os.unlink(tf.name)
        assert result["count"] == 0


class TestEvaluateHumanMovePrediction:
    def test_returns_expected_keys(self, evaluator, monkeypatch):
        monkeypatch.setattr(
            evaluator,
            "_pick_topk_moves_uci",
            lambda fen, k=3: ["e2e4", "d2d4", "g1f3"] if fen == STARTING_FEN else ["e7e5", "c7c5", "d7d5"],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
            _write_sample_games_jsonl(tf.name)
            result = evaluator.evaluate_human_move_prediction(tf.name, k=3)
        os.unlink(tf.name)
        assert "top1_acc" in result
        assert "top3_acc" in result
        assert "count" in result

    def test_count_matches_move_transitions(self, evaluator, monkeypatch):
        monkeypatch.setattr(
            evaluator,
            "_pick_topk_moves_uci",
            lambda fen, k=3: ["e2e4", "d2d4", "g1f3"] if fen == STARTING_FEN else ["e7e5", "c7c5", "d7d5"],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
            _write_sample_games_jsonl(tf.name)
            result = evaluator.evaluate_human_move_prediction(tf.name, k=3)
        os.unlink(tf.name)
        assert result["count"] == 2

    def test_topk_ge_top1(self, evaluator, monkeypatch):
        monkeypatch.setattr(
            evaluator,
            "_pick_topk_moves_uci",
            lambda fen, k=3: ["a2a3", "e2e4", "d2d4"] if fen == STARTING_FEN else ["a7a6", "e7e5", "c7c5"],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
            _write_sample_games_jsonl(tf.name)
            result = evaluator.evaluate_human_move_prediction(tf.name, k=3)
        os.unlink(tf.name)
        assert result["top3_acc"] >= result["top1_acc"]


# ---------------------------------------------------------------------------
# evaluate_value_correlation
# ---------------------------------------------------------------------------


class TestEvaluateValueCorrelation:
    def test_returns_expected_keys(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            result = evaluator.evaluate_value_correlation(tf.name)
        os.unlink(tf.name)
        assert "pearson_r" in result
        assert "spearman_rho" in result
        assert "count" in result

    def test_count_matches_input(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            result = evaluator.evaluate_value_correlation(tf.name)
        os.unlink(tf.name)
        assert result["count"] == len(SAMPLE_LABELS)

    def test_correlations_in_valid_range(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            result = evaluator.evaluate_value_correlation(tf.name)
        os.unlink(tf.name)
        assert -1.0 <= result["pearson_r"] <= 1.0
        assert -1.0 <= result["spearman_rho"] <= 1.0

    def test_too_few_samples_returns_zeros(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            tf.write(json.dumps(SAMPLE_LABELS[0]) + "\n")
            result = evaluator.evaluate_value_correlation(tf.name)
        os.unlink(tf.name)
        assert result["pearson_r"] == pytest.approx(0.0)
        assert result["spearman_rho"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_puzzles
# ---------------------------------------------------------------------------


class TestEvaluatePuzzles:
    def test_with_header_returns_correct_keys(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tf:
            _write_puzzle_csv(tf.name, with_header=True)
            result = evaluator.evaluate_puzzles(tf.name)
        os.unlink(tf.name)
        assert "accuracy" in result
        assert "count" in result
        assert "solved" in result

    def test_without_header_returns_correct_keys(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tf:
            _write_puzzle_csv(tf.name, with_header=False)
            result = evaluator.evaluate_puzzles(tf.name)
        os.unlink(tf.name)
        assert "accuracy" in result
        assert "count" in result

    def test_empty_csv_returns_zero(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tf:
            tf.write("")
            result = evaluator.evaluate_puzzles(tf.name)
        os.unlink(tf.name)
        assert result["count"] == 0
        assert result["accuracy"] == pytest.approx(0.0)

    def test_accuracy_in_unit_interval(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tf:
            _write_puzzle_csv(tf.name, with_header=True)
            result = evaluator.evaluate_puzzles(tf.name)
        os.unlink(tf.name)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_n_limit_respected(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tf:
            _write_puzzle_csv(tf.name, with_header=True)
            result = evaluator.evaluate_puzzles(tf.name, n=0)
        os.unlink(tf.name)
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------


class TestReliabilityDiagram:
    def test_saves_png(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf_data:
            _write_sample_jsonl(tf_data.name)
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tf_out:
                out_png = tf_out.name

        stats = evaluator.reliability_diagram(tf_data.name, out_png)
        os.unlink(tf_data.name)
        assert os.path.exists(out_png)
        os.unlink(out_png)
        assert isinstance(stats, list)

    def test_stats_have_expected_keys(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf_data:
            _write_sample_jsonl(tf_data.name)
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tf_out:
                out_png = tf_out.name

        stats = evaluator.reliability_diagram(tf_data.name, out_png, n_bins=5)
        os.unlink(tf_data.name)
        os.unlink(out_png)
        assert len(stats) == 5
        for s in stats:
            assert "bin_center" in s
            assert "mean_pred" in s
            assert "mean_target" in s
            assert "count" in s

    def test_bin_centers_in_unit_interval(self, evaluator):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf_data:
            _write_sample_jsonl(tf_data.name)
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tf_out:
                out_png = tf_out.name

        stats = evaluator.reliability_diagram(tf_data.name, out_png, n_bins=10)
        os.unlink(tf_data.name)
        os.unlink(out_png)
        for s in stats:
            assert 0.0 <= s["bin_center"] <= 1.0


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_result_per_model(self, gateau_model):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            results = compare_models(
                {"gateau": gateau_model},
                positions_jsonl=tf.name,
                device=torch.device("cpu"),
            )
        os.unlink(tf.name)
        assert "gateau" in results

    def test_multiple_models_independent(self, gateau_model):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tf:
            _write_sample_jsonl(tf.name)
            results = compare_models(
                {"model_a": gateau_model, "model_b": gateau_model},
                positions_jsonl=tf.name,
                device=torch.device("cpu"),
            )
        os.unlink(tf.name)
        assert set(results.keys()) == {"model_a", "model_b"}

    def test_no_positions_no_crash(self, gateau_model):
        """compare_models with no evaluation data should return empty dicts per model."""
        results = compare_models({"m": gateau_model})
        assert "m" in results
