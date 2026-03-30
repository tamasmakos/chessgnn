"""Tests for win-probability evaluation against real game outcomes.

Covers:
- brier_score() and log_loss_metric() module-level helpers
- Evaluator.evaluate_pgn_outcomes() method
"""

import io
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

import chess
import chess.pgn

from chessgnn.eval import Evaluator, brier_score, log_loss_metric
from chessgnn.model import GATEAUChessModel
from chessgnn.graph_builder import ChessGraphBuilder

# ---------------------------------------------------------------------------
# Minimal PGN helpers
# ---------------------------------------------------------------------------

_GAME_WHITE_WIN = """\
[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0
"""

_GAME_BLACK_WIN = """\
[Event "Test"]
[White "A"]
[Black "B"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 0-1
"""

_GAME_DRAW = """\
[Event "Test"]
[White "A"]
[Black "B"]
[Result "1/2-1/2"]

1. e4 e5 1/2-1/2
"""

_GAME_UNKNOWN = """\
[Event "Test"]
[White "A"]
[Black "B"]
[Result "*"]

1. e4 *
"""

_TWO_GAMES_PGN = _GAME_WHITE_WIN + "\n" + _GAME_BLACK_WIN


def _write_pgn(content: str):
    """Write PGN string to a temp file; return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name


def _count_positions(pgn_str: str) -> int:
    """Count total positions (one per half-move) across all games in a PGN string."""
    count = 0
    buf = io.StringIO(pgn_str)
    while True:
        game = chess.pgn.read_game(buf)
        if game is None:
            break
        # Skip games with unknown result
        if game.headers.get("Result", "*") == "*":
            continue
        count += sum(1 for _ in game.mainline_moves())
    return count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model():
    builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
    model = GATEAUChessModel(
        builder.get_metadata(),
        hidden_channels=16,
        num_layers=1,
        temporal_mode="none",
    )
    model.eval()
    return model


@pytest.fixture
def evaluator(tiny_model):
    return Evaluator(tiny_model, device=torch.device("cpu"), use_global_node=True)


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------


def test_brier_score_perfect():
    probs = np.array([1.0, 0.0, 0.5])
    outcomes = np.array([1.0, 0.0, 0.5])
    assert brier_score(probs, outcomes) == pytest.approx(0.0)


def test_brier_score_worst_binary():
    probs = np.array([0.0, 1.0])
    outcomes = np.array([1.0, 0.0])
    assert brier_score(probs, outcomes) == pytest.approx(1.0)


def test_brier_score_constant_prediction():
    probs = np.full(10, 0.5)
    outcomes = np.array([1.0, 0.0] * 5)
    assert brier_score(probs, outcomes) == pytest.approx(0.25)


def test_brier_score_empty():
    assert brier_score(np.array([]), np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# log_loss_metric
# ---------------------------------------------------------------------------


def test_log_loss_near_zero_for_confident_correct_predictions():
    probs = np.array([0.999, 0.001])
    outcomes = np.array([1.0, 0.0])
    assert log_loss_metric(probs, outcomes) < 0.02


def test_log_loss_draw_outcome():
    # outcome=0.5, pred=0.5 should give -log(0.5) ≈ 0.693
    probs = np.array([0.5])
    outcomes = np.array([0.5])
    expected = -0.5 * np.log(0.5) - 0.5 * np.log(0.5)
    assert log_loss_metric(probs, outcomes) == pytest.approx(float(expected), rel=1e-5)


def test_log_loss_no_inf_at_boundaries():
    # pred=0 and pred=1 must not produce inf due to eps clipping
    probs = np.array([0.0, 1.0])
    outcomes = np.array([0.0, 1.0])
    result = log_loss_metric(probs, outcomes)
    assert np.isfinite(result)


def test_log_loss_empty():
    assert log_loss_metric(np.array([]), np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_pgn_outcomes
# ---------------------------------------------------------------------------


def test_evaluate_pgn_outcomes_basic(evaluator):
    path = _write_pgn(_TWO_GAMES_PGN)
    n_expected = _count_positions(_TWO_GAMES_PGN)

    result = evaluator.evaluate_pgn_outcomes(path)

    assert result["n_games"] == 2
    assert result["n_positions"] == n_expected
    assert 0.0 <= result["brier_score"] <= 1.0
    assert result["log_loss"] >= 0.0
    assert 0.0 <= result["ece"] <= 1.0


def test_evaluate_pgn_outcomes_returns_all_required_keys(evaluator):
    path = _write_pgn(_GAME_WHITE_WIN)
    result = evaluator.evaluate_pgn_outcomes(path)
    for key in ("brier_score", "log_loss", "ece", "n_positions", "n_games"):
        assert key in result


def test_evaluate_pgn_outcomes_skip_unknown_result(evaluator):
    combined = _GAME_WHITE_WIN + "\n" + _GAME_UNKNOWN
    path = _write_pgn(combined)
    result = evaluator.evaluate_pgn_outcomes(path)
    # Only the white-win game (6 half-moves) should be counted
    assert result["n_games"] == 1
    assert result["n_positions"] == _count_positions(combined)


def test_evaluate_pgn_outcomes_max_games(evaluator):
    path = _write_pgn(_TWO_GAMES_PGN)
    result_full = evaluator.evaluate_pgn_outcomes(path, max_games=0)
    result_one = evaluator.evaluate_pgn_outcomes(path, max_games=1)
    assert result_one["n_games"] == 1
    assert result_one["n_positions"] < result_full["n_positions"]


def test_evaluate_pgn_outcomes_empty_pgn(evaluator):
    path = _write_pgn("")
    result = evaluator.evaluate_pgn_outcomes(path)
    assert result == {
        "brier_score": 0.0,
        "log_loss": 0.0,
        "ece": 0.0,
        "n_positions": 0,
        "n_games": 0,
    }


def test_evaluate_pgn_outcomes_draw_outcome(evaluator):
    path = _write_pgn(_GAME_DRAW)
    result = evaluator.evaluate_pgn_outcomes(path)
    assert result["n_games"] == 1
    assert result["n_positions"] == _count_positions(_GAME_DRAW)


def test_evaluate_pgn_white_win_outcome_used(evaluator):
    """Mock _predict_value to a constant; verify Brier score is deterministic."""
    path = _write_pgn(_GAME_WHITE_WIN)
    # Ground truth: all outcomes = 1.0 (white wins)
    # Prediction: 0.7 → Brier = (0.7 - 1.0)^2 = 0.09
    with patch.object(evaluator, "_predict_value", return_value=0.7):
        result = evaluator.evaluate_pgn_outcomes(path)

    n = result["n_positions"]
    assert n > 0
    assert result["brier_score"] == pytest.approx(0.09, rel=1e-5)


def test_evaluate_pgn_black_win_outcome_used(evaluator):
    """Outcome = 0.0 for black-win game; constant pred=0.3 → Brier=(0.3-0)^2=0.09."""
    path = _write_pgn(_GAME_BLACK_WIN)
    with patch.object(evaluator, "_predict_value", return_value=0.3):
        result = evaluator.evaluate_pgn_outcomes(path)

    assert result["n_positions"] > 0
    assert result["brier_score"] == pytest.approx(0.09, rel=1e-5)
