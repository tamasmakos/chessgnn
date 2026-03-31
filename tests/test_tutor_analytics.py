"""Tests for CaseTutor.analyse_game() and CaseTutor.estimate_elo().

Fixtures use a real game extracted from the lichess 2013-01 PGN
(game 1: BFG9k 1639 vs mamalak 1403, French Defence, 12 moves, 1-0).
A randomly-initialised GATEAUChessModel with global_gru is used so the
tests run fast without loading a checkpoint; they verify output structure
and invariants rather than numeric quality.
"""

import json

import chess
import chess.pgn
import io
import pytest
import torch

from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel
from tutor import CaseTutor

# ---------------------------------------------------------------------------
# Real game from lichess_db_standard_rated_2013-01.pgn (game 1)
# BFG9k (1639, white) vs mamalak (1403, black). Result: 1-0
# ---------------------------------------------------------------------------
_PGN_GAME1 = """\
[Event "Rated Classical game"]
[Site "https://lichess.org/j1dkb5dw"]
[White "BFG9k"]
[Black "mamalak"]
[Result "1-0"]
[WhiteElo "1639"]
[BlackElo "1403"]

1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5
8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0
"""

_WHITE_ELO = 1639
_BLACK_ELO = 1403


def _parse_game(pgn_text: str) -> tuple[list[str], list[str]]:
    """Return (fens, ucis) from a PGN string. fens[i] precedes ucis[i]."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()
    fens, ucis = [board.fen()], []
    for move in game.mainline_moves():
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    return fens, ucis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tutor() -> CaseTutor:
    builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
    meta = builder.get_metadata()
    model = GATEAUChessModel(meta, hidden_channels=32, num_layers=2,
                              temporal_mode="global_gru")
    model.eval()
    return CaseTutor(model, device="cpu")


@pytest.fixture(scope="module")
def game_data() -> tuple[list[str], list[str]]:
    return _parse_game(_PGN_GAME1)


@pytest.fixture(scope="module")
def stats(tutor, game_data) -> dict:
    fens, ucis = game_data
    return tutor.analyse_game(fens, ucis,
                               elo_white=_WHITE_ELO, elo_black=_BLACK_ELO)


@pytest.fixture(scope="module")
def stats_no_moves(tutor, game_data) -> dict:
    """analyse_game without move attribution."""
    fens, _ = game_data
    return tutor.analyse_game(fens)


# ---------------------------------------------------------------------------
# Structure and basic sanity
# ---------------------------------------------------------------------------

class TestAnalyseGameStructure:
    def test_required_keys_present(self, stats: dict) -> None:
        required = {
            "n_positions", "n_moves",
            "eval_trajectory", "uncertainty_trajectory",
            "legal_moves_trajectory", "piece_count_trajectory",
            "center_pressure_trajectory", "q_gini_trajectory",
            "move_ranks", "move_percentiles", "eval_drops",
            "white", "black",
            "avg_complexity", "decisiveness", "avg_piece_count",
            "avg_branching", "game_sharpness",
            "accumulated_dest_heatmap", "accumulated_src_heatmap",
            "white_territory", "black_territory",
        }
        assert required.issubset(stats.keys())

    def test_json_serialisable(self, stats: dict) -> None:
        """All values must be plain Python types."""
        json.dumps(stats)

    def test_n_positions_matches_fens(self, stats: dict, game_data) -> None:
        fens, _ = game_data
        # analyse_game stops at terminal or exhausts fens; last fen is checkmate
        # so n_positions == len(fens) - 1 (no legal moves in final position)
        assert stats["n_positions"] == len(fens) - 1

    def test_n_moves_matches_ucis(self, stats: dict, game_data) -> None:
        _, ucis = game_data
        assert stats["n_moves"] == len(ucis)

    def test_trajectory_lengths(self, stats: dict) -> None:
        n = stats["n_positions"]
        for key in ("eval_trajectory", "uncertainty_trajectory",
                    "legal_moves_trajectory", "piece_count_trajectory",
                    "center_pressure_trajectory", "q_gini_trajectory"):
            assert len(stats[key]) == n, f"{key} has wrong length"

    def test_move_attribution_lengths(self, stats: dict) -> None:
        n_moves = stats["n_moves"]
        for key in ("move_ranks", "move_percentiles", "eval_drops"):
            assert stats[key] is not None
            assert len(stats[key]) == n_moves, f"{key} length mismatch"


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------

class TestValueRanges:
    def test_eval_in_range(self, stats: dict) -> None:
        for v in stats["eval_trajectory"]:
            assert -1.0 <= v <= 1.0, f"eval {v} out of [-1, 1]"

    def test_uncertainty_in_range(self, stats: dict) -> None:
        for u in stats["uncertainty_trajectory"]:
            assert 0.0 <= u <= 1.0, f"uncertainty {u} out of [0, 1]"

    def test_center_pressure_in_range(self, stats: dict) -> None:
        for p in stats["center_pressure_trajectory"]:
            assert 0.0 <= p <= 1.0, f"center_pressure {p} out of [0, 1]"

    def test_q_gini_in_range(self, stats: dict) -> None:
        for g in stats["q_gini_trajectory"]:
            assert 0.0 <= g <= 1.0, f"gini {g} out of [0, 1]"

    def test_legal_moves_positive(self, stats: dict) -> None:
        for m in stats["legal_moves_trajectory"]:
            assert m > 0, "legal move count must be positive"

    def test_piece_count_positive_and_decreasing_or_equal(self, stats: dict) -> None:
        counts = stats["piece_count_trajectory"]
        assert all(c > 0 for c in counts)
        # pieces can only leave (captures), never return mid-game
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], (
                f"piece count increased at move {i}: {counts[i-1]} → {counts[i]}"
            )

    def test_first_position_piece_count_32(self, stats: dict) -> None:
        assert stats["piece_count_trajectory"][0] == 32

    def test_move_ranks_positive_int(self, stats: dict) -> None:
        for r in stats["move_ranks"]:
            if r is not None:
                assert isinstance(r, int) and r >= 1

    def test_move_percentiles_in_range(self, stats: dict) -> None:
        for p in stats["move_percentiles"]:
            if p is not None:
                assert 0.0 <= p <= 1.0

    def test_decisiveness_in_range(self, stats: dict) -> None:
        assert 0.0 <= stats["decisiveness"] <= 1.0

    def test_game_sharpness_nonnegative(self, stats: dict) -> None:
        assert stats["game_sharpness"] >= 0.0

    def test_territory_sums_to_one(self, stats: dict) -> None:
        total = stats["white_territory"] + stats["black_territory"]
        assert abs(total - 1.0) < 1e-4, f"territory sum {total} != 1.0"

    def test_heatmap_shape(self, stats: dict) -> None:
        for key in ("accumulated_dest_heatmap", "accumulated_src_heatmap"):
            hm = stats[key]
            assert len(hm) == 8, f"{key} must have 8 ranks"
            for rank in hm:
                assert len(rank) == 8, f"{key} rank must have 8 files"

    def test_heatmap_values_normalised(self, stats: dict) -> None:
        for key in ("accumulated_dest_heatmap", "accumulated_src_heatmap"):
            flat = [v for row in stats[key] for v in row]
            assert max(flat) <= 1.0 + 1e-6
            assert min(flat) >= 0.0 - 1e-6
            # at least one non-zero entry (we played moves)
            assert max(flat) > 0.0, f"{key} is all-zero"


# ---------------------------------------------------------------------------
# Per-side stats
# ---------------------------------------------------------------------------

class TestPerSideStats:
    def test_white_black_dicts_present(self, stats: dict) -> None:
        assert isinstance(stats["white"], dict)
        assert isinstance(stats["black"], dict)

    def test_per_side_keys(self, stats: dict) -> None:
        expected = {
            "moves_played", "avg_move_rank", "avg_move_percentile",
            "agreement_top1", "agreement_top3",
            "blunders", "mistakes", "best_moves", "avg_uncertainty_faced",
        }
        for side in ("white", "black"):
            assert expected.issubset(stats[side].keys()), \
                f"{side} dict missing keys"

    def test_moves_played_sum(self, stats: dict) -> None:
        total = stats["white"]["moves_played"] + stats["black"]["moves_played"]
        assert total == stats["n_moves"]

    def test_agreement_top3_ge_top1(self, stats: dict) -> None:
        for side in ("white", "black"):
            a1 = stats[side]["agreement_top1"]
            a3 = stats[side]["agreement_top3"]
            if a1 is not None and a3 is not None:
                assert a3 >= a1 - 1e-6, f"{side}: top3 < top1"

    def test_blunders_are_list(self, stats: dict) -> None:
        for side in ("white", "black"):
            assert isinstance(stats[side]["blunders"], list)

    def test_blunder_eval_drop_threshold(self, stats: dict) -> None:
        for side in ("white", "black"):
            for b in stats[side]["blunders"]:
                assert b["eval_drop"] > 0.15 - 1e-6


# ---------------------------------------------------------------------------
# Without move attribution
# ---------------------------------------------------------------------------

class TestNoMoveAttribution:
    def test_move_keys_none_without_ucis(self, stats_no_moves: dict) -> None:
        assert stats_no_moves["move_ranks"] is None
        assert stats_no_moves["move_percentiles"] is None
        assert stats_no_moves["eval_drops"] is None

    def test_per_side_none_without_ucis(self, stats_no_moves: dict) -> None:
        assert stats_no_moves["white"] is None
        assert stats_no_moves["black"] is None

    def test_trajectories_still_populated(self, stats_no_moves: dict) -> None:
        assert len(stats_no_moves["eval_trajectory"]) > 0


# ---------------------------------------------------------------------------
# estimate_elo
# ---------------------------------------------------------------------------

class TestEstimateElo:
    def test_returns_expected_keys(self, tutor, stats) -> None:
        result = tutor.estimate_elo(stats, "white")
        assert "estimated_elo" in result
        assert "confidence_range" in result
        assert "features" in result
        assert "note" in result

    def test_elo_in_clamped_range(self, tutor, stats) -> None:
        for side in ("white", "black"):
            result = tutor.estimate_elo(stats, side)
            elo = result["estimated_elo"]
            assert 400 <= elo <= 3200, f"{side}: elo {elo} out of clamped range"

    def test_confidence_range_width(self, tutor, stats) -> None:
        result = tutor.estimate_elo(stats, "white")
        lo, hi = result["confidence_range"]
        assert hi - lo == 600  # always ±300

    def test_confidence_centred_on_estimate(self, tutor, stats) -> None:
        result = tutor.estimate_elo(stats, "white")
        elo = result["estimated_elo"]
        lo, hi = result["confidence_range"]
        assert lo == elo - 300 and hi == elo + 300

    def test_feature_keys(self, tutor, stats) -> None:
        feats = tutor.estimate_elo(stats, "white")["features"]
        expected = {
            "avg_move_percentile", "agreement_top1", "agreement_top3",
            "blunder_rate", "mistake_rate", "n_moves",
        }
        assert expected.issubset(feats.keys())

    def test_graceful_no_attribution(self, tutor, stats_no_moves) -> None:
        result = tutor.estimate_elo(stats_no_moves, "white")
        assert result["estimated_elo"] is None

    def test_json_serialisable(self, tutor, stats) -> None:
        result = tutor.estimate_elo(stats, "white")
        json.dumps(result)
