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
            "structural_fingerprint_trajectory", "structural_drift_trajectory",
            "avg_structural_drift", "peak_structural_drift",
            "final_structural_distance",
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


# ---------------------------------------------------------------------------
# Structural trajectories (new graph-theoretic features)
# ---------------------------------------------------------------------------

class TestStructuralTrajectories:
    """Verify the six new per-position structural trajectory keys."""

    _TRAJ_KEYS = (
        "coordination_trajectory",
        "centrality_trajectory",
        "community_count_trajectory",
        "tension_trajectory",
        "pin_count_trajectory",
        "fork_count_trajectory",
    )

    def test_structural_keys_present(self, stats: dict) -> None:
        for key in self._TRAJ_KEYS:
            assert key in stats, f"Missing key: {key}"

    def test_structural_trajectory_lengths(self, stats: dict) -> None:
        n = stats["n_positions"]
        for key in self._TRAJ_KEYS:
            assert len(stats[key]) == n, f"{key} length mismatch"

    def test_coordination_in_range(self, stats: dict) -> None:
        for v in stats["coordination_trajectory"]:
            assert 0.0 <= v <= 1.0, f"coordination {v} out of [0, 1]"

    def test_centrality_in_range(self, stats: dict) -> None:
        for v in stats["centrality_trajectory"]:
            assert 0.0 <= v <= 1.0, f"centrality {v} out of [0, 1]"

    def test_community_count_positive(self, stats: dict) -> None:
        for v in stats["community_count_trajectory"]:
            assert isinstance(v, int) and v >= 1, f"community_count {v} invalid"

    def test_tension_in_range(self, stats: dict) -> None:
        for v in stats["tension_trajectory"]:
            assert 0.0 <= v <= 1.0, f"tension {v} out of [0, 1]"

    def test_pin_count_nonnegative(self, stats: dict) -> None:
        for v in stats["pin_count_trajectory"]:
            assert isinstance(v, int) and v >= 0

    def test_fork_count_nonnegative(self, stats: dict) -> None:
        for v in stats["fork_count_trajectory"]:
            assert isinstance(v, int) and v >= 0

    # Game-level aggregates
    def test_aggregate_keys_present(self, stats: dict) -> None:
        for key in ("avg_coordination", "avg_centrality", "avg_tension",
                    "peak_forks", "peak_pins"):
            assert key in stats, f"Missing aggregate key: {key}"

    def test_avg_coordination_in_range(self, stats: dict) -> None:
        assert 0.0 <= stats["avg_coordination"] <= 1.0

    def test_avg_centrality_in_range(self, stats: dict) -> None:
        assert 0.0 <= stats["avg_centrality"] <= 1.0

    def test_peak_forks_int(self, stats: dict) -> None:
        assert isinstance(stats["peak_forks"], int) and stats["peak_forks"] >= 0

    def test_peak_pins_int(self, stats: dict) -> None:
        assert isinstance(stats["peak_pins"], int) and stats["peak_pins"] >= 0

    def test_structural_json_serialisable(self, stats: dict) -> None:
        """New keys must contain only JSON-safe types."""
        subset = {k: stats[k] for k in self._TRAJ_KEYS}
        json.dumps(subset)


class TestStructuralFingerprints:
    def test_fingerprint_trajectory_present(self, stats: dict) -> None:
        assert "structural_fingerprint_trajectory" in stats

    def test_fingerprint_trajectory_length(self, stats: dict) -> None:
        assert len(stats["structural_fingerprint_trajectory"]) == stats["n_positions"]

    def test_fingerprint_dimensions_stable(self, stats: dict) -> None:
        traj = stats["structural_fingerprint_trajectory"]
        assert traj
        expected_dim = len(traj[0])
        assert expected_dim > 0
        for vector in traj:
            assert len(vector) == expected_dim

    def test_fingerprint_components_in_range(self, stats: dict) -> None:
        for vector in stats["structural_fingerprint_trajectory"]:
            for value in vector:
                assert 0.0 <= value <= 1.0

    def test_structural_drift_trajectory_present(self, stats: dict) -> None:
        assert "structural_drift_trajectory" in stats

    def test_structural_drift_length(self, stats: dict) -> None:
        assert len(stats["structural_drift_trajectory"]) == stats["n_positions"]

    def test_first_structural_drift_zero(self, stats: dict) -> None:
        assert stats["structural_drift_trajectory"][0] == pytest.approx(0.0, abs=1e-6)

    def test_structural_drift_in_range(self, stats: dict) -> None:
        for value in stats["structural_drift_trajectory"]:
            assert 0.0 <= value <= 1.0

    def test_structural_drift_aggregates_in_range(self, stats: dict) -> None:
        assert 0.0 <= stats["avg_structural_drift"] <= 1.0
        assert 0.0 <= stats["peak_structural_drift"] <= 1.0
        assert 0.0 <= stats["final_structural_distance"] <= 1.0

    def test_fingerprint_json_serialisable(self, stats: dict) -> None:
        subset = {
            "structural_fingerprint_trajectory": stats["structural_fingerprint_trajectory"],
            "structural_drift_trajectory": stats["structural_drift_trajectory"],
            "avg_structural_drift": stats["avg_structural_drift"],
            "peak_structural_drift": stats["peak_structural_drift"],
            "final_structural_distance": stats["final_structural_distance"],
        }
        json.dumps(subset)


class TestBuildExplainFingerprint:
    @pytest.fixture(scope="class")
    def explain_dict(self, tutor, game_data):
        fens, _ = game_data
        _, _, _, _, internals = tutor.recommend_move(fens[0], explain=True)
        return internals

    def test_structural_fingerprint_present(self, explain_dict: dict) -> None:
        assert "structural_fingerprint" in explain_dict

    def test_structural_fingerprint_nonempty(self, explain_dict: dict) -> None:
        fp = explain_dict["structural_fingerprint"]
        assert isinstance(fp, list)
        assert len(fp) > 0

    def test_structural_fingerprint_range(self, explain_dict: dict) -> None:
        for value in explain_dict["structural_fingerprint"]:
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Module-level tactical helpers
# ---------------------------------------------------------------------------

class TestDetectTactics:
    """Unit tests for _detect_tactics() on known tactical positions."""

    def test_returns_expected_keys(self) -> None:
        from tutor import _detect_tactics
        board = chess.Board()  # start position: no tactics
        result = _detect_tactics(board)
        assert set(result.keys()) == {
            "pins", "forks", "tension_squares", "overloaded_squares",
            "contested_count",
        }

    def test_start_position_no_pins_no_forks(self) -> None:
        from tutor import _detect_tactics
        result = _detect_tactics(chess.Board())
        assert result["pins"] == []
        assert result["forks"] == []

    def test_start_position_no_tension(self) -> None:
        from tutor import _detect_tactics
        # In the start position no piece of either side can reach the other
        result = _detect_tactics(chess.Board())
        assert result["contested_count"] == 0

    def test_pin_detected(self) -> None:
        """e3-bishop pins the d4-knight to the e1-king (Ruy López-style fixture)."""
        from tutor import _detect_tactics
        # Place white king e1, white knight d4; black bishop on h7 diagonally
        # pinning the knight to the king via the f5-g6-h7 diagonal is complex.
        # Use a well-known simple absolute pin instead:
        # White Ke1, Rd1; Black Qd8 — pin the rook along d-file to the king.
        # Actually simplest: construct a position where is_pinned() fires.
        # Fen: White Ke1 Ra1; Black Qa8 - rook is pinned along a-file? No.
        # Use: White Ke1, Bf3; Black Qh5 pins nothing.
        # Most reliable: 8/8/8/8/3r4/8/3R4/3K4 w - - 0 1
        # White Kd1, Rd2; Black Rd4 — Rd2 is pinned to Kd1 by Rd4.
        board = chess.Board("8/8/8/8/3r4/8/3R4/3K4 w - - 0 1")
        result = _detect_tactics(board)
        assert "d2" in result["pins"], f"Expected d2 pinned, got {result['pins']}"

    def test_fork_detected(self) -> None:
        """Knight on e5 attacks c4-rook and g4-queen (both higher-value than knight)."""
        from tutor import _detect_tactics
        # White knight e5; Black rook c4 (val=5), black queen g4 (val=9)
        # Knight value = 3; both victims strictly exceed it → fork fires.
        board = chess.Board("8/8/8/4N3/2r3q1/8/8/8 w - - 0 1")
        result = _detect_tactics(board)
        attacker_squares = {f["attacker"] for f in result["forks"]}
        assert "e5" in attacker_squares, f"Expected e5 fork, got {result['forks']}"

    def test_tension_squares_detected(self) -> None:
        """After 1.e4 e5 2.Nf3, d5/e4/e5 squares should be contested."""
        from tutor import _detect_tactics
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        result = _detect_tactics(board)
        assert result["contested_count"] > 0, "Expected contested squares after e4 e5"

    def test_json_serialisable(self) -> None:
        from tutor import _detect_tactics
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        json.dumps(_detect_tactics(board))


# ---------------------------------------------------------------------------
# _build_explain structural fields
# ---------------------------------------------------------------------------

class TestBuildExplainStructural:
    """Verify _build_explain returns the four new structural fields."""

    @pytest.fixture(scope="class")
    def explain_dict(self, tutor, game_data):
        fens, _ = game_data
        board = chess.Board(fens[0])
        legal_moves = list(board.legal_moves)
        # Use recommend_move with explain=True
        _, _, _, _, internals = tutor.recommend_move(fens[0], explain=True)
        return internals

    def test_tension_map_present(self, explain_dict: dict) -> None:
        assert "tension_map" in explain_dict

    def test_tension_map_shape(self, explain_dict: dict) -> None:
        t = explain_dict["tension_map"]
        assert len(t) == 8
        for row in t:
            assert len(row) == 8

    def test_tension_map_range(self, explain_dict: dict) -> None:
        for row in explain_dict["tension_map"]:
            for v in row:
                assert 0.0 <= v <= 1.0, f"tension_map value {v} out of [0, 1]"

    def test_pin_map_shape(self, explain_dict: dict) -> None:
        p = explain_dict["pin_map"]
        assert len(p) == 8
        for row in p:
            assert len(row) == 8

    def test_pin_map_start_all_false(self, explain_dict: dict) -> None:
        # No pieces are pinned in the starting position
        for row in explain_dict["pin_map"]:
            for cell in row:
                assert cell is False

    def test_community_groups_is_list(self, explain_dict: dict) -> None:
        assert isinstance(explain_dict["community_groups"], list)

    def test_community_groups_nonempty(self, explain_dict: dict) -> None:
        # Starting position has 32 pieces → at least 1 community
        assert len(explain_dict["community_groups"]) >= 1

    def test_piece_centrality_is_dict(self, explain_dict: dict) -> None:
        assert isinstance(explain_dict["piece_centrality"], dict)

    def test_piece_centrality_values_in_range(self, explain_dict: dict) -> None:
        for sq, c in explain_dict["piece_centrality"].items():
            assert 0.0 <= c <= 1.0, f"{sq}: centrality {c} out of [0, 1]"

    def test_explain_json_serialisable(self, explain_dict: dict) -> None:
        json.dumps(explain_dict)


# ---------------------------------------------------------------------------
# GNN piece embeddings in _build_explain (new keys)
# ---------------------------------------------------------------------------

class TestBuildExplainGNNEmbeddings:
    """Verify piece_gnn_embeddings and piece_importance in _build_explain."""

    @pytest.fixture(scope="class")
    def explain_dict(self, tutor, game_data):
        fens, _ = game_data
        _, _, _, _, internals = tutor.recommend_move(fens[0], explain=True)
        return internals

    def test_piece_gnn_embeddings_present(self, explain_dict):
        assert "piece_gnn_embeddings" in explain_dict

    def test_piece_gnn_embeddings_is_dict(self, explain_dict):
        assert isinstance(explain_dict["piece_gnn_embeddings"], dict)

    def test_piece_gnn_embeddings_nonempty(self, explain_dict):
        assert len(explain_dict["piece_gnn_embeddings"]) > 0

    def test_piece_gnn_embeddings_values_are_lists(self, explain_dict):
        for sq, emb in explain_dict["piece_gnn_embeddings"].items():
            assert isinstance(emb, list), f"{sq}: expected list, got {type(emb)}"
            assert len(emb) > 0

    def test_piece_importance_present(self, explain_dict):
        assert "piece_importance" in explain_dict

    def test_piece_importance_is_dict(self, explain_dict):
        assert isinstance(explain_dict["piece_importance"], dict)

    def test_piece_importance_in_range(self, explain_dict):
        for sq, v in explain_dict["piece_importance"].items():
            assert 0.0 <= v <= 1.0, f"{sq}: importance {v} out of [0, 1]"

    def test_piece_importance_has_max_one(self, explain_dict):
        vals = list(explain_dict["piece_importance"].values())
        if vals:
            assert max(vals) == pytest.approx(1.0, abs=1e-4)

    def test_explain_gnn_json_serialisable(self, explain_dict):
        json.dumps(explain_dict)


# ---------------------------------------------------------------------------
# piece_importance_trajectory in analyse_game
# ---------------------------------------------------------------------------

class TestPieceImportanceTrajectory:
    def test_key_present(self, stats):
        assert "piece_importance_trajectory" in stats

    def test_length_matches_n_positions(self, stats):
        assert len(stats["piece_importance_trajectory"]) == stats["n_positions"]

    def test_each_entry_is_dict(self, stats):
        for i, d in enumerate(stats["piece_importance_trajectory"]):
            assert isinstance(d, dict), f"Position {i}: expected dict"

    def test_values_in_range(self, stats):
        for i, d in enumerate(stats["piece_importance_trajectory"]):
            for sq, v in d.items():
                assert 0.0 <= v <= 1.0, f"Position {i}, {sq}: {v} out of [0,1]"

    def test_json_serialisable(self, stats):
        json.dumps(stats["piece_importance_trajectory"])
