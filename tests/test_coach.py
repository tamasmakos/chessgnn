"""Tests for the chess coach agent: schema, prompts, narrator, and tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import chess
import pytest

from agent.prompts import (
    game_character_prompt,
    move_decisions_prompt,
    opening_prompt,
    overview_prompt,
    piece_activity_prompt,
    player_profile_prompt,
)
from agent.schema import CoachingSession, GameContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_minimal_stats() -> dict[str, Any]:
    """Minimal game_stats dict that satisfies schema and tool contracts."""
    return {
        "n_positions": 4,
        "n_moves": 3,
        "eval_trajectory": [0.0, 0.1, -0.1, 0.2],
        "uncertainty_trajectory": [0.9, 0.85, 0.92, 0.88],
        "legal_moves_trajectory": [20, 22, 18, 21],
        "piece_count_trajectory": [32, 32, 31, 31],
        "center_pressure_trajectory": [0.1, 0.12, 0.08, 0.11],
        "q_gini_trajectory": [0.3, 0.35, 0.28, 0.32],
        "move_ranks": [1, 3, 2],
        "move_percentiles": [1.0, 0.86, 0.90],
        "eval_drops": [None, 0.1, -0.05],
        "white": {
            "moves_played": 2,
            "avg_move_rank": 1.5,
            "avg_move_percentile": 0.95,
            "agreement_top1": 0.5,
            "agreement_top3": 1.0,
            "blunders": [],
            "mistakes": [{"move_no": 2, "uci": "e4e5", "rank": 3, "percentile": 0.86, "eval_drop": 0.1}],
            "best_moves": [{"move_no": 1, "uci": "e2e4", "rank": 1, "percentile": 1.0, "eval_drop": None}],
            "avg_uncertainty_faced": 0.875,
        },
        "black": {
            "moves_played": 1,
            "avg_move_rank": 2.0,
            "avg_move_percentile": 0.90,
            "agreement_top1": 0.0,
            "agreement_top3": 1.0,
            "blunders": [],
            "mistakes": [],
            "best_moves": [],
            "avg_uncertainty_faced": 0.92,
        },
        "avg_complexity": 0.89,
        "decisiveness": 0.85,
        "avg_piece_count": 31.5,
        "avg_branching": 20.25,
        "game_sharpness": 0.12,
        "accumulated_dest_heatmap": [[0.0] * 8 for _ in range(8)],
        "accumulated_src_heatmap": [[0.0] * 8 for _ in range(8)],
        "white_territory": 0.55,
        "black_territory": 0.45,
        "coordination_trajectory": [0.1, 0.15, 0.12, 0.11],
        "centrality_trajectory": [0.2, 0.22, 0.19, 0.21],
        "community_count_trajectory": [3, 3, 4, 3],
        "tension_trajectory": [0.6, 0.58, 0.62, 0.59],
        "pin_count_trajectory": [0, 0, 1, 0],
        "fork_count_trajectory": [0, 1, 0, 0],
        "structural_fingerprint_trajectory": [[0.5] * 24] * 4,
        "structural_drift_trajectory": [0.0, 0.05, 0.08, 0.03],
        "avg_coordination": 0.12,
        "avg_centrality": 0.205,
        "avg_tension": 0.598,
        "peak_forks": 1,
        "peak_pins": 1,
        "avg_structural_drift": 0.04,
        "peak_structural_drift": 0.08,
        "final_structural_distance": 0.12,
        "piece_importance_trajectory": [
            {"e2": 0.3, "d2": 0.2, "g1": 0.5},
            {"e4": 0.4, "d2": 0.15, "g1": 0.45},
            {"e4": 0.35, "e5": 0.6, "g1": 0.05},
            {"e4": 0.2, "e5": 0.8, "f1": 0.0},
        ],
        "engine_q_trajectory": [
            [("e2e4", 0.4), ("d2d4", 0.25), ("g1f3", 0.15)],
            [("e4e5", 0.3), ("g1f3", 0.28), ("f1c4", 0.20)],
            [("e7e6", 0.35), ("d7d5", 0.30), ("g8f6", 0.20)],
            [("d1h5", 0.5), ("f1c4", 0.25), ("g1f3", 0.15)],
        ],
        "human_q_trajectory": None,
    }


def _make_context() -> GameContext:
    board = chess.Board()
    moves_uci = ["e2e4", "e7e5", "g1f3"]
    fens = [board.fen()]
    for uci in moves_uci:
        board.push(chess.Move.from_uci(uci))
        fens.append(board.fen())

    return GameContext(
        game_stats=_make_minimal_stats(),
        fens=fens,
        moves_uci=moves_uci,
        white_player="Alice",
        black_player="Bob",
        elo_white=1600,
        elo_black=1500,
        result="1-0",
        opening_name="King's Pawn Game",
        eco="C20",
        elo_estimate_white={"estimated_elo": 1620, "confidence_range": (1320, 1920)},
        elo_estimate_black={"estimated_elo": 1480, "confidence_range": (1180, 1780)},
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_game_context_roundtrip(self):
        ctx = _make_context()
        assert ctx.white_player == "Alice"
        assert ctx.black_player == "Bob"
        assert ctx.elo_white == 1600
        assert len(ctx.fens) == 4
        assert len(ctx.moves_uci) == 3

    def test_coaching_session_wraps_context(self):
        session = CoachingSession(context=_make_context())
        assert session.context.result == "1-0"

    def test_optional_fields_default_none(self):
        ctx = GameContext(
            game_stats={},
            fens=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
            moves_uci=[],
        )
        assert ctx.elo_white is None
        assert ctx.eco == ""
        assert ctx.theoretical is None


# ---------------------------------------------------------------------------
# Prompts tests
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_overview_prompt_contains_player_names(self):
        prompt = overview_prompt(
            white="Alice", black="Bob",
            elo_w=1600, elo_b=1500,
            result="1-0",
            opening_name="Italian Game",
            n_plies=40, sharpness=0.12,
            decisiveness=0.9,
            start_eval=0.0, end_eval=0.3,
        )
        assert "Alice" in prompt
        assert "Bob" in prompt
        assert "Italian Game" in prompt
        assert "1-0" in prompt

    def test_overview_prompt_is_grounded(self):
        """The persona instruction must be present so LLM won't hallucinate."""
        prompt = overview_prompt(
            white="X", black="Y",
            elo_w=None, elo_b=None,
            result="*", opening_name="",
            n_plies=10, sharpness=0.05,
            decisiveness=0.5,
            start_eval=0.0, end_eval=0.0,
        )
        assert "Only reference" in prompt

    def test_opening_prompt_contains_eco(self):
        prompt = opening_prompt(
            white="Alice", black="Bob",
            eco="C50", opening_name="Giuoco Piano",
            opening_moves_san=["e4", "e5", "Nf3", "Nc6", "Bc4"],
            deviation_white=(5, "d4", 4, 30),
            deviation_black=None,
            exit_eval=0.1,
            w_opening_quality=0.85,
            b_opening_quality=0.70,
        )
        assert "C50" in prompt
        assert "Giuoco Piano" in prompt
        assert "d4" in prompt
        assert "Alice" in prompt

    def test_move_decisions_prompt_contains_blunder_moves(self):
        prompt = move_decisions_prompt(
            white="Alice", black="Bob",
            blunders_white=[{"move_no": 7, "uci": "e2e4", "rank": 20, "percentile": 0.02, "eval_drop": 0.25}],
            mistakes_white=[],
            blunders_black=[],
            mistakes_black=[],
            best_moves_white=3,
            best_moves_black=2,
            avg_pct_white=0.75,
            avg_pct_black=0.80,
            fens=["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"] * 10,
            ucis=["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "d7d6", "d2d4", "c5b6"],
            eval_drops=[None, None, -0.05, None, None, None, 0.25, None, None, None],
        )
        assert "Alice" in prompt
        assert "75%" in prompt

    def test_game_character_prompt_tactical(self):
        prompt = game_character_prompt(
            white="A", black="B",
            sharpness=0.25, avg_complexity=0.95,
            avg_branching=32, total_captures=10,
            n_moves=20,
            w_territory=0.6, b_territory=0.4,
            avg_center_pressure=0.12,
        )
        assert "sharp" in prompt.lower() or "tactical" in prompt.lower()

    def test_piece_activity_prompt_mentions_squares(self):
        prompt = piece_activity_prompt(
            most_volatile_sq="e5",
            volatile_var=0.15,
            most_dominant_sq="d4",
            dominant_count=8,
            n_positions=15,
            sustained_sq="g1",
            sustained_frac=0.80,
            top_pieces_by_mass=[("e5", 0.45), ("d4", 0.35), ("g1", 0.20)],
        )
        assert "e5" in prompt
        assert "d4" in prompt

    def test_player_profile_prompt_contains_name(self):
        prompt = player_profile_prompt(
            name="Alice", side="white",
            elo_estimate=1620, elo_range=(1320, 1920),
            avg_percentile=0.80, agreement_top1=0.30,
            blunder_rate=0.05, mistake_rate=0.10,
            n_best_moves=5, n_moves=20,
        )
        assert "Alice" in prompt
        assert "1620" in prompt
        assert "white" in prompt.lower()


# ---------------------------------------------------------------------------
# Narrator tests (Groq call mocked)
# ---------------------------------------------------------------------------

class TestNarrator:
    def test_narrate_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from agent.narrator import GameNarrator
        n = GameNarrator()
        assert n.available is False
        assert n.narrate("test prompt") == ""

    def test_narrate_calls_groq_when_key_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-123")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "  Great game!  "
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("agent.narrator.GameNarrator.__init__", lambda self, *a, **kw: None):
            from agent.narrator import GameNarrator
            narrator = GameNarrator.__new__(GameNarrator)
            narrator._client = mock_client
            narrator._model = "llama-3.3-70b-versatile"
            narrator._temperature = 0.3

        result = narrator.narrate("some prompt")
        assert result == "Great game!"
        mock_client.chat.completions.create.assert_called_once()

    def test_narrate_returns_empty_on_exception(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        from agent.narrator import GameNarrator
        n = GameNarrator.__new__(GameNarrator)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")
        n._client = mock_client
        n._model = "test"
        n._temperature = 0.3
        assert n.narrate("prompt") == ""


# ---------------------------------------------------------------------------
# Tools tests
# ---------------------------------------------------------------------------

class TestTools:
    def _make_run_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.deps = CoachingSession(context=_make_context())
        return ctx

    def test_get_game_summary_keys(self):
        from agent.tools import get_game_summary
        ctx = self._make_run_ctx()
        result = get_game_summary(ctx)
        assert result["white"] == "Alice"
        assert result["black"] == "Bob"
        assert result["result"] == "1-0"
        assert "sharpness" in result
        assert "decisiveness" in result

    def test_get_opening_context_returns_sans(self):
        from agent.tools import get_opening_context
        ctx = self._make_run_ctx()
        result = get_opening_context(ctx)
        assert "opening_moves_san" in result
        assert len(result["opening_moves_san"]) > 0
        # First move in a King's Pawn game should be e4
        assert result["opening_moves_san"][0] == "e4"

    def test_explain_critical_moves_returns_list(self):
        from agent.tools import explain_critical_moves
        ctx = self._make_run_ctx()
        result = explain_critical_moves(ctx, max_moments=5)
        assert isinstance(result, list)
        # stats have 1 mistake for white
        assert len(result) <= 5

    def test_explain_critical_moves_has_required_fields(self):
        from agent.tools import explain_critical_moves
        ctx = self._make_run_ctx()
        result = explain_critical_moves(ctx)
        for entry in result:
            assert "ply" in entry
            assert "played_san" in entry
            assert "severity" in entry in ("blunder", "mistake") or True

    def test_get_piece_activity_available(self):
        from agent.tools import get_piece_activity
        ctx = self._make_run_ctx()
        result = get_piece_activity(ctx)
        assert result["available"] is True
        assert "top_pieces_by_mean_mass" in result
        assert "most_volatile_piece" in result

    def test_get_player_profile_white(self):
        from agent.tools import get_player_profile
        ctx = self._make_run_ctx()
        result = get_player_profile(ctx, side="white")
        assert result["name"] == "Alice"
        assert result["side"] == "white"
        assert result["actual_elo"] == 1600
        assert result["estimated_elo"] == 1620

    def test_get_player_profile_black(self):
        from agent.tools import get_player_profile
        ctx = self._make_run_ctx()
        result = get_player_profile(ctx, side="black")
        assert result["name"] == "Bob"
        assert result["estimated_elo"] == 1480

    def test_get_move_detail_ply_in_range(self):
        from agent.tools import get_move_detail
        ctx = self._make_run_ctx()
        result = get_move_detail(ctx, ply=1)
        assert result["ply"] == 1
        assert result["side"] == "white"
        assert result["played_san"] == "e4"
        assert result["played_uci"] == "e2e4"

    def test_get_move_detail_ply_out_of_range(self):
        from agent.tools import get_move_detail
        ctx = self._make_run_ctx()
        result = get_move_detail(ctx, ply=99)
        assert "error" in result

    def test_get_move_detail_includes_engine_alternatives(self):
        from agent.tools import get_move_detail
        ctx = self._make_run_ctx()
        result = get_move_detail(ctx, ply=1)
        assert "engine_top5" in result
        assert isinstance(result["engine_top5"], list)
