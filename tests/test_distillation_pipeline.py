import json
import math
import os
import tempfile

import chess
import pytest

from chessgnn.distillation_pipeline import (
    cp_to_winprob,
    evaluate_positions,
    load_jsonl,
    sample_positions_from_pgn,
    save_jsonl,
)

PGN_FILE = "input/lichess_db_standard_rated_2013-01.pgn"
STOCKFISH_PATH = "stockfish/src/stockfish"


# ------------------------------------------------------------------
# cp_to_winprob
# ------------------------------------------------------------------

class TestCpToWinprob:
    def test_zero_gives_half(self):
        assert cp_to_winprob(0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert cp_to_winprob(5000) > 0.99

    def test_large_negative(self):
        assert cp_to_winprob(-5000) < 0.01

    def test_mate_positive(self):
        assert cp_to_winprob(15_000) == 1.0

    def test_mate_negative(self):
        assert cp_to_winprob(-15_000) == 0.0

    def test_symmetry(self):
        assert cp_to_winprob(200) + cp_to_winprob(-200) == pytest.approx(1.0)

    def test_monotonic(self):
        prev = 0.0
        for cp in range(-2000, 2001, 100):
            wp = cp_to_winprob(cp)
            assert wp >= prev
            prev = wp


# ------------------------------------------------------------------
# PGN sampler
# ------------------------------------------------------------------

class TestSamplePositions:
    @pytest.mark.skipif(not os.path.exists(PGN_FILE), reason="PGN file not present")
    def test_yields_valid_fens(self):
        fens = list(sample_positions_from_pgn(PGN_FILE, max_positions=20))
        assert len(fens) == 20
        for fen in fens:
            board = chess.Board(fen)
            assert board.is_valid()

    @pytest.mark.skipif(not os.path.exists(PGN_FILE), reason="PGN file not present")
    def test_respects_min_move(self):
        fens = list(sample_positions_from_pgn(PGN_FILE, max_positions=10, min_move=15))
        for fen in fens:
            board = chess.Board(fen)
            assert board.fullmove_number >= 15

    @pytest.mark.skipif(not os.path.exists(PGN_FILE), reason="PGN file not present")
    def test_respects_max_positions(self):
        fens = list(sample_positions_from_pgn(PGN_FILE, max_positions=5))
        assert len(fens) <= 5


# ------------------------------------------------------------------
# JSONL I/O
# ------------------------------------------------------------------

class TestJsonlRoundtrip:
    def test_write_read(self, tmp_path):
        records = [
            {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
             "eval_wp": 0.52, "top_k_moves": [{"uci": "e7e5", "cp": 30, "wp": 0.52}]},
            {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "eval_wp": 0.50, "top_k_moves": []},
        ]
        path = str(tmp_path / "labels.jsonl")
        count = save_jsonl(iter(records), path)
        assert count == 2

        loaded = list(load_jsonl(path))
        assert len(loaded) == 2
        assert loaded[0]["fen"] == records[0]["fen"]
        assert loaded[1]["eval_wp"] == pytest.approx(records[1]["eval_wp"])


# ------------------------------------------------------------------
# Stockfish integration
# ------------------------------------------------------------------

class TestEvaluatePositions:
    @pytest.mark.skipif(not os.path.exists(STOCKFISH_PATH), reason="Stockfish binary not present")
    def test_evaluate_three_positions(self):
        fens = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        ]
        results = list(evaluate_positions(iter(fens), STOCKFISH_PATH, depth=8, multipv_k=3))
        assert len(results) == 3

        for r in results:
            assert "fen" in r
            assert 0.0 <= r["eval_wp"] <= 1.0
            assert isinstance(r["top_k_moves"], list)
            assert len(r["top_k_moves"]) <= 3
            for m in r["top_k_moves"]:
                assert "uci" in m
                assert "cp" in m
                assert 0.0 <= m["wp"] <= 1.0
