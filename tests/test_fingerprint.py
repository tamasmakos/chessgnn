import chess
import pytest

from chessgnn.fingerprint import (
    find_similar_fingerprints,
    fingerprint_similarity,
    position_fingerprint,
)


def test_position_fingerprint_has_stable_dimension() -> None:
    start = position_fingerprint(chess.Board())
    after_e4 = chess.Board()
    after_e4.push_san("e4")
    shifted = position_fingerprint(after_e4)

    assert len(start) == len(shifted)
    assert len(start) > 0


def test_position_fingerprint_components_are_normalised() -> None:
    board = chess.Board()
    fingerprint = position_fingerprint(board)

    for value in fingerprint:
        assert 0.0 <= value <= 1.0


def test_identical_positions_have_similarity_one() -> None:
    board = chess.Board()
    fingerprint = position_fingerprint(board)

    assert fingerprint_similarity(fingerprint, fingerprint) == pytest.approx(1.0, abs=1e-6)


def test_transposed_positions_share_same_fingerprint() -> None:
    board_a = chess.Board()
    for san in ("Nf3", "Nf6", "g3", "g6"):
        board_a.push_san(san)

    board_b = chess.Board()
    for san in ("g3", "g6", "Nf3", "Nf6"):
        board_b.push_san(san)

    assert board_a.board_fen() == board_b.board_fen()
    assert board_a.turn == board_b.turn
    assert board_a.castling_rights == board_b.castling_rights

    fp_a = position_fingerprint(board_a)
    fp_b = position_fingerprint(board_b)
    assert fp_a == fp_b


def test_similarity_distinguishes_changed_structure() -> None:
    start = position_fingerprint(chess.Board())
    board = chess.Board()
    for san in ("e4", "d5", "exd5", "Qxd5"):
        board.push_san(san)
    changed = position_fingerprint(board)

    assert fingerprint_similarity(start, changed) < 1.0


def test_find_similar_fingerprints_ranks_identical_candidate_first() -> None:
    start = position_fingerprint(chess.Board())
    board = chess.Board()
    board.push_san("e4")
    e4_fp = position_fingerprint(board)
    candidates = [e4_fp, start]

    ranked = find_similar_fingerprints(start, candidates, top_k=2)

    assert ranked[0][0] == 1
    assert ranked[0][1] == pytest.approx(1.0, abs=1e-6)