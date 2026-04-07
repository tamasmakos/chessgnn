import chess

from show_analytics import _eval_summary, _lead_stability_summary, _move_tag


def test_eval_summary_balanced_near_zero() -> None:
    assert _eval_summary(-0.007) == "balanced"


def test_eval_summary_slight_edge() -> None:
    assert _eval_summary(0.12) == "slight white edge"


def test_lead_stability_summary_high() -> None:
    assert _lead_stability_summary(0.95) == "one side kept the edge"


def test_move_tag_uses_rank_for_best_not_negative_drop() -> None:
    assert _move_tag(rank=1, drop=-0.20) == "Best ✓"
    assert _move_tag(rank=26, drop=-0.20) == ""


def test_move_tag_prefers_terminal_mate() -> None:
    board = chess.Board("r1bkQbnr/pppp1B1p/1qn1p3/8/3P4/P1N5/1PP2PPP/R3KBNR b KQ - 1 13")
    assert board.is_checkmate()
    assert _move_tag(rank=42, drop=None, terminal_board=board) == "Mate #"