"""Pydantic-AI tool functions for the chess coach agent.

Tools extract and package analytics data from ``CoachingSession.context``
for the LLM to reason over.  They do NOT make LLM calls themselves — they
are pure data accessors registered on the Pydantic-AI agent.
"""

from __future__ import annotations

from typing import Any

import chess

try:
    from pydantic_ai import RunContext
except ImportError:  # pydantic_ai optional — tests use plain mock objects
    from typing import Any as RunContext  # type: ignore[misc]

from .schema import CoachingSession


def get_game_summary(ctx: RunContext[CoachingSession]) -> dict[str, Any]:
    """Return key game-level statistics: eval arc, sharpness, result, player info."""
    c = ctx.deps.context
    gs = c.game_stats
    ev = gs.get("eval_trajectory") or []
    return {
        "white": c.white_player,
        "black": c.black_player,
        "white_elo": c.elo_white,
        "black_elo": c.elo_black,
        "result": c.result,
        "opening": c.opening_name,
        "n_plies": gs.get("n_moves"),
        "sharpness": gs.get("game_sharpness"),
        "decisiveness": gs.get("decisiveness"),
        "avg_complexity": gs.get("avg_complexity"),
        "start_eval": ev[0] if ev else None,
        "end_eval": ev[-1] if ev else None,
        "white_territory": gs.get("white_territory"),
        "black_territory": gs.get("black_territory"),
        "peak_forks": gs.get("peak_forks"),
        "peak_pins": gs.get("peak_pins"),
    }


def get_opening_context(ctx: RunContext[CoachingSession]) -> dict[str, Any]:
    """Return opening phase data: classification, quality, first deviation per side."""
    c = ctx.deps.context
    gs = c.game_stats
    pc = gs.get("piece_count_trajectory") or []
    mr = gs.get("move_ranks") or []
    mp = gs.get("move_percentiles") or []
    lt = gs.get("legal_moves_trajectory") or []
    ev = gs.get("eval_trajectory") or []

    opening_end_idx = min(len(pc) - 1, 24) if pc else 0
    for i in range(1, len(pc)):
        if pc[0] - pc[i] >= 4:
            opening_end_idx = i
            break

    n_op = min(opening_end_idx, len(mr))
    w_pcts = [mp[i] for i in range(0, n_op, 2) if i < len(mp) and mp[i] is not None]
    b_pcts = [mp[i] for i in range(1, n_op, 2) if i < len(mp) and mp[i] is not None]

    opening_sans: list[str] = []
    board = chess.Board()
    for u in c.moves_uci[:16]:
        try:
            move = chess.Move.from_uci(u)
            opening_sans.append(board.san(move))
            board.push(move)
        except Exception:
            break

    return {
        "eco": c.eco,
        "opening_name": c.opening_name,
        "opening_moves_san": opening_sans,
        "opening_end_ply": opening_end_idx,
        "exit_eval": ev[opening_end_idx] if opening_end_idx < len(ev) else None,
        "white_opening_quality": sum(w_pcts) / len(w_pcts) if w_pcts else None,
        "black_opening_quality": sum(b_pcts) / len(b_pcts) if b_pcts else None,
    }


def explain_critical_moves(
    ctx: RunContext[CoachingSession],
    max_moments: int = 5,
) -> list[dict[str, Any]]:
    """Return the top critical moments (blunders + mistakes) with full context.

    Each entry includes: ply, SAN move played, eval drop, rank, severity,
    and the top 3 engine alternatives from the Q distribution.
    """
    c = ctx.deps.context
    gs = c.game_stats
    ws = gs.get("white") or {}
    bs = gs.get("black") or {}

    blunders = (ws.get("blunders") or []) + (bs.get("blunders") or [])
    mistakes = (ws.get("mistakes") or []) + (bs.get("mistakes") or [])

    # Sort by severity: largest eval_drop first
    all_errors = sorted(
        [(m, "blunder") for m in blunders] + [(m, "mistake") for m in mistakes],
        key=lambda x: -(x[0].get("eval_drop") or 0),
    )[:max_moments]

    results: list[dict[str, Any]] = []
    for m, severity in all_errors:
        ply = m["move_no"] - 1  # 0-indexed
        san = "?"
        top_alt_san: list[str] = []

        if ply < len(c.fens) and ply < len(c.moves_uci):
            try:
                board = chess.Board(c.fens[ply])
                san = board.san(chess.Move.from_uci(c.moves_uci[ply]))
            except Exception:
                pass

            eq = gs.get("engine_q_trajectory") or []
            if ply < len(eq):
                for alt_uci, _prob in eq[ply][:4]:
                    if alt_uci != c.moves_uci[ply]:
                        try:
                            brd = chess.Board(c.fens[ply])
                            top_alt_san.append(brd.san(chess.Move.from_uci(alt_uci)))
                        except Exception:
                            top_alt_san.append(alt_uci)
                    if len(top_alt_san) >= 3:
                        break

        results.append({
            "ply": m["move_no"],
            "move_number": (m["move_no"] + 1) // 2,
            "side": "white" if m["move_no"] % 2 == 1 else "black",
            "played_san": san,
            "eval_drop": m.get("eval_drop"),
            "rank": m.get("rank"),
            "severity": severity,
            "top_alternatives_san": top_alt_san,
        })
    return results


def get_piece_activity(ctx: RunContext[CoachingSession]) -> dict[str, Any]:
    """Summarise per-piece policy mass activity across the whole game."""
    c = ctx.deps.context
    traj = c.game_stats.get("piece_importance_trajectory") or []
    if not traj:
        return {"available": False}

    n_pos = len(traj)
    all_squares: dict[str, list[float | None]] = {}
    for pos_idx, pos_dict in enumerate(traj):
        for sq, imp in pos_dict.items():
            if sq not in all_squares:
                all_squares[sq] = [None] * n_pos
            all_squares[sq][pos_idx] = imp

    def _mean(vals: list[float | None]) -> float:
        present = [v for v in vals if v is not None]
        return sum(present) / len(present) if present else 0.0

    def _var(vals: list[float | None]) -> float:
        present = [v for v in vals if v is not None]
        if len(present) < 2:
            return 0.0
        mean = sum(present) / len(present)
        return sum((v - mean) ** 2 for v in present) / len(present)

    top_by_mean = sorted(
        [(sq, round(_mean(v), 4)) for sq, v in all_squares.items()],
        key=lambda x: -x[1],
    )[:8]
    most_volatile = max(all_squares.items(), key=lambda kv: _var(kv[1]))

    top_count: dict[str, int] = {}
    for pos_dict in traj:
        if pos_dict:
            best = max(pos_dict, key=lambda s: pos_dict[s])
            top_count[best] = top_count.get(best, 0) + 1
    dominant_sq = max(top_count, key=lambda s: top_count[s]) if top_count else None

    return {
        "available": True,
        "n_positions": n_pos,
        "top_pieces_by_mean_mass": top_by_mean,
        "most_volatile_piece": most_volatile[0],
        "most_volatile_variance": round(_var(most_volatile[1]), 4),
        "most_dominant_piece": dominant_sq,
        "dominant_piece_top1_count": top_count.get(dominant_sq, 0) if dominant_sq else 0,
    }


def get_player_profile(
    ctx: RunContext[CoachingSession],
    side: str = "white",
) -> dict[str, Any]:
    """Return a full player profile dict for the given side (white or black)."""
    c = ctx.deps.context
    gs = c.game_stats
    s = gs.get(side) or {}
    nm = s.get("moves_played") or 0
    elo_est = c.elo_estimate_white if side == "white" else c.elo_estimate_black

    return {
        "name": c.white_player if side == "white" else c.black_player,
        "side": side,
        "actual_elo": c.elo_white if side == "white" else c.elo_black,
        "estimated_elo": elo_est.get("estimated_elo") if elo_est else None,
        "confidence_range": elo_est.get("confidence_range") if elo_est else None,
        "moves_played": nm,
        "avg_move_percentile": s.get("avg_move_percentile"),
        "agreement_top1": s.get("agreement_top1"),
        "agreement_top3": s.get("agreement_top3"),
        "blunders": s.get("blunders") or [],
        "mistakes": s.get("mistakes") or [],
        "best_moves_count": len(s.get("best_moves") or []),
        "blunder_rate": round(len(s.get("blunders") or []) / max(nm, 1), 4),
        "mistake_rate": round(len(s.get("mistakes") or []) / max(nm, 1), 4),
    }


def get_move_detail(
    ctx: RunContext[CoachingSession],
    ply: int,
) -> dict[str, Any]:
    """Return a deep analysis of the move at the given 1-indexed ply.

    Includes: FEN, played move SAN, engine top-5 alternatives, eval
    before/after, rank in the legal-move list, and eval drop.
    """
    c = ctx.deps.context
    gs = c.game_stats
    idx = ply - 1

    if idx < 0 or idx >= len(c.moves_uci):
        return {"error": f"Ply {ply} is out of range (1\u2013{len(c.moves_uci)})"}

    fen = c.fens[idx] if idx < len(c.fens) else None
    uci = c.moves_uci[idx]
    san = "?"
    alt_sans: list[dict] = []

    if fen:
        try:
            board = chess.Board(fen)
            san = board.san(chess.Move.from_uci(uci))
        except Exception:
            pass

        eq = gs.get("engine_q_trajectory") or []
        if idx < len(eq):
            for alt_uci, prob in eq[idx][:5]:
                try:
                    brd = chess.Board(fen)
                    alt_san = brd.san(chess.Move.from_uci(alt_uci))
                    alt_sans.append({"san": alt_san, "probability_pct": round(prob * 100, 1)})
                except Exception:
                    alt_sans.append({"san": alt_uci, "probability_pct": round(prob * 100, 1)})

    ev = gs.get("eval_trajectory") or []
    drops = gs.get("eval_drops") or []
    ranks = gs.get("move_ranks") or []
    pcts = gs.get("move_percentiles") or []

    return {
        "ply": ply,
        "move_number": (ply + 1) // 2,
        "side": "white" if ply % 2 == 1 else "black",
        "fen": fen,
        "played_san": san,
        "played_uci": uci,
        "rank": ranks[idx] if idx < len(ranks) else None,
        "percentile": pcts[idx] if idx < len(pcts) else None,
        "eval_before": ev[idx] if idx < len(ev) else None,
        "eval_after": ev[idx + 1] if idx + 1 < len(ev) else None,
        "eval_drop": drops[idx] if idx < len(drops) else None,
        "engine_top5": alt_sans,
    }
