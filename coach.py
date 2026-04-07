"""Chess coaching report with LLM narration.

Runs the same analytics pipeline as ``show_analytics.py`` but prints natural-
language coaching paragraphs after every section, turning the numbers into
actionable advice a club player can understand.

LLM narration requires a ``GROQ_API_KEY`` environment variable.  If the key
is not set the report still prints in full; only the coaching paragraphs are
omitted.

Usage
-----
    python coach.py [--model PATH] [--calib PATH] [--game N] [--device cpu]
    python coach.py [--model PATH] --lichess-game GAME_ID_OR_URL

All flags are identical to ``show_analytics.py``.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import Any

import chess
import chess.pgn
import torch

from agent.narrator import GameNarrator
from chessgnn.calibration import TemperatureScaler
from chessgnn.lichess_api import read_lichess_game
from chessgnn.theoretical import analyze_theoretical
from show_analytics import (
    _bar,
    _eval_summary,
    _extract_fens_ucis,
    _heatmap_board,
    _lead_stability_summary,
    _load_model,
    _move_tag,
    _pct_str,
    _rank_str,
    _read_nth_game,
    _sparkline,
    _BOARD_FILE_LABELS,
    _HEATMAP_CHARS,
    _SPARK,
)
from tutor import CaseTutor, _detect_tactics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "output/gateau_sequential_h128_l4.pt"
DEFAULT_CALIB = "output/gateau_sequential_h128_l4.pt.calib.json"
DEFAULT_PGN = "input/lichess_db_standard_rated_2013-01.pgn"

_WIDTH = 60  # report column width


# ---------------------------------------------------------------------------
# Coaching paragraph printer
# ---------------------------------------------------------------------------

def _print_coaching(text: str, label: str = "Coach") -> None:
    """Print an LLM coaching paragraph indented below a section."""
    if not text:
        return
    print()
    print(f"  [{label}]")
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for line in textwrap.wrap(para, width=_WIDTH - 4):
            print(f"  | {line}")
    print()


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_eval_trajectory(
    stats: dict[str, Any],
    fens: list[str],
) -> None:
    print("─" * _WIDTH)
    print("  EVALUATION TRAJECTORY  (▁ = black winning  ▇ = white winning)")
    print()
    ev = stats["eval_trajectory"]
    spark = _sparkline(ev)
    print(f"  {spark}")
    n = len(ev)
    tick_labels = "".join(
        str((i // 2) + 1).center(1) if i % 2 == 0 else " "
        for i in range(n)
    )
    print(f"  {tick_labels}")
    print()
    terminal_board = chess.Board(fens[-1]) if fens else None
    print(f"  Start eval : {ev[0]:+.3f}  (\u00b10 = equal)")
    print(f"  Last analysed eval : {ev[-1]:+.3f}  ({_eval_summary(ev[-1])})")
    if terminal_board is not None and terminal_board.is_game_over():
        if terminal_board.is_checkmate():
            winner = "black" if terminal_board.turn == chess.WHITE else "white"
            print(f"  Terminal outcome   : checkmate, {winner} won")
        elif terminal_board.is_stalemate():
            print("  Terminal outcome   : stalemate")
        elif terminal_board.is_insufficient_material():
            print("  Terminal outcome   : draw by insufficient material")
    print(f"  Sharpness  : {stats['game_sharpness']:.3f}  "
          f"({'volatile' if stats['game_sharpness'] > 0.15 else 'stable'} game)")
    print(f"  Lead stability: {stats['decisiveness']:.2f}  "
          f"({_lead_stability_summary(stats['decisiveness'])})")


def _section_opening(
    game: chess.pgn.Game,
    stats: dict[str, Any],
    fens: list[str],
    ucis: list[str],
) -> None:
    print()
    print("─" * _WIDTH)
    print("  OPENING PHASE ANALYSIS")
    print()

    _eco = game.headers.get("ECO", "")
    opening = game.headers.get("Opening", "")
    pc = stats["piece_count_trajectory"]
    ev = stats["eval_trajectory"]
    mr = stats.get("move_ranks") or []
    mp = stats.get("move_percentiles") or []
    lt = stats.get("legal_moves_trajectory") or []
    w_name = game.headers.get("White", "White")
    b_name = game.headers.get("Black", "Black")

    opening_end_idx = min(len(pc) - 1, 24)
    for i in range(1, len(pc)):
        if pc[0] - pc[i] >= 4:
            opening_end_idx = i
            break

    endgame_start_idx = len(pc)
    for i in range(opening_end_idx, len(pc)):
        if pc[i] <= 14:
            endgame_start_idx = i
            break

    opening_moves_n = (opening_end_idx + 1) // 2
    midgame_moves_n = max(0, endgame_start_idx - opening_end_idx) // 2
    endgame_moves_n = max(0, len(pc) - endgame_start_idx) // 2
    has_endgame = endgame_start_idx < len(pc)

    if _eco:
        print(f"  Classification : {_eco} · {opening}")
    elif opening:
        print(f"  Classification : {opening}")

    phase_str = f"Opening ~{opening_moves_n} moves"
    if midgame_moves_n > 0:
        phase_str += f"  →  Middlegame ~{midgame_moves_n} moves"
    if has_endgame:
        phase_str += f"  →  Endgame ~{endgame_moves_n} moves"
    print(f"  Game phases    : {phase_str}")

    if opening_end_idx < len(ev):
        exit_eval = ev[opening_end_idx]
        if abs(exit_eval) < 0.1:
            balance_desc = "balanced"
        elif exit_eval > 0.2:
            balance_desc = f"white advantage ({exit_eval:+.2f})"
        elif exit_eval < -0.2:
            balance_desc = f"black advantage ({exit_eval:+.2f})"
        else:
            balance_desc = f"slight {'white' if exit_eval > 0 else 'black'} edge ({exit_eval:+.2f})"
        print(f"  Opening exit   : {balance_desc}")

    n_op = min(opening_end_idx, len(mr))
    w_pcts = [mp[i] for i in range(0, n_op, 2) if i < len(mp) and mp[i] is not None]
    b_pcts = [mp[i] for i in range(1, n_op, 2) if i < len(mp) and mp[i] is not None]

    if w_pcts:
        w_acc = sum(w_pcts) / len(w_pcts)
        print(f"  {w_name:<12} opening quality : {_bar(w_acc, 15)}  {w_acc * 100:.0f}%")
    if b_pcts:
        b_acc = sum(b_pcts) / len(b_pcts)
        print(f"  {b_name:<12} opening quality : {_bar(b_acc, 15)}  {b_acc * 100:.0f}%")

    for side_name_nov, start_nov in ((w_name, 0), (b_name, 1)):
        for i in range(start_nov, n_op, 2):
            if i < len(mr) and mr[i] is not None and mr[i] > 3:
                mv_num = (i // 2) + 1
                try:
                    board_t = chess.Board(fens[i])
                    san_t = board_t.san(chess.Move.from_uci(ucis[i]))
                    print(f"  {side_name_nov:<12} first deviation : "
                          f"move {mv_num} ({san_t}) — "
                          f"ranked {mr[i]}/{lt[i] if i < len(lt) else '?'}")
                except Exception:
                    pass
                break


def _section_move_quality(
    game: chess.pgn.Game,
    stats: dict[str, Any],
    elo_white_result: dict,
    elo_black_result: dict,
) -> None:
    print()
    print("─" * _WIDTH)
    print("  MOVE QUALITY SUMMARY")
    print()
    w_name = game.headers.get("White", "White")
    b_name = game.headers.get("Black", "Black")
    for side, elo_res, side_label in (
        ("white", elo_white_result, f"⬜ {w_name}"),
        ("black", elo_black_result, f"⬛ {b_name}"),
    ):
        s = stats[side]
        if s is None:
            continue
        pct_bar = _bar(s["avg_move_percentile"] or 0)
        print(f"  {side_label}")
        print(f"    Accuracy  : {pct_bar}  {_pct_str(s['avg_move_percentile'])}")
        print(f"    Top-1 match: {_pct_str(s['agreement_top1'])}  "
              f"Top-3 match: {_pct_str(s['agreement_top3'])}")
        print(f"    Blunders  : {len(s['blunders'])}   "
              f"Mistakes: {len(s['mistakes'])}   "
              f"Best moves: {len(s['best_moves'])}")
        est = elo_res.get("estimated_elo")
        if est is not None:
            lo, hi = elo_res["confidence_range"]
            print(f"    Est. ELO  : {est}  [{lo}–{hi}]")
        print()


def _section_move_table(
    stats: dict[str, Any],
    fens: list[str],
    ucis: list[str],
) -> None:
    print("─" * _WIDTH)
    print("  MOVE-BY-MOVE BREAKDOWN")
    print()
    header = f"  {'#':>3}  {'Side':<6}  {'Move':<7}  {'Rank':>7}  {'Pct':>5}  {'Eval':>6}  {'Drop':>6}  Tag"
    print(header)
    print("  " + "─" * (len(header) - 2))

    board = chess.Board(fens[0])
    for i, uci in enumerate(ucis):
        if i >= len(stats["move_ranks"]):
            break
        is_white = board.turn == chess.WHITE
        side_str = "White" if is_white else "Black"
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        n_legal = stats["legal_moves_trajectory"][i]
        rank = stats["move_ranks"][i]
        pct = stats["move_percentiles"][i]
        drop = stats["eval_drops"][i]
        next_board = chess.Board(fens[i + 1]) if i + 1 < len(fens) else None
        eval_after = stats["eval_trajectory"][i + 1] if i + 1 < len(stats["eval_trajectory"]) else None
        tag = _move_tag(rank, drop, next_board)
        rank_s = _rank_str(rank, n_legal)
        pct_s = _pct_str(pct)
        if next_board is not None and next_board.is_checkmate():
            eval_s = " MATE"
        elif next_board is not None and (next_board.is_stalemate() or next_board.is_insufficient_material()):
            eval_s = " DRAW"
        else:
            eval_s = f"{eval_after:+.3f}" if eval_after is not None else "     "
        drop_s = f"{drop:+.3f}" if drop is not None else "     "
        print(f"  {i + 1:>3}  {side_str:<6}  {san:<7}  {rank_s:>7}  {pct_s:>5}  {eval_s:>6}  {drop_s:>6}  {tag}")
        board.push(move)

    print()
    print("  Rank: move rank vs all legal moves.  Pct: fraction of legal moves beaten.")
    print("  Drop: model-eval change from mover's perspective (positive = advantage lost).")


def _section_game_character(
    game: chess.pgn.Game,
    stats: dict[str, Any],
    ucis: list[str],
) -> None:
    print()
    print("─" * _WIDTH)
    print("  GAME CHARACTER & MATERIAL")
    print()
    avg_unc = stats["avg_complexity"]
    avg_br = stats["avg_branching"]
    w_terr = stats["white_territory"]
    b_terr = stats["black_territory"]
    pc = stats["piece_count_trajectory"]
    cp_traj = stats.get("center_pressure_trajectory") or []
    n_full_moves = max((len(ucis) + 1) // 2, 1)
    total_captures = pc[0] - pc[-1]
    captures_per_10 = total_captures / n_full_moves * 10

    if avg_unc > 0.97:
        char_label = "Highly Tactical"
    elif avg_unc > 0.92:
        char_label = "Sharp / Tactical"
    elif avg_unc > 0.85:
        char_label = "Dynamic"
    elif avg_unc > 0.70:
        char_label = "Balanced"
    else:
        char_label = "Positional / Strategic"

    exch_style = (
        "heavy exchanges" if captures_per_10 > 3
        else "moderate trading" if captures_per_10 > 1.5
        else "quiet, few trades"
    )

    print(f"  Character       : {char_label}  (uncertainty {avg_unc:.3f})")
    print(f"  Branching       : {avg_br:.1f} avg legal moves  "
          f"({'complex' if avg_br > 35 else 'normal' if avg_br > 25 else 'simplified'})")
    print(f"  Sharpness       : {stats['game_sharpness']:.3f}  "
          f"({'volatile' if stats['game_sharpness'] > 0.15 else 'steady'})")
    print()

    pc_spark = _sparkline([float(v) for v in pc], lo=min(pc), hi=max(pc))
    print(f"  Material arc    : {pc[0]} → {pc[-1]} pieces  "
          f"({total_captures} captures, {exch_style})")
    print(f"    {pc_spark}")

    if cp_traj:
        avg_cp = sum(cp_traj) / len(cp_traj)
        cp_spark = _sparkline(cp_traj, 0.0, max(cp_traj) or 1.0)
        cp_style = (
            "centre-dominant" if avg_cp > 0.15
            else "flank-oriented" if avg_cp < 0.05
            else "mixed"
        )
        print()
        print(f"  Centre pressure : avg {avg_cp:.3f} ({cp_style})")
        print(f"    {cp_spark}")

    print()
    print(f"  Territory (where pieces aimed to go):")
    print(f"    White half  {_bar(w_terr, 20)}  {w_terr * 100:.0f}%")
    print(f"    Black half  {_bar(b_terr, 20)}  {b_terr * 100:.0f}%")


def _section_critical_moments(
    stats: dict[str, Any],
    fens: list[str],
    ucis: list[str],
) -> list[tuple]:
    """Print critical moments and return them for the narrator."""
    print()
    print("─" * _WIDTH)
    print("  CRITICAL MOMENTS & TACTICAL PATTERNS")
    print()

    pin_traj = stats["pin_count_trajectory"]
    fork_traj = stats["fork_count_trajectory"]
    n_tac = len(pin_traj)
    tactical_positions = sum(1 for i in range(n_tac) if pin_traj[i] > 0 or fork_traj[i] > 0)
    tac_density = tactical_positions / n_tac if n_tac else 0.0

    print(f"  Tactical density : {tac_density * 100:.0f}% of positions "
          f"({tactical_positions}/{n_tac}) had active motifs")
    print(f"  Peak pins: {stats['peak_pins']}   Peak forks: {stats['peak_forks']}")
    print()

    critical_moments: list[tuple] = []
    for i in range(n_tac):
        if pin_traj[i] == 0 and fork_traj[i] == 0:
            continue
        board_t = chess.Board(fens[i])
        tactics = _detect_tactics(board_t)
        motifs: list[str] = []
        if tactics["pins"]:
            motifs.append(f"Pin: {', '.join(tactics['pins'])}")
        if tactics["forks"]:
            for fk in tactics["forks"]:
                motifs.append(f"Fork: {fk['attacker']}→{', '.join(fk['victims'])}")
        if tactics["overloaded_squares"]:
            motifs.append(f"Overloaded: {', '.join(tactics['overloaded_squares'])}")

        mv_n = (i // 2) + 1
        sd = "w" if i % 2 == 0 else "b"
        played = ""
        if i < len(ucis):
            try:
                played = board_t.san(chess.Move.from_uci(ucis[i]))
            except Exception:
                played = ucis[i]

        drop_s = ""
        ed = stats.get("eval_drops")
        rank_list = stats.get("move_ranks")
        rank_i = rank_list[i] if rank_list and i < len(rank_list) else None
        if ed and i < len(ed) and ed[i] is not None:
            d = ed[i]
            if d > 0.15:
                drop_s = f"BLUNDER {d:+.2f}"
            elif d > 0.075:
                drop_s = f"miss {d:+.2f}"
            elif rank_i == 1:
                drop_s = f"top choice {d:+.2f}"
            elif d < -0.05:
                drop_s = f"improved {d:+.2f}"
            else:
                drop_s = f"{d:+.2f}"
        critical_moments.append((mv_n, sd, "; ".join(motifs), played, drop_s))

    if critical_moments:
        print(f"  {'Move':>4}  {'Motif':<38}  {'Played':<8}  {'Result'}")
        print("  " + "─" * 60)
        for mv_n, sd, mot, played, dr in critical_moments:
            mot_d = (mot[:36] + "..") if len(mot) > 38 else mot
            print(f"  {mv_n:>3}{sd}  {mot_d:<38}  {played:<8}  {dr}")
        print()
    else:
        print("  No pins, forks, or overloaded defenders detected.")
        print()

    tension_vals = stats.get("tension_trajectory")
    if tension_vals:
        avg_tension_t = sum(tension_vals) / len(tension_vals)
        t_spark = _sparkline(tension_vals, 0.0, 1.0)
        print(f"  Tension arc (↓ = more cross-army conflict):")
        print(f"    {t_spark}  avg {avg_tension_t:.2f}")
        print()

    return critical_moments


def _section_piece_importance(
    stats: dict[str, Any],
) -> None:
    traj = stats.get("piece_importance_trajectory")
    if not traj:
        return
    print("─" * _WIDTH)
    print("  PIECE IMPORTANCE TRAJECTORY")
    print("  (Share of move-probability mass carried by each source piece)")
    print()

    n_pos = len(traj)
    all_squares: dict[str, list[float | None]] = {}
    for pos_idx, pos_dict in enumerate(traj):
        for sq, imp in pos_dict.items():
            if sq not in all_squares:
                all_squares[sq] = [None] * n_pos
            all_squares[sq][pos_idx] = imp

    def _mean_imp(vals: list[float | None]) -> float:
        present = [v for v in vals if v is not None]
        return sum(present) / len(present) if present else 0.0

    def _var_imp(vals: list[float | None]) -> float:
        present = [v for v in vals if v is not None]
        if len(present) < 2:
            return 0.0
        mean = sum(present) / len(present)
        return sum((v - mean) ** 2 for v in present) / len(present)

    def _peak_pos(vals: list[float | None]) -> int:
        return max(
            (i for i, v in enumerate(vals) if v is not None),
            key=lambda i: vals[i] or 0.0,
            default=0,
        )

    ranked = sorted(all_squares.items(), key=lambda kv: _var_imp(kv[1]), reverse=True)
    top_n = ranked[:8]

    print(f"  {'Square':<8}  {'Sparkline':<42}  {'Avg':>5}  {'Var':>5}  {'@peak':>5}")
    print("  " + "─" * 68)
    for sq, vals in top_n:
        spark_chars = []
        for pos_idx, v in enumerate(vals):
            if v is None:
                spark_chars.append(" ")
            else:
                pos_dict = traj[pos_idx]
                pos_max = max(pos_dict.values()) if pos_dict else 1.0
                pos_min = min(pos_dict.values()) if pos_dict else 0.0
                rng = pos_max - pos_min or 1.0
                rel = (v - pos_min) / rng
                idx_sp = int(round(rel * (len(_SPARK) - 1)))
                spark_chars.append(_SPARK[max(0, min(len(_SPARK) - 1, idx_sp))])
        spark_str = "".join(spark_chars)
        avg = _mean_imp(vals)
        var = _var_imp(vals)
        pmov = _peak_pos(vals) + 1
        print(f"  {sq:<8}  {spark_str:<42}  {avg:>5.2f}  {var:>5.3f}  {pmov:>5}")
    print()

    # Game-level insights
    volatile = sorted(all_squares.items(), key=lambda kv: _var_imp(kv[1]), reverse=True)
    if volatile:
        sq_v, vals_v = volatile[0]
        print(f"  Most volatile piece : {sq_v}  (var={_var_imp(vals_v):.3f})")

    top_count: dict[str, int] = {}
    for pos_dict in traj:
        if pos_dict:
            best = max(pos_dict, key=lambda s: pos_dict[s])
            top_count[best] = top_count.get(best, 0) + 1
    if top_count:
        dominant_sq = max(top_count, key=lambda s: top_count[s])
        print(f"  Most frequently #1  : {dominant_sq}  ({top_count[dominant_sq]} positions)")

    full_game_sqs = [
        sq for sq, vals in all_squares.items()
        if sum(1 for v in vals if v is not None) >= int(n_pos * 0.75)
    ]
    if full_game_sqs:
        def _above_median_frac(sq: str) -> float:
            count = above = 0
            for pos_dict in traj:
                if sq not in pos_dict:
                    continue
                count += 1
                median_val = sorted(pos_dict.values())[len(pos_dict) // 2]
                if pos_dict[sq] >= median_val:
                    above += 1
            return above / count if count else 0.0

        sq_fg = max(full_game_sqs, key=_above_median_frac)
        frac = _above_median_frac(sq_fg)
        print(f"  Most sustained      : {sq_fg}  ({frac * 100:.0f}% above median)")
    print()


def _section_step_by_step(
    game: chess.pgn.Game,
    stats: dict[str, Any],
    fens: list[str],
    ucis: list[str],
    narrator: GameNarrator,
) -> None:
    """Per-move coaching for every notable move in the game.

    A move is considered notable if it is:
    - A blunder (eval drop > 0.15)
    - A mistake (eval drop > 0.075)
    - A best move (rank 1) with tactical context
    - Inside a position where pins or forks exist
    """
    if not narrator.available:
        return

    print()
    print("─" * _WIDTH)
    print("  STEP-BY-STEP MOVE ANALYSIS")
    print()
    print("  (Coach comments on key moments — blunders, mistakes,")
    print("   tactical positions, and best moves with alternatives)")
    print()

    mr = stats.get("move_ranks") or []
    mp = stats.get("move_percentiles") or []
    ed = stats.get("eval_drops") or []
    lt = stats.get("legal_moves_trajectory") or []
    eq = stats.get("engine_q_trajectory") or []
    pin_traj = stats.get("pin_count_trajectory") or []
    fork_traj = stats.get("fork_count_trajectory") or []

    w_name = game.headers.get("White", "White")
    b_name = game.headers.get("Black", "Black")

    board = chess.Board(fens[0]) if fens else chess.Board()
    n = min(len(ucis), len(mr))

    noted = False
    for i in range(n):
        rank = mr[i]
        pct = mp[i] if i < len(mp) else None
        drop = ed[i] if i < len(ed) else None
        n_legal = lt[i] if i < len(lt) else None
        has_tactic = (i < len(pin_traj) and pin_traj[i] > 0) or \
                     (i < len(fork_traj) and fork_traj[i] > 0)

        is_blunder = drop is not None and drop > 0.15
        is_mistake = drop is not None and drop > 0.075
        is_best_with_context = rank == 1 and (drop is not None and drop > 0.02 or has_tactic)

        if not (is_blunder or is_mistake or is_best_with_context or has_tactic):
            board.push(chess.Move.from_uci(ucis[i]))
            continue

        is_white = board.turn == chess.WHITE
        side_str = "White" if is_white else "Black"
        player = w_name if is_white else b_name
        move_no = (i // 2) + 1
        fen_before = fens[i] if i < len(fens) else board.fen()

        try:
            san = board.san(chess.Move.from_uci(ucis[i]))
        except Exception:
            san = ucis[i]

        # Engine alternatives for this ply
        engine_top: list[tuple[str, float]] = []
        if i < len(eq) and eq[i]:
            for uci_alt, score in eq[i][:5]:
                try:
                    brd_alt = chess.Board(fen_before)
                    san_alt = brd_alt.san(chess.Move.from_uci(uci_alt))
                    engine_top.append((san_alt, score))
                except Exception:
                    engine_top.append((uci_alt, score))

        # Tactical context
        tactics = None
        if has_tactic:
            try:
                tactics = _detect_tactics(chess.Board(fen_before))
            except Exception:
                pass

        # Section header
        severity = ""
        if is_blunder:
            severity = " ⚠ BLUNDER"
        elif is_mistake:
            severity = " ✗ Mistake"
        elif rank == 1:
            severity = " ✓ Best move"
        elif has_tactic:
            severity = " ◆ Tactical"

        print(f"  Move {move_no} ({side_str}) — {san}{severity}")
        if rank is not None and n_legal is not None:
            pct_s = f"  {pct * 100:.0f}% of moves beaten" if pct is not None else ""
            print(f"  Rank: {rank}/{n_legal}{pct_s}")
        if drop is not None:
            print(f"  Eval drop: {drop:+.3f}")
        if engine_top:
            alt_str = "  Alternatives: " + "  |  ".join(
                f"{s} ({v:+.2f})" for s, v in engine_top[:3]
            )
            print(alt_str)

        comment = narrator.narrate_move(
            ply=i,
            side=side_str.lower(),
            player_name=player,
            san=san,
            rank=rank or 0,
            n_legal=n_legal or 1,
            percentile=pct,
            eval_drop=drop,
            engine_top=engine_top,
            fen_before=fen_before,
            tactics=tactics,
        )
        if comment:
            for line in textwrap.wrap(comment, width=_WIDTH - 4):
                print(f"  > {line}")
        print()
        noted = True
        board.push(chess.Move.from_uci(ucis[i]))

    if not noted:
        print("  No notable moves detected (no blunders, mistakes, or tactical moments).")
        print()


def _section_player_profiles(
    game: chess.pgn.Game,
    stats: dict[str, Any],
    elo_white_result: dict,
    elo_black_result: dict,
    narrator: GameNarrator,
) -> None:
    print("─" * _WIDTH)
    print("  PLAYER PROFILES & COACHING INSIGHTS")
    print()
    w_name = game.headers.get("White", "White")
    b_name = game.headers.get("Black", "Black")

    for side, name, elo_res in (
        ("white", w_name, elo_white_result),
        ("black", b_name, elo_black_result),
    ):
        s = stats.get(side)
        if s is None:
            continue
        side_sym = "⬜" if side == "white" else "⬛"
        print(f"  {side_sym} {name}")

        pct = s.get("avg_move_percentile")
        t1 = s.get("agreement_top1")
        t3 = s.get("agreement_top3")
        nb = len(s.get("blunders") or [])
        nm_err = len(s.get("mistakes") or [])
        nbest = len(s.get("best_moves") or [])
        nm = s.get("moves_played") or 0

        print(f"    Accuracy  : {_bar(pct or 0)}  {_pct_str(pct)}")
        print(f"    Top-1 match: {_pct_str(t1)}   Top-3: {_pct_str(t3)}")
        print(f"    Blunders  : {nb}   Mistakes: {nm_err}   Best moves: {nbest}")
        est = elo_res.get("estimated_elo")
        if est is not None:
            lo, hi = elo_res["confidence_range"]
            print(f"    Est. ELO  : {est}  [{lo}–{hi}]")
        print()

        narration = narrator.narrate_player_profile(game, stats, side, elo_res)
        _print_coaching(narration)


# ---------------------------------------------------------------------------
# Full coached report
# ---------------------------------------------------------------------------

def print_coached_report(
    game: chess.pgn.Game,
    stats: dict[str, Any],
    elo_white_result: dict,
    elo_black_result: dict,
    fens: list[str],
    ucis: list[str],
    narrator: GameNarrator,
    theoretical: dict | None = None,
) -> None:
    w_name = game.headers.get("White", "White")
    b_name = game.headers.get("Black", "Black")
    w_elo = game.headers.get("WhiteElo", "?")
    b_elo = game.headers.get("BlackElo", "?")
    result = game.headers.get("Result", "*")
    opening = game.headers.get("Opening", "")

    print()
    print("╔" + "═" * 58 + "╗")
    print("║{:^58}║".format("  ChessGNN · Coached Game Analysis Report  "))
    print("╚" + "═" * 58 + "╝")
    print(f"  {w_name} ({w_elo}) ⬜  vs  ⬛ {b_name} ({b_elo})")
    if opening:
        print(f"  {opening}")
    analysed_plies = stats["n_moves"]
    analysed_full = (analysed_plies + 1) // 2
    print(f"  Result: {result}   |   {analysed_plies} plies analysed (~{analysed_full} full moves)")
    if narrator.available:
        print(f"  LLM coaching: {narrator._model}")
    else:
        print("  LLM coaching: disabled (set GROQ_API_KEY to enable)")

    # Overview narration precedes the first section
    narration = narrator.narrate_overview(game, stats)
    _print_coaching(narration, label="Coach — Overview")

    # ── Evaluation trajectory ──────────────────────────────────────────────
    _section_eval_trajectory(stats, fens)

    # ── Opening phase ──────────────────────────────────────────────────────
    _section_opening(game, stats, fens, ucis)
    narration = narrator.narrate_opening(game, stats, fens, ucis)
    _print_coaching(narration)

    # ── Move quality summary ───────────────────────────────────────────────
    _section_move_quality(game, stats, elo_white_result, elo_black_result)
    narration = narrator.narrate_move_decisions(game, stats, fens, ucis)
    _print_coaching(narration)

    # ── Full move table (no narration — data speaks for itself) ───────────
    _section_move_table(stats, fens, ucis)

    # ── Step-by-step move analysis (LLM comment per notable move) ─────────
    _section_step_by_step(game, stats, fens, ucis, narrator)

    # ── Game character & material ──────────────────────────────────────────
    _section_game_character(game, stats, ucis)
    narration = narrator.narrate_game_character(game, stats, ucis)
    _print_coaching(narration)

    # ── Heatmaps ───────────────────────────────────────────────────────────
    print()
    print("─" * _WIDTH)
    print("  SPATIAL HEATMAPS  (darker = more Q-probability mass)")
    print()
    dest_lines = _heatmap_board(stats["accumulated_dest_heatmap"], "Destination")
    src_lines = _heatmap_board(stats["accumulated_src_heatmap"], "Source")
    for d, s in zip(dest_lines, src_lines):
        print(f"  {d:<30}    {s}")
    print()

    # ── Critical moments & tactics ─────────────────────────────────────────
    critical_moments = _section_critical_moments(stats, fens, ucis)
    narration = narrator.narrate_critical_moments(game, stats, critical_moments)
    _print_coaching(narration)

    # ── Piece importance ───────────────────────────────────────────────────
    _section_piece_importance(stats)
    narration = narrator.narrate_piece_activity(game, stats)
    _print_coaching(narration)

    # ── Structural fingerprint drift ───────────────────────────────────────
    drift_traj = stats.get("structural_drift_trajectory") or []
    if drift_traj:
        print("─" * _WIDTH)
        print("  STRUCTURAL FINGERPRINT DRIFT")
        print()
        drift_spark = _sparkline(drift_traj, 0.0, max(drift_traj) or 1.0)
        print(f"  Local graph drift (↑ = bigger positional shift):")
        print(f"    {drift_spark}")
        print()
        print(f"  Avg local drift  : {stats['avg_structural_drift']:.3f}")
        print(f"  Peak local drift : {stats['peak_structural_drift']:.3f}")
        print(f"  Start→end drift  : {stats['final_structural_distance']:.3f}")
        print()

    # ── Theoretical profile (if available) ────────────────────────────────
    if theoretical:
        print("─" * _WIDTH)
        print("  THEORETICAL PROFILE")
        print()
        print(f"  Avg Piece Scope: ⬜ {theoretical['w_avg_scope']:.1%}  "
              f"vs  ⬛ {theoretical['b_avg_scope']:.1%}")
        print(f"  Centre Dominance: ⬜ {theoretical['w_avg_center']:.1f}  "
              f"vs  ⬛ {theoretical['b_avg_center']:.1f}")
        ws = theoretical["w_struct"]
        bs = theoretical["b_struct"]
        print(f"  Pawn structure: ⬜ {ws['doubled']} doubled, {ws['isolated']} isolated  |  "
              f"⬛ {bs['doubled']} doubled, {bs['isolated']} isolated")
        print()

    # ── Player profiles (full coaching paragraphs per player) ─────────────
    _section_player_profiles(game, stats, elo_white_result, elo_black_result, narrator)

    print("═" * _WIDTH)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_CHECKPOINT,
                   help="Path to .pt checkpoint (default: %(default)s)")
    p.add_argument("--calib", default=DEFAULT_CALIB,
                   help="Path to calibration JSON sidecar (optional)")
    p.add_argument("--pgn", default=DEFAULT_PGN,
                   help="Path to PGN file (default: %(default)s)")
    p.add_argument("--game", type=int, default=1, metavar="N",
                   help="1-indexed game number in the PGN (default: 1)")
    p.add_argument("--lichess-game",
                   help="Lichess game id or URL to fetch via the API")
    p.add_argument("--lichess-token-env", default="LICHESS_API_TOKEN",
                   help="Env var holding an optional Lichess API token")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    print(f"\nLoading model from {args.model} …", flush=True)
    model = _load_model(args.model, device)

    tutor = CaseTutor(model, device)
    if args.calib and os.path.isfile(args.calib):
        scaler = TemperatureScaler()
        scaler.load(args.calib)
        tutor.set_calibration(scaler)
        print(f"Calibration loaded (T={scaler.T:.4f})", flush=True)

    if args.lichess_game:
        token = os.getenv(args.lichess_token_env) or None
        print(f"Fetching Lichess game {args.lichess_game} …", flush=True)
        game = read_lichess_game(args.lichess_game, token=token)
    else:
        print(f"Parsing game {args.game} from {args.pgn} …", flush=True)
        game = _read_nth_game(args.pgn, args.game)

    fens, ucis = _extract_fens_ucis(game)
    elo_w = int(game.headers.get("WhiteElo", 1500) or 1500)
    elo_b = int(game.headers.get("BlackElo", 1500) or 1500)

    print(f"Analysing {len(ucis)} moves …", flush=True)
    stats = tutor.analyse_game(fens, ucis, elo_white=elo_w, elo_black=elo_b)
    elo_white_result = tutor.estimate_elo(stats, "white")
    elo_black_result = tutor.estimate_elo(stats, "black")
    theoretical = analyze_theoretical(fens)

    narrator = GameNarrator()
    if narrator.available:
        print("LLM narration enabled (Groq)", flush=True)
    else:
        print("LLM narration disabled (no GROQ_API_KEY)", flush=True)

    print_coached_report(
        game, stats, elo_white_result, elo_black_result,
        fens, ucis, narrator, theoretical,
    )


if __name__ == "__main__":
    main()
