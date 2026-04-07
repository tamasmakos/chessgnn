"""Learner-facing game analytics report using CaseTutor.analyse_game().

Usage
-----
    python show_analytics.py [--model PATH] [--game N]
    python show_analytics.py [--model PATH] --lichess-game GAME_ID_OR_URL

Loads the best available checkpoint, parses game N (1-indexed) from the
lichess PGN (or fetches a game directly from Lichess), runs full post-game
analysis and prints a structured coaching report to stdout.

Default checkpoint: output/gateau_sequential_h128_l4.pt
Default game:       1  (BFG9k vs mamalak, French Defence, 12 moves)
"""

import argparse
import io
import os
import sys

import chess
import chess.pgn
import torch

from chessgnn.calibration import TemperatureScaler
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.lichess_api import read_lichess_game
from chessgnn.model import GATEAUChessModel
from chessgnn.theoretical import analyze_theoretical
from tutor import CaseTutor, _detect_tactics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "output/gateau_sequential_h128_l4.pt"
DEFAULT_CALIB      = "output/gateau_sequential_h128_l4.pt.calib.json"
DEFAULT_PGN        = "input/lichess_db_standard_rated_2013-01.pgn"

_SPARK = " ▁▂▃▄▅▆▇█"   # 9 levels (0→8)
_BOARD_FILE_LABELS = "  a b c d e f g h"
_HEATMAP_CHARS = " ·░▒▓█"   # 6 levels (0→5)


# ---------------------------------------------------------------------------
# Model loading (same pattern as benchmark.py / calibrate.py)
# ---------------------------------------------------------------------------

def _load_model(path: str, device: torch.device) -> GATEAUChessModel:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "global_gru.weight_ih_l0" in ckpt:
        hidden_channels = ckpt["global_gru.weight_ih_l0"].shape[0] // 3
    else:
        hidden_channels = next(
            v.shape[0] for k, v in ckpt.items() if k.startswith("convs.0.k_lin.")
        )
    num_layers = max(int(k.split(".")[1]) for k in ckpt if k.startswith("convs.")) + 1
    builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
    model = GATEAUChessModel(
        builder.get_metadata(),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        temporal_mode="global_gru",
    )
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# PGN helpers
# ---------------------------------------------------------------------------

def _read_nth_game(pgn_path: str, n: int) -> chess.pgn.Game:
    """Return the n-th game (1-indexed) from a PGN file."""
    with open(pgn_path) as fh:
        for i in range(n):
            game = chess.pgn.read_game(fh)
            if game is None:
                raise ValueError(f"PGN has fewer than {n} games")
    return game


def _extract_fens_ucis(game: chess.pgn.Game) -> tuple[list[str], list[str]]:
    board = game.board()
    fens = [board.fen()]
    ucis: list[str] = []
    for move in game.mainline_moves():
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    return fens, ucis


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sparkline(values: list[float], lo: float = -1.0, hi: float = 1.0,
               width: int = 40) -> str:
    """Produce a Unicode sparkline string from a list of floats."""
    n_chars = len(_SPARK) - 1  # 8 levels
    result = []
    for v in values:
        normalised = (v - lo) / (hi - lo)
        idx = int(round(normalised * n_chars))
        idx = max(0, min(n_chars, idx))
        result.append(_SPARK[idx])
    return "".join(result)


def _heatmap_board(heatmap: list[list[float]], label: str) -> list[str]:
    """Render an 8×8 heatmap as an ASCII board. heatmap[rank][file], rank 0 = rank 1."""
    n_chars = len(_HEATMAP_CHARS) - 1  # 5 levels
    lines = [f"  {label}"]
    lines.append(_BOARD_FILE_LABELS)
    for rank in range(7, -1, -1):   # rank 8 at top
        row_chars = ""
        for file in range(8):
            v = heatmap[rank][file]
            idx = int(round(v * n_chars))
            idx = max(0, min(n_chars, idx))
            row_chars += _HEATMAP_CHARS[idx] + " "
        lines.append(f"{rank + 1} {row_chars.rstrip()}")
    return lines


def _bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    filled = int(round(value * width))
    return fill * filled + empty * (width - filled)


def _tag(drop: float | None) -> str:
    if drop is None:
        return ""
    if drop > 0.15:
        return "BLUNDER ❗"
    if drop > 0.075:
        return "Mistake ⚠"
    if drop < -0.075:
        return "Best ✓"
    return ""


def _pct_str(v: float | None) -> str:
    return f"{v * 100:.0f}%" if v is not None else "  — "


def _rank_str(r: int | None, total: int) -> str:
    return f"{r}/{total}" if r is not None else "—"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    game: chess.pgn.Game,
    stats: dict,
    elo_white_result: dict,
    elo_black_result: dict,
    fens: list[str],
    ucis: list[str],
    theoretical: dict | None = None,
) -> None:
    w_name  = game.headers.get("White", "White")
    b_name  = game.headers.get("Black", "Black")
    w_elo   = game.headers.get("WhiteElo", "?")
    b_elo   = game.headers.get("BlackElo", "?")
    result  = game.headers.get("Result", "*")
    opening = game.headers.get("Opening", "")

    print()
    print("╔" + "═" * 58 + "╗")
    print("║{:^58}║".format("  ChessGNN · Learner Game Analytics Report  "))
    print("╚" + "═" * 58 + "╝")
    print(f"  {w_name} ({w_elo}) ⬜  vs  ⬛ {b_name} ({b_elo})")
    if opening:
        print(f"  {opening}")
    print(f"  Result: {result}   |   {stats['n_moves']} moves analysed")
    print()

    # -----------------------------------------------------------------------
    # Eval trajectory sparkline
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  EVALUATION TRAJECTORY  (▁ = black winning  ▇ = white winning)")
    print()
    spark = _sparkline(stats["eval_trajectory"])
    print(f"  {spark}")
    # label move numbers underneath
    n = len(stats["eval_trajectory"])
    tick_labels = "".join(
        str((i // 2) + 1).center(1) if i % 2 == 0 else " "
        for i in range(n)
    )
    print(f"  {tick_labels}")
    print()
    # Quick stats
    ev = stats["eval_trajectory"]
    print(f"  Start eval : {ev[0]:+.3f}  (±0 = equal)")
    print(f"  Final eval : {ev[-1]:+.3f}  ({'white' if ev[-1] > 0 else 'black'} advantage)")
    print(f"  Sharpness  : {stats['game_sharpness']:.3f}  "
          f"({'volatile' if stats['game_sharpness'] > 0.15 else 'stable'} game)")
    print(f"  Decisiveness: {stats['decisiveness']:.2f}  "
          f"({'consistent direction' if stats['decisiveness'] > 0.7 else 'direction reversed frequently'})")
    print()

    # -----------------------------------------------------------------------
    # Opening phase analysis
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  OPENING PHASE ANALYSIS")
    print()

    _eco = game.headers.get("ECO", "")
    pc = stats["piece_count_trajectory"]

    # Phase detection: opening ends after 4+ captures or move 12
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

    mr = stats.get("move_ranks")
    mp = stats.get("move_percentiles")
    if mr and mp:
        n_op = min(opening_end_idx, len(mr))
        w_pcts = [mp[i] for i in range(0, n_op, 2) if mp[i] is not None]
        b_pcts = [mp[i] for i in range(1, n_op, 2) if mp[i] is not None]

        if w_pcts:
            w_acc = sum(w_pcts) / len(w_pcts)
            print(f"  {w_name:<12} opening quality : {_bar(w_acc, 15)}  {w_acc*100:.0f}%")
        if b_pcts:
            b_acc = sum(b_pcts) / len(b_pcts)
            print(f"  {b_name:<12} opening quality : {_bar(b_acc, 15)}  {b_acc*100:.0f}%")

        for side_name_nov, start_nov in ((w_name, 0), (b_name, 1)):
            for i in range(start_nov, n_op, 2):
                if mr[i] is not None and mr[i] > 3:
                    mv_num = (i // 2) + 1
                    board_t = chess.Board(fens[i])
                    san_t = board_t.san(chess.Move.from_uci(ucis[i]))
                    print(f"  {side_name_nov:<12} first novelty   : "
                          f"move {mv_num} ({san_t}) — "
                          f"ranked {mr[i]}/{stats['legal_moves_trajectory'][i]}")
                    break
    print()

    # -----------------------------------------------------------------------
    # Per-side summary
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  MOVE QUALITY SUMMARY")
    print()
    for side, elo_res in (("white", elo_white_result), ("black", elo_black_result)):
        s = stats[side]
        if s is None:
            continue
        side_label = f"⬜ {w_name}" if side == "white" else f"⬛ {b_name}"
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

    # -----------------------------------------------------------------------
    # Move-by-move table
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  MOVE-BY-MOVE BREAKDOWN")
    print()
    header = f"  {'#':>3}  {'Side':<6}  {'UCI':<7}  {'Rank':>7}  {'Pct':>5}  {'Eval':>6}  {'Drop':>6}  Tag"
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
        pct  = stats["move_percentiles"][i]
        drop = stats["eval_drops"][i]
        eval_after = stats["eval_trajectory"][i + 1] if i + 1 < len(stats["eval_trajectory"]) else None
        tag  = _tag(drop)
        rank_s = _rank_str(rank, n_legal)
        pct_s  = _pct_str(pct)
        eval_s = f"{eval_after:+.3f}" if eval_after is not None else "     "
        drop_s = f"{drop:+.3f}" if drop is not None else "     "
        print(f"  {i+1:>3}  {side_str:<6}  {san:<7}  {rank_s:>7}  {pct_s:>5}  {eval_s:>6}  {drop_s:>6}  {tag}")
        board.push(move)

    print()
    print("  Rank: move rank vs all legal moves. Pct: fraction of moves beaten (higher = better).")
    print("  Drop: win-prob change from the mover's perspective (positive = advantage lost).")

    # -----------------------------------------------------------------------
    # Theoretical Profile
    # -----------------------------------------------------------------------
    if theoretical:
        print()
        print("─" * 60)
        print("  THEORETICAL PROFILE (Non-Engine Heuristics)")
        print()
        w_scope = theoretical["w_avg_scope"]
        b_scope = theoretical["b_avg_scope"]
        print(f"  Avg Piece Scope Efficiency: ⬜ {w_scope:.1%}  vs  ⬛ {b_scope:.1%}")
        
        w_cen = theoretical["w_avg_center"]
        b_cen = theoretical["b_avg_center"]
        print(f"  Avg Central Dominance Pts : ⬜ {w_cen:.1f}  vs  ⬛ {b_cen:.1f}")
        
        print(f"  Max Outposts Established  : ⬜ {theoretical['w_outposts_max']}  vs  ⬛ {theoretical['b_outposts_max']}")
        
        ws = theoretical["w_struct"]
        bs = theoretical["b_struct"]
        print(f"  Final Pawn Structure      : ⬜ {ws['doubled']} doubled, {ws['isolated']} isolated")
        print(f"                              ⬛ {bs['doubled']} doubled, {bs['isolated']} isolated")

    # -----------------------------------------------------------------------
    # Game character & material
    # -----------------------------------------------------------------------
    print()
    print("─" * 60)
    print("  GAME CHARACTER & MATERIAL")
    print()
    avg_unc  = stats["avg_complexity"]
    avg_br   = stats["avg_branching"]
    w_terr   = stats["white_territory"]
    b_terr   = stats["black_territory"]
    pc = stats["piece_count_trajectory"]

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

    total_captures = pc[0] - pc[-1]
    n_full_moves = max((len(ucis) + 1) // 2, 1)
    captures_per_10 = total_captures / n_full_moves * 10

    print(f"  Character       : {char_label}  (uncertainty {avg_unc:.3f})")
    print(f"  Branching       : {avg_br:.1f} avg legal moves  "
          f"({'complex' if avg_br > 35 else 'normal' if avg_br > 25 else 'simplified'})")
    print(f"  Sharpness       : {stats['game_sharpness']:.3f}  "
          f"({'volatile' if stats['game_sharpness'] > 0.15 else 'steady'})")
    print()

    # Material arc
    if captures_per_10 > 3:
        exch_style = "heavy exchanges"
    elif captures_per_10 > 1.5:
        exch_style = "moderate trading"
    else:
        exch_style = "quiet, few trades"

    pc_spark = _sparkline([float(v) for v in pc], lo=min(pc), hi=max(pc))
    print(f"  Material arc    : {pc[0]} → {pc[-1]} pieces  "
          f"({total_captures} captures, {exch_style})")
    print(f"    {pc_spark}")

    exchanges = []
    for i in range(1, len(pc)):
        drop_pc = pc[i - 1] - pc[i]
        if drop_pc >= 2:
            mv_n = (i // 2) + 1
            sd = "white" if (i - 1) % 2 == 0 else "black"
            exchanges.append((mv_n, sd, drop_pc))
    if exchanges:
        print("  Exchanges       :")
        for mv_n, sd, dp in exchanges[:5]:
            print(f"    Move {mv_n} ({sd}) — {dp} pieces traded")
    print()

    # Centre pressure
    cp_traj = stats.get("center_pressure_trajectory")
    if cp_traj:
        avg_cp = sum(cp_traj) / len(cp_traj)
        cp_spark = _sparkline(cp_traj, 0.0, max(cp_traj) or 1.0)
        if avg_cp > 0.15:
            cp_style = "centre-dominant"
        elif avg_cp < 0.05:
            cp_style = "flank-oriented"
        else:
            cp_style = "mixed"
        print(f"  Centre pressure : avg {avg_cp:.3f} ({cp_style})")
        print(f"    {cp_spark}")
        print()

    # Territory
    print(f"  Territory (where pieces aimed to go):")
    print(f"    White half  {_bar(w_terr, 20)}  {w_terr*100:.0f}%")
    print(f"    Black half  {_bar(b_terr, 20)}  {b_terr*100:.0f}%")

    # -----------------------------------------------------------------------
    # Heatmaps side by side
    # -----------------------------------------------------------------------
    print()
    print("─" * 60)
    print("  SPATIAL HEATMAPS  (darker = more Q-probability mass)")
    print()
    dest_lines = _heatmap_board(stats["accumulated_dest_heatmap"], "Destination (where to move?)")
    src_lines  = _heatmap_board(stats["accumulated_src_heatmap"],  "Source      (which piece moved?)")
    for d, s in zip(dest_lines, src_lines):
        print(f"  {d:<30}    {s}")
    print()

    # -----------------------------------------------------------------------
    # Structural analysis (centrality, coordination, community, tension)
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  STRUCTURAL ANALYSIS")
    print()

    coord_spark    = _sparkline(stats["coordination_trajectory"],   0.0, 1.0, width=40)
    central_spark  = _sparkline(stats["centrality_trajectory"],     0.0, 1.0, width=40)
    tension_spark  = _sparkline(stats["tension_trajectory"],        0.0, 1.0, width=40)
    comm_spark     = _sparkline(
        [float(v) for v in stats["community_count_trajectory"]],
        0.0, max(stats["community_count_trajectory"]) or 1.0,
        width=40,
    )

    print(f"  Piece coordination  (active side, ↑ = tightly defended):")
    print(f"    {coord_spark}")
    print(f"  Mean centrality     (how connected pieces are, ↑ = high):")
    print(f"    {central_spark}")
    print(f"  Graph tension       (intra-color edge fraction, ↓ = more conflict):")
    print(f"    {tension_spark}")
    print(f"  Community clusters  (number of piece groups per position):")
    print(f"    {comm_spark}")
    print()
    print(f"  Game averages:")
    print(f"    Avg coordination : {stats['avg_coordination']:.3f}   "
          "(0 = isolated pieces, 1 = all mutually defending)")
    print(f"    Avg centrality   : {stats['avg_centrality']:.3f}   "
          "(higher = pieces in the thick of the action)")
    print(f"    Avg tension      : {stats['avg_tension']:.3f}   "
          "(lower = more interplay between the two armies)")
    print()

    # -----------------------------------------------------------------------
    # Structural fingerprint drift
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  STRUCTURAL FINGERPRINT DRIFT")
    print()

    drift_traj = stats.get("structural_drift_trajectory") or []
    if drift_traj:
        drift_spark = _sparkline(drift_traj, 0.0, max(drift_traj) or 1.0, width=40)
        print("  Local graph drift   (↑ = bigger change in relational structure):")
        print(f"    {drift_spark}")
        print()
        print(f"  Avg local drift     : {stats['avg_structural_drift']:.3f}")
        print(f"  Peak local drift    : {stats['peak_structural_drift']:.3f}")
        print(f"  Opening→final drift : {stats['final_structural_distance']:.3f}")

        turning_points = sorted(
            range(1, len(drift_traj)),
            key=lambda idx: drift_traj[idx],
            reverse=True,
        )[:3]
        if turning_points:
            print()
            print("  Biggest structural shifts:")
            for idx in turning_points:
                board_t = chess.Board(fens[idx - 1])
                move = chess.Move.from_uci(ucis[idx - 1])
                move_label = f"{((idx - 1) // 2) + 1}{'w' if (idx - 1) % 2 == 0 else 'b'}"
                print(
                    f"    {move_label:<4} {board_t.san(move):<8} "
                    f"drift {drift_traj[idx]:.3f}"
                )
        print()

    # -----------------------------------------------------------------------
    # Move Gap Analysis (dual-head model only)
    # -----------------------------------------------------------------------
    human_q_traj  = stats.get("human_q_trajectory")
    engine_q_traj = stats.get("engine_q_trajectory")
    if human_q_traj is not None and engine_q_traj is not None:
        print("─" * 60)
        print("  MOVE GAP ANALYSIS  (engine recommendation vs typical-player prediction)")
        print("  Shown only where the model's engine head and human head disagree on top-1 move.")
        print()

        # Find positions where engine top-1 ≠ human top-1
        gaps: list[tuple] = []
        for pos_idx in range(min(len(engine_q_traj), len(human_q_traj), len(ucis))):
            eq = engine_q_traj[pos_idx]
            hq = human_q_traj[pos_idx]
            if not eq or not hq:
                continue
            eng_top1_uci, eng_top1_prob = eq[0]
            hum_top1_uci, hum_top1_prob = hq[0]
            if eng_top1_uci != hum_top1_uci:
                played_uci = ucis[pos_idx]
                eng_rank_played = next(
                    (r + 1 for r, (u, _) in enumerate(eq) if u == played_uci), None
                )
                hum_rank_played = next(
                    (r + 1 for r, (u, _) in enumerate(hq) if u == played_uci), None
                )
                # Probability gap: how much more probable engine top-1 is vs human top-1
                prob_gap = abs(eng_top1_prob - hum_top1_prob)
                gaps.append((
                    pos_idx, eng_top1_uci, eng_top1_prob,
                    hum_top1_uci, hum_top1_prob,
                    played_uci, eng_rank_played, hum_rank_played,
                    prob_gap, eq[:3], hq[:3],
                ))

        if not gaps:
            print("  Engine and human heads agreed on top-1 move at every position.")
            print("  (Consider training the human head on player-game data to create divergence.)")
        else:
            print(f"  {'#':>3}  {'Side':<5}  {'Played':<7}  "
                  f"{'Eng.top1':<8}  {'Hum.top1':<8}  "
                  f"{'Eng.prob':>8}  {'Hum.prob':>8}")
            print("  " + "─" * 62)
            for (pos_idx, eng_t1, eng_p, hum_t1, hum_p,
                 played_uci, eng_rk, hum_rk, pgap, eq3, hq3) in gaps[:20]:
                b_s = chess.Board(fens[pos_idx])
                is_white = b_s.turn == chess.WHITE
                side_str = "White" if is_white else "Black"
                try:
                    played_san = b_s.san(chess.Move.from_uci(played_uci))
                    eng_san    = b_s.san(chess.Move.from_uci(eng_t1))
                    hum_san    = b_s.san(chess.Move.from_uci(hum_t1))
                except Exception:
                    played_san, eng_san, hum_san = played_uci, eng_t1, hum_t1
                print(f"  {pos_idx+1:>3}  {side_str:<5}  {played_san:<7}  "
                      f"{eng_san:<8}  {hum_san:<8}  "
                      f"{eng_p*100:>7.1f}%  {hum_p*100:>7.1f}%")
            if len(gaps) > 20:
                print(f"  … {len(gaps) - 20} more divergent positions omitted")

        if gaps:
            avg_gap = sum(g[8] for g in gaps) / len(gaps)
            print()
            print(f"  Divergent positions: {len(gaps)}/{min(len(engine_q_traj), len(ucis))}")
            print(f"  Avg probability gap : {avg_gap*100:.1f}%  "
                  f"(difference between engine and human top-1 probability)")
            print()
            print("  Interpretation: When the engine and human heads pick different moves,")
            print("  the human head is predicting how a player at your ELO *typically* plays.")
            print("  Study these positions — they reveal where your moves diverge from engine")
            print("  recommendations and what human-typical patterns look like at your level.")
        print()

    # -----------------------------------------------------------------------
    # Critical moments & tactical patterns
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  CRITICAL MOMENTS & TACTICAL PATTERNS")
    print()

    pin_traj = stats["pin_count_trajectory"]
    fork_traj = stats["fork_count_trajectory"]
    n_tac = len(pin_traj)

    tactical_positions = sum(
        1 for i in range(n_tac) if pin_traj[i] > 0 or fork_traj[i] > 0
    )
    tac_density = tactical_positions / n_tac if n_tac else 0

    print(f"  Tactical density : {tac_density * 100:.0f}% of positions "
          f"({tactical_positions}/{n_tac}) had active motifs")
    print(f"  Peak pins: {stats['peak_pins']}   "
          f"Peak forks: {stats['peak_forks']}")
    print()

    # Build critical moments table with per-piece tactical details
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
            played = board_t.san(chess.Move.from_uci(ucis[i]))
        drop_s = ""
        ed = stats.get("eval_drops")
        if ed and i < len(ed) and ed[i] is not None:
            d = ed[i]
            if d > 0.15:
                drop_s = f"BLUNDER {d:+.2f}"
            elif d > 0.075:
                drop_s = f"miss {d:+.2f}"
            elif d < -0.05:
                drop_s = f"exploited {d:+.2f}"
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

    # Tension arc (contested squares between armies)
    tension_vals = stats.get("tension_trajectory")
    if tension_vals:
        avg_tension_t = sum(tension_vals) / len(tension_vals)
        t_spark = _sparkline(tension_vals, 0.0, 1.0)
        print(f"  Tension arc (contested squares — ↓ = more cross-army conflict):")
        print(f"    {t_spark}  avg {avg_tension_t:.2f}")

    # Game-level tactical insights
    insights_tac: list[str] = []
    if tac_density > 0.3:
        insights_tac.append("  ● Tactically rich — motifs in >30% of positions.")
    elif tac_density > 0.1:
        insights_tac.append("  ● Moderate tactical content.")
    else:
        insights_tac.append("  · Quiet positional game — few tactical complications.")
    if stats["peak_pins"] >= 2:
        insights_tac.append("  ● Pin-heavy game — pin awareness was important.")
    if stats["peak_forks"] >= 2:
        insights_tac.append("  ● Multiple fork threats arose — check for missed wins.")
    for ins_t in insights_tac:
        print(ins_t)
    print()

    # -----------------------------------------------------------------------
    # GNN piece importance
    # -----------------------------------------------------------------------
    traj = stats.get("piece_importance_trajectory")
    if traj:
        print("─" * 60)
        print("  GNN PIECE IMPORTANCE TRAJECTORY")
        print("  (L2-norm of each piece's final GNN embedding, normalised per position)")
        print("  Measures how much model computation passed through each piece node.")
        print()

        # Collect all squares that appear across the game and their per-position importance
        all_squares: dict[str, list[float | None]] = {}
        n_pos = len(traj)
        for pos_idx, pos_dict in enumerate(traj):
            for sq, imp in pos_dict.items():
                if sq not in all_squares:
                    all_squares[sq] = [None] * n_pos
                all_squares[sq][pos_idx] = imp

        # Rank squares by mean importance (ignoring None = piece not on board)
        def _mean_imp(vals: list[float | None]) -> float:
            present = [v for v in vals if v is not None]
            return sum(present) / len(present) if present else 0.0

        def _peak_imp(vals: list[float | None]) -> float:
            present = [v for v in vals if v is not None]
            return max(present) if present else 0.0

        def _peak_pos(vals: list[float | None]) -> int:
            return max(
                (i for i, v in enumerate(vals) if v is not None),
                key=lambda i: vals[i] or 0.0,
                default=0,
            )

        ranked = sorted(all_squares.items(), key=lambda kv: _mean_imp(kv[1]), reverse=True)
        top_n = ranked[:8]

        # Top piece per move (the single most activated node at each position)
        top_per_move = []
        for pos_dict in traj:
            if pos_dict:
                best_sq = max(pos_dict, key=lambda s: pos_dict[s])
                top_per_move.append((best_sq, pos_dict[best_sq]))
            else:
                top_per_move.append(("—", 0.0))

        # Rank by variance: the pieces that fluctuate most are tactically relevant
        def _variance(vals: list[float | None]) -> float:
            present = [v for v in vals if v is not None]
            if len(present) < 2:
                return 0.0
            mean = sum(present) / len(present)
            return sum((v - mean) ** 2 for v in present) / len(present)

        ranked = sorted(all_squares.items(), key=lambda kv: _variance(kv[1]), reverse=True)
        top_n = ranked[:8]

        print(f"  {'Square':<8}  {'Sparkline (relative importance per position)':<42}  {'Avg':>5}  {'Var':>5}  {'@peak':>5}")
        print("  " + "─" * 68)
        for sq, vals in top_n:
            # Sparkline relative to each *position's* max — show rank within position
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
                    idx = int(round(rel * (len(_SPARK) - 1)))
                    spark_chars.append(_SPARK[max(0, min(len(_SPARK) - 1, idx))])
            spark_str = "".join(spark_chars)
            avg  = _mean_imp(vals)
            var  = _variance(vals)
            pmov = _peak_pos(vals) + 1
            print(f"  {sq:<8}  {spark_str:<42}  {avg:>5.2f}  {var:>5.3f}  {pmov:>5}")

        print()
        print(f"  Top piece by GNN activation at each half-move:")
        for i, (sq, imp) in enumerate(top_per_move):
            move_label = f"{(i // 2) + 1}{'w' if i % 2 == 0 else 'b'}"
            print(f"    {move_label:<5} {sq:<6} ({imp:.2f})", end="  ")
            if (i + 1) % 5 == 0:
                print()
        if len(top_per_move) % 5 != 0:
            print()
        print()

        # Game-level insights using relative ranking
        # Most volatile piece (highest variance = most tactically active)
        volatile = sorted(all_squares.items(), key=lambda kv: _variance(kv[1]), reverse=True)
        if volatile:
            sq_v, vals_v = volatile[0]
            print(f"  ● Most volatile piece (tactics indicator) : {sq_v}  (var={_variance(vals_v):.3f})")

        # Piece that was consistently *top-ranked* within positions the most often
        top_count: dict[str, int] = {}
        for pos_dict in traj:
            if pos_dict:
                best = max(pos_dict, key=lambda s: pos_dict[s])
                top_count[best] = top_count.get(best, 0) + 1
        if top_count:
            dominant_sq = max(top_count, key=lambda s: top_count[s])
            print(f"  ● Most frequently #1 piece               : {dominant_sq}  ({top_count[dominant_sq]} positions)")

        # Pieces whose importance spiked late (last quarter of the game)
        late_start = int(n_pos * 0.75)
        late_entry = [
            (sq, max(v for v in vals[late_start:] if v is not None), _variance(vals[late_start:]))
            for sq, vals in all_squares.items()
            if any(v is not None for v in vals[late_start:])
        ]
        if late_entry:
            sq_ls, imp_ls, var_ls = max(late_entry, key=lambda x: x[2])
            print(f"  ● Most dynamically active late-game piece : {sq_ls}  (late var={var_ls:.3f})")

        # Pieces present the whole game (long-lived) with highest mean relative rank
        full_game_sqs = [
            sq for sq, vals in all_squares.items()
            if sum(1 for v in vals if v is not None) >= int(n_pos * 0.75)
        ]
        if full_game_sqs:
            # Relative rank: fraction of positions where this piece is above median importance
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
            frac  = _above_median_frac(sq_fg)
            print(f"  ● Most sustained above-median piece       : {sq_fg}  ({frac*100:.0f}% of positions above median)")
        print()

    # -----------------------------------------------------------------------
    # Coaching insights
    # -----------------------------------------------------------------------
    print("─" * 60)
    print("  COACHING INSIGHTS")
    print()
    insights: list[str] = []
    for side in ("white", "black"):
        s = stats[side]
        if s is None:
            continue
        name = w_name if side == "white" else b_name
        b_count = len(s["blunders"])
        m_count = len(s["mistakes"])
        pct = s["avg_move_percentile"] or 0.0
        top1 = s["agreement_top1"] or 0.0

        if b_count > 0:
            blunder_moves = ", ".join(b["uci"] for b in s["blunders"])
            insights.append(f"  · {name}: {b_count} blunder(s) — re-examine moves {blunder_moves}.")
        if m_count > 1:
            insights.append(f"  · {name}: {m_count} mistakes — consider spending more time on tactics.")
        if top1 < 0.20:
            insights.append(f"  · {name}: top-1 agreement only {top1*100:.0f}% — model often preferred a different move.")
        if pct > 0.80:
            insights.append(f"  · {name}: strong average move quality ({pct*100:.0f}th percentile).")
        if pct < 0.55:
            insights.append(f"  · {name}: below-average move quality ({pct*100:.0f}th percentile) — review opening principles.")

    if stats["game_sharpness"] > 0.20:
        insights.append("  · The eval swung significantly — this was a tactically rich game.")
    if stats["decisiveness"] < 0.5:
        insights.append("  · Advantage changed hands frequently — neither side maintained momentum.")
    if not insights:
        insights.append("  · No significant errors detected (this is an untrained model — results improve with a trained checkpoint).")
    for ins in insights:
        print(ins)
    print()
    print("═" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",  default=DEFAULT_CHECKPOINT,
                   help="Path to .pt checkpoint (default: %(default)s)")
    p.add_argument("--calib",  default=DEFAULT_CALIB,
                   help="Path to calibration JSON sidecar (optional)")
    p.add_argument("--pgn",    default=DEFAULT_PGN,
                   help="Path to PGN file (default: %(default)s)")
    p.add_argument("--game",   type=int, default=1, metavar="N",
                   help="1-indexed game number in the PGN (default: 1)")
    p.add_argument("--lichess-game",
                   help="Lichess game id or URL to fetch via the API")
    p.add_argument("--lichess-token-env", default="LICHESS_API_TOKEN",
                   help="Environment variable holding an optional Lichess API token")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    print(f"\nLoading model from {args.model} …", flush=True)
    model = _load_model(args.model, device)

    tutor = CaseTutor(model, device)
    if args.calib:
        if os.path.isfile(args.calib):
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

    print_report(game, stats, elo_white_result, elo_black_result, fens, ucis, theoretical)


if __name__ == "__main__":
    main()
