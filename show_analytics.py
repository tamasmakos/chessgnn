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
from tutor import CaseTutor

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
    # Game character & territory
    # -----------------------------------------------------------------------
    print()
    print("─" * 60)
    print("  GAME CHARACTER")
    print()
    avg_unc  = stats["avg_complexity"]
    avg_br   = stats["avg_branching"]
    w_terr   = stats["white_territory"]
    b_terr   = stats["black_territory"]
    print(f"  Avg legal moves (branching factor) : {avg_br:.1f}")
    print(f"  Avg model uncertainty              : {avg_unc:.3f}  "
          f"({'highly tactical' if avg_unc > 0.97 else 'model confident' if avg_unc < 0.85 else 'mixed'})")
    print(f"  Avg pieces on board                : {stats['avg_piece_count']:.1f}")
    print()
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

    print_report(game, stats, elo_white_result, elo_black_result, fens, ucis)


if __name__ == "__main__":
    main()
