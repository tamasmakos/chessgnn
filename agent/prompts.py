"""Prompt template functions for the chess coach agent.

Each function returns a complete prompt string ready to send to an LLM.
No LLM calls are made here — this module is pure string construction.

Design principles:
- Inject all specific facts (move numbers, squares, eval values) from data.
- Always include a grounding note: "Only reference moves listed above."
- Ask for 2-4 sentence coaching paragraphs in plain club-player language.
"""

from __future__ import annotations

_COACH_PERSONA = (
    "You are a chess coach reviewing a completed game. Your style is clear, "
    "practical, and encouraging — suitable for club players rated 800–2000. "
    "You focus on patterns a player can act on: piece activity, pawn structure, "
    "king safety, and tactical themes like pins, forks, and overloaded defenders. "
    "IMPORTANT: Only reference moves, positions, and statistics explicitly listed "
    "in the data below. Never invent chess moves, squares, or numbers."
)


def _eval_pct(v: float) -> str:
    """Convert [-1, 1] eval to a win-probability percentage string."""
    return f"{(v + 1) / 2 * 100:.0f}%"


# ---------------------------------------------------------------------------
# Section prompts
# ---------------------------------------------------------------------------

def overview_prompt(
    white: str,
    black: str,
    elo_w: int | None,
    elo_b: int | None,
    result: str,
    opening_name: str,
    n_plies: int,
    sharpness: float,
    decisiveness: float,
    start_eval: float,
    end_eval: float,
) -> str:
    """Prompt for a 2-3 sentence game overview."""
    elo_w_str = str(elo_w) if elo_w else "?"
    elo_b_str = str(elo_b) if elo_b else "?"
    n_moves = (n_plies + 1) // 2

    sharp_desc = "volatile and tactical" if sharpness > 0.15 else "steady and positional"
    if decisiveness >= 0.85:
        decisive_desc = "one side maintained the advantage throughout"
    elif decisiveness < 0.5:
        decisive_desc = "the lead changed hands several times"
    else:
        decisive_desc = "there were some momentum swings"

    if result == "1-0":
        result_desc = "White won"
    elif result == "0-1":
        result_desc = "Black won"
    elif "1/2" in result:
        result_desc = "the game was drawn"
    else:
        result_desc = "the result is unknown"

    data = (
        f"GAME DATA:\n"
        f"White: {white} ({elo_w_str})  vs  Black: {black} ({elo_b_str})\n"
        f"Opening: {opening_name or 'unknown'}\n"
        f"Result: {result} ({result_desc})\n"
        f"Length: {n_plies} half-moves (~{n_moves} full moves)\n"
        f"Starting eval: {_eval_pct(start_eval)} white win probability\n"
        f"Final eval: {_eval_pct(end_eval)} white win probability\n"
        f"Game character: {sharp_desc}  |  "
        f"Lead stability: {decisiveness:.2f} ({decisive_desc})\n"
    )
    return (
        f"{_COACH_PERSONA}\n\n"
        f"{data}\n"
        "Write a 2-3 sentence coaching overview of this game. Describe the overall "
        "game character, which side controlled the position, and the outcome. Use "
        "plain language a club player would understand."
    )


def opening_prompt(
    white: str,
    black: str,
    eco: str,
    opening_name: str,
    opening_moves_san: list[str],
    deviation_white: tuple[int, str, int, int] | None,
    deviation_black: tuple[int, str, int, int] | None,
    exit_eval: float,
    w_opening_quality: float | None,
    b_opening_quality: float | None,
) -> str:
    """Prompt for 2-3 sentences on the opening phase."""
    moves_str = " ".join(
        f"{(i // 2) + 1}.{s}" if i % 2 == 0 else s
        for i, s in enumerate(opening_moves_san[:16])
    )

    dev_lines: list[str] = []
    for side_name, dev in ((white, deviation_white), (black, deviation_black)):
        if dev:
            mv, san, rank, n_legal = dev
            dev_lines.append(
                f"  {side_name} first deviation: move {mv} ({san}), "
                f"ranked {rank}/{n_legal} by the model"
            )

    quality_lines: list[str] = []
    if w_opening_quality is not None:
        quality_lines.append(f"  {white} opening accuracy: {w_opening_quality * 100:.0f}%")
    if b_opening_quality is not None:
        quality_lines.append(f"  {black} opening accuracy: {b_opening_quality * 100:.0f}%")

    dev_block = "\n".join(dev_lines) if dev_lines else "  No clear deviations detected."
    qual_block = "\n".join(quality_lines) if quality_lines else ""

    data = (
        f"OPENING DATA:\n"
        f"Classification: {(eco + ' ') if eco else ''}{opening_name or '(unclassified)'}\n"
        f"Moves played: {moves_str}\n"
        f"Opening exit evaluation: {_eval_pct(exit_eval)} white win probability "
        f"({exit_eval:+.2f})\n"
        f"{dev_block}\n"
        f"{qual_block}\n"
    )
    return (
        f"{_COACH_PERSONA}\n\n"
        f"{data}\n"
        "Write 2-3 sentences about the opening phase. Comment on the opening chosen, "
        "which player managed the opening better, and where either side first deviated "
        "from likely best play. Reference specific move numbers and move names only if "
        "they appear in the data above."
    )


def move_decisions_prompt(
    white: str,
    black: str,
    blunders_white: list[dict],
    mistakes_white: list[dict],
    blunders_black: list[dict],
    mistakes_black: list[dict],
    best_moves_white: int,
    best_moves_black: int,
    avg_pct_white: float | None,
    avg_pct_black: float | None,
    fens: list[str],
    ucis: list[str],
    eval_drops: list[float | None] | None,
) -> str:
    """Prompt for 3-4 sentences summarising move quality for both sides."""
    import chess as _chess

    def _moment(m: dict) -> str:
        ply = m["move_no"] - 1
        try:
            board = _chess.Board(fens[ply])
            san = board.san(_chess.Move.from_uci(ucis[ply]))
        except Exception:
            san = m["uci"]
        mv_full = (m["move_no"] + 1) // 2
        drop_pct = abs((m.get("eval_drop") or 0.0) * 50)
        return f"move {mv_full} ({san}, \u2212{drop_pct:.0f}% eval)"

    w_blunder_str = ", ".join(_moment(b) for b in blunders_white[:3]) or "none"
    b_blunder_str = ", ".join(_moment(b) for b in blunders_black[:3]) or "none"
    w_pct_str = f"{avg_pct_white * 100:.0f}%" if avg_pct_white is not None else "unknown"
    b_pct_str = f"{avg_pct_black * 100:.0f}%" if avg_pct_black is not None else "unknown"

    data = (
        f"MOVE QUALITY DATA:\n"
        f"{white}:\n"
        f"  Accuracy: {w_pct_str} (avg move percentile)  "
        f"Best moves (rank-1): {best_moves_white}  "
        f"Mistakes: {len(mistakes_white)}  Blunders: {w_blunder_str}\n"
        f"{black}:\n"
        f"  Accuracy: {b_pct_str}  "
        f"Best moves (rank-1): {best_moves_black}  "
        f"Mistakes: {len(mistakes_black)}  Blunders: {b_blunder_str}\n"
    )
    return (
        f"{_COACH_PERSONA}\n\n"
        f"{data}\n"
        "Write 3-4 sentences summarising the move quality of both sides. Highlight "
        "the most significant errors by move number (as listed above), and note "
        "which side played more accurately overall. Give one concrete, actionable "
        "suggestion to the player(s) with the most errors based on the patterns shown."
    )


def game_character_prompt(
    white: str,
    black: str,
    sharpness: float,
    avg_complexity: float,
    avg_branching: float,
    total_captures: int,
    n_moves: int,
    w_territory: float,
    b_territory: float,
    avg_center_pressure: float,
) -> str:
    """Prompt for 2-3 sentences on game style and material flow."""
    captures_per_10 = total_captures / max(n_moves, 1) * 10

    if avg_complexity > 0.97:
        char = "highly tactical"
    elif avg_complexity > 0.92:
        char = "sharp and tactical"
    elif avg_complexity > 0.85:
        char = "dynamic"
    elif avg_complexity > 0.70:
        char = "balanced"
    else:
        char = "positional and strategic"

    if w_territory > b_territory + 0.05:
        territory_note = (
            f"{white} pushed more into Black's territory "
            f"({w_territory * 100:.0f}% of moves aimed that way)."
        )
    elif b_territory > w_territory + 0.05:
        territory_note = (
            f"{black} pushed more into White's territory "
            f"({b_territory * 100:.0f}% of moves aimed that way)."
        )
    else:
        territory_note = "Both sides played evenly across the board."

    center_note = (
        "centre-dominant"
        if avg_center_pressure > 0.15
        else "flank-oriented"
        if avg_center_pressure < 0.05
        else "with mixed central and flank play"
    )

    data = (
        f"GAME CHARACTER DATA:\n"
        f"Style: {char}  (model uncertainty score: {avg_complexity:.2f})\n"
        f"Sharpness: {sharpness:.2f} "
        f"({'eval swings were volatile' if sharpness > 0.15 else 'eval stayed steady'})\n"
        f"Average branching: {avg_branching:.1f} legal moves per position\n"
        f"Total captures: {total_captures} ({captures_per_10:.1f} per 10 moves)\n"
        f"Territory: White aimed to Black's half {w_territory * 100:.0f}% | "
        f"Black aimed to White's half {b_territory * 100:.0f}%\n"
        f"Centre pressure: {center_note}\n"
        f"{territory_note}\n"
    )
    return (
        f"{_COACH_PERSONA}\n\n"
        f"{data}\n"
        "Write 2-3 sentences describing the character of this game. Explain the style "
        "(tactical or positional), how much material was exchanged, and which side "
        "controlled the space. Connect the style to what it reveals about each player's "
        "tendencies."
    )


def critical_moments_prompt(
    white: str,
    black: str,
    tac_density: float,
    peak_pins: int,
    peak_forks: int,
    critical_moments: list[tuple],
) -> str:
    """Prompt for 2-3 sentences on tactical patterns and critical positions."""
    moments_lines = ""
    for mv_n, sd, mot, played, dr in critical_moments[:4]:
        motif_short = mot[:60] + ".." if len(mot) > 60 else mot
        moments_lines += f"  Move {mv_n}{sd}: {motif_short} | Played: {played} | {dr}\n"

    data = (
        f"TACTICAL PATTERNS DATA:\n"
        f"Tactical density: {tac_density * 100:.0f}% of positions had active motifs\n"
        f"Peak simultaneous pins: {peak_pins}\n"
        f"Peak simultaneous forks: {peak_forks}\n"
        f"Critical positions:\n"
        f"{moments_lines or '  (no tactical motifs detected)'}"
    )
    return (
        f"{_COACH_PERSONA}\n\n"
        f"{data}\n"
        "Write 2-3 sentences about tactical patterns in this game. Describe "
        "the most important tactical theme (e.g. pins, forks), which player it "
        "affected, and whether the tactics were exploited or missed. Reference "
        "only the positions listed above."
    )


def piece_activity_prompt(
    most_volatile_sq: str | None,
    volatile_var: float,
    most_dominant_sq: str | None,
    dominant_count: int,
    n_positions: int,
    sustained_sq: str | None,
    sustained_frac: float,
    top_pieces_by_mass: list[tuple[str, float]],
) -> str:
    """Prompt for 2-3 sentences on which pieces drove the game."""
    top_str = (
        ", ".join(f"{sq} ({mass * 100:.0f}% mass)" for sq, mass in top_pieces_by_mass[:5])
        or "(not available)"
    )
    dominant_pct = (
        f"{dominant_count}/{n_positions} positions"
        if most_dominant_sq
        else "N/A"
    )

    data = (
        f"PIECE ACTIVITY DATA (policy probability mass per piece):\n"
        f"Most volatile piece: {most_volatile_sq or '?'}  (variance={volatile_var:.3f})\n"
        f"Most consistently top-ranked piece: {most_dominant_sq or '?'}  "
        f"({dominant_pct})\n"
        f"Most sustained above-median: {sustained_sq or '?'}  "
        f"({sustained_frac * 100:.0f}% of positions above median)\n"
        f"Top pieces by policy mass: {top_str}\n"
    )
    return (
        f"{_COACH_PERSONA}\n\n"
        f"{data}\n"
        "Write 2-3 sentences about piece activity in this game. Which pieces carried "
        "the most weight according to the model? Translate the policy-mass figures into "
        "practical chess terms (e.g. 'the queen on e5 was the most considered piece "
        "throughout the middlegame'). Mention if any piece appears underutilised."
    )


def player_profile_prompt(
    name: str,
    side: str,
    elo_estimate: int | None,
    elo_range: tuple[int, int] | None,
    avg_percentile: float | None,
    agreement_top1: float | None,
    blunder_rate: float,
    mistake_rate: float,
    n_best_moves: int,
    n_moves: int,
) -> str:
    """Prompt for 3-4 sentences coaching a specific player."""
    est_str = (
        f"{elo_estimate} [{elo_range[0]}\u2013{elo_range[1]}]"
        if elo_estimate and elo_range
        else "not available"
    )
    pct_str = f"{avg_percentile * 100:.0f}%" if avg_percentile is not None else "unknown"
    t1_str = f"{agreement_top1 * 100:.0f}%" if agreement_top1 is not None else "unknown"

    strengths: list[str] = []
    weaknesses: list[str] = []
    if avg_percentile is not None:
        if avg_percentile > 0.80:
            strengths.append("strong average move quality")
        elif avg_percentile < 0.55:
            weaknesses.append("below-average move selection")
    if blunder_rate < 0.05:
        strengths.append("low blunder rate")
    elif blunder_rate > 0.15:
        weaknesses.append("high blunder rate (work on calculation)")
    if agreement_top1 is not None:
        if agreement_top1 > 0.35:
            strengths.append("frequently selects top-engine moves")
        elif agreement_top1 < 0.15:
            weaknesses.append("rarely matches engine's top choice")
    best_rate = n_best_moves / max(n_moves, 1)
    if best_rate > 0.25:
        strengths.append(f"played {n_best_moves} best moves ({best_rate * 100:.0f}%)")

    data = (
        f"PLAYER PROFILE: {name} ({side.upper()})\n"
        f"Estimated ELO: {est_str}\n"
        f"Accuracy (avg move percentile): {pct_str}\n"
        f"Engine top-1 agreement: {t1_str}\n"
        f"Blunder rate: {blunder_rate * 100:.0f}%  |  "
        f"Mistake rate: {mistake_rate * 100:.0f}%\n"
        f"Best (rank-1) moves: {n_best_moves}/{n_moves}\n"
        f"Strengths: {', '.join(strengths) if strengths else 'none identified'}\n"
        f"Areas to improve: {', '.join(weaknesses) if weaknesses else 'none identified'}\n"
    )
    return (
        f"{_COACH_PERSONA}\n\n"
        f"{data}\n"
        f"Write 3-4 sentences coaching {name}. Start with something they did well, "
        "then address the most important area to improve based on the statistics above. "
        "Give one specific, actionable training recommendation (e.g. practise tactical "
        "puzzles, review opening principles, study endgame technique). Be encouraging "
        "but honest."
    )
