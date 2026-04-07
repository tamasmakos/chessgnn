"""GameNarrator: wraps LLM calls for chess coaching narration.

Each public method corresponds to one coaching section. It builds a
grounded prompt via ``agent/prompts.py`` and calls Groq for the text.

If ``GROQ_API_KEY`` is not set, all methods return empty strings
gracefully — the numeric tables still print normally.
"""

from __future__ import annotations

import os
from typing import Any

import chess
import chess.pgn

from .prompts import (
    overview_prompt,
    opening_prompt,
    move_decisions_prompt,
    game_character_prompt,
    critical_moments_prompt,
    piece_activity_prompt,
    player_profile_prompt,
)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_MAX_TOKENS = 400


class GameNarrator:
    """Produces coaching prose paragraphs from ``analyse_game()`` statistics.

    Parameters
    ----------
    model_name : str, optional
        Groq model name.  Defaults to ``GROQ_MODEL`` env var or
        ``llama-3.3-70b-versatile``.
    temperature : float
        Sampling temperature.  0.3 gives consistent but natural coaching text.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        self._client = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=api_key)
            except ImportError:
                pass
        self._model = model_name or os.getenv("GROQ_MODEL") or _DEFAULT_MODEL
        self._temperature = temperature

    @property
    def available(self) -> bool:
        """True when a Groq client was successfully initialised."""
        return self._client is not None

    # -------------------------------------------------------------------
    # Core call
    # -------------------------------------------------------------------

    def narrate(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the coaching paragraph.

        Returns an empty string on any failure (no API key, network error,
        rate limit, etc.) so callers can safely print without checking.
        """
        if self._client is None:
            return ""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=_DEFAULT_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""

    # -------------------------------------------------------------------
    # Section-specific helpers
    # -------------------------------------------------------------------

    def narrate_overview(
        self,
        game: chess.pgn.Game,
        stats: dict[str, Any],
    ) -> str:
        ev = stats.get("eval_trajectory") or []
        h = game.headers
        return self.narrate(overview_prompt(
            white=h.get("White", "White"),
            black=h.get("Black", "Black"),
            elo_w=int(h.get("WhiteElo") or 0) or None,
            elo_b=int(h.get("BlackElo") or 0) or None,
            result=h.get("Result", "*"),
            opening_name=h.get("Opening", ""),
            n_plies=stats.get("n_moves", 0),
            sharpness=stats.get("game_sharpness", 0.0),
            decisiveness=stats.get("decisiveness", 0.5),
            start_eval=ev[0] if ev else 0.0,
            end_eval=ev[-1] if ev else 0.0,
        ))

    def narrate_opening(
        self,
        game: chess.pgn.Game,
        stats: dict[str, Any],
        fens: list[str],
        ucis: list[str],
    ) -> str:
        h = game.headers
        white = h.get("White", "White")
        black = h.get("Black", "Black")

        # Build SAN list for the opening moves
        board = chess.Board()
        opening_sans: list[str] = []
        for u in ucis[:16]:
            try:
                move = chess.Move.from_uci(u)
                opening_sans.append(board.san(move))
                board.push(move)
            except Exception:
                break

        # Opening extent: ends after 4+ captures or move 12
        pc = stats.get("piece_count_trajectory") or []
        opening_end_idx = min(len(pc) - 1, 24) if pc else 0
        for i in range(1, len(pc)):
            if pc[0] - pc[i] >= 4:
                opening_end_idx = i
                break

        mr = stats.get("move_ranks") or []
        mp = stats.get("move_percentiles") or []
        lt = stats.get("legal_moves_trajectory") or []
        ev = stats.get("eval_trajectory") or []

        n_op = min(opening_end_idx, len(mr))
        w_pcts = [mp[i] for i in range(0, n_op, 2) if i < len(mp) and mp[i] is not None]
        b_pcts = [mp[i] for i in range(1, n_op, 2) if i < len(mp) and mp[i] is not None]

        w_dev = b_dev = None
        for i in range(0, n_op, 2):
            if i < len(mr) and mr[i] is not None and mr[i] > 3:
                try:
                    brd = chess.Board(fens[i])
                    san = brd.san(chess.Move.from_uci(ucis[i]))
                    n_legal = lt[i] if i < len(lt) else mr[i]
                    w_dev = ((i // 2) + 1, san, mr[i], n_legal)
                except Exception:
                    pass
                break
        for i in range(1, n_op, 2):
            if i < len(mr) and mr[i] is not None and mr[i] > 3:
                try:
                    brd = chess.Board(fens[i])
                    san = brd.san(chess.Move.from_uci(ucis[i]))
                    n_legal = lt[i] if i < len(lt) else mr[i]
                    b_dev = ((i // 2) + 1, san, mr[i], n_legal)
                except Exception:
                    pass
                break

        exit_eval = ev[opening_end_idx] if opening_end_idx < len(ev) else 0.0
        return self.narrate(opening_prompt(
            white=white, black=black,
            eco=h.get("ECO", ""),
            opening_name=h.get("Opening", ""),
            opening_moves_san=opening_sans,
            deviation_white=w_dev,
            deviation_black=b_dev,
            exit_eval=exit_eval,
            w_opening_quality=sum(w_pcts) / len(w_pcts) if w_pcts else None,
            b_opening_quality=sum(b_pcts) / len(b_pcts) if b_pcts else None,
        ))

    def narrate_move_decisions(
        self,
        game: chess.pgn.Game,
        stats: dict[str, Any],
        fens: list[str],
        ucis: list[str],
    ) -> str:
        h = game.headers
        ws = stats.get("white") or {}
        bs = stats.get("black") or {}
        return self.narrate(move_decisions_prompt(
            white=h.get("White", "White"),
            black=h.get("Black", "Black"),
            blunders_white=ws.get("blunders") or [],
            mistakes_white=ws.get("mistakes") or [],
            blunders_black=bs.get("blunders") or [],
            mistakes_black=bs.get("mistakes") or [],
            best_moves_white=len(ws.get("best_moves") or []),
            best_moves_black=len(bs.get("best_moves") or []),
            avg_pct_white=ws.get("avg_move_percentile"),
            avg_pct_black=bs.get("avg_move_percentile"),
            fens=fens,
            ucis=ucis,
            eval_drops=stats.get("eval_drops"),
        ))

    def narrate_game_character(
        self,
        game: chess.pgn.Game,
        stats: dict[str, Any],
        ucis: list[str],
    ) -> str:
        pc = stats.get("piece_count_trajectory") or []
        cp = stats.get("center_pressure_trajectory") or []
        n_moves = max((len(ucis) + 1) // 2, 1)
        return self.narrate(game_character_prompt(
            white=game.headers.get("White", "White"),
            black=game.headers.get("Black", "Black"),
            sharpness=stats.get("game_sharpness", 0.0),
            avg_complexity=stats.get("avg_complexity", 0.0),
            avg_branching=stats.get("avg_branching", 0.0),
            total_captures=(pc[0] - pc[-1]) if len(pc) >= 2 else 0,
            n_moves=n_moves,
            w_territory=stats.get("white_territory", 0.5),
            b_territory=stats.get("black_territory", 0.5),
            avg_center_pressure=sum(cp) / len(cp) if cp else 0.0,
        ))

    def narrate_critical_moments(
        self,
        game: chess.pgn.Game,
        stats: dict[str, Any],
        critical_moments: list[tuple],
    ) -> str:
        pin_traj = stats.get("pin_count_trajectory") or []
        fork_traj = stats.get("fork_count_trajectory") or []
        n_tac = len(pin_traj)
        tactical_positions = sum(
            1 for i in range(n_tac) if pin_traj[i] > 0 or fork_traj[i] > 0
        )
        tac_density = tactical_positions / n_tac if n_tac else 0.0
        return self.narrate(critical_moments_prompt(
            white=game.headers.get("White", "White"),
            black=game.headers.get("Black", "Black"),
            tac_density=tac_density,
            peak_pins=stats.get("peak_pins", 0),
            peak_forks=stats.get("peak_forks", 0),
            critical_moments=critical_moments,
        ))

    def narrate_piece_activity(
        self,
        game: chess.pgn.Game,
        stats: dict[str, Any],
    ) -> str:
        traj = stats.get("piece_importance_trajectory") or []
        if not traj:
            return ""

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

        volatile = sorted(all_squares.items(), key=lambda kv: _var(kv[1]), reverse=True)
        top_by_mean = sorted(all_squares.items(), key=lambda kv: _mean(kv[1]), reverse=True)

        top_count: dict[str, int] = {}
        for pos_dict in traj:
            if pos_dict:
                best = max(pos_dict, key=lambda s: pos_dict[s])
                top_count[best] = top_count.get(best, 0) + 1
        dominant_sq = max(top_count, key=lambda s: top_count[s]) if top_count else None
        dominant_count = top_count.get(dominant_sq, 0) if dominant_sq else 0

        # Most sustained above-median piece
        full_game_sqs = [
            sq for sq, vals in all_squares.items()
            if sum(1 for v in vals if v is not None) >= int(n_pos * 0.75)
        ]
        sustained_sq: str | None = None
        sustained_frac = 0.0
        if full_game_sqs:
            def _above_med(sq: str) -> float:
                count = above = 0
                for pos_dict in traj:
                    if sq not in pos_dict:
                        continue
                    count += 1
                    med = sorted(pos_dict.values())[len(pos_dict) // 2]
                    if pos_dict[sq] >= med:
                        above += 1
                return above / count if count else 0.0

            sustained_sq = max(full_game_sqs, key=_above_med)
            sustained_frac = _above_med(sustained_sq)

        return self.narrate(piece_activity_prompt(
            most_volatile_sq=volatile[0][0] if volatile else None,
            volatile_var=_var(volatile[0][1]) if volatile else 0.0,
            most_dominant_sq=dominant_sq,
            dominant_count=dominant_count,
            n_positions=n_pos,
            sustained_sq=sustained_sq,
            sustained_frac=sustained_frac,
            top_pieces_by_mass=[(sq, _mean(vals)) for sq, vals in top_by_mean[:5]],
        ))

    def narrate_move(
        self,
        ply: int,
        side: str,
        player_name: str,
        san: str,
        rank: int,
        n_legal: int,
        percentile: float | None,
        eval_drop: float | None,
        engine_top: list[tuple[str, float]],
        fen_before: str,
        tactics: dict[str, Any] | None,
    ) -> str:
        """Produce a one- or two-sentence comment on a single notable move.

        Only called for moves that are blunders, mistakes, best moves, or inside
        a tactically active position, so the LLM budget stays reasonable.
        """
        move_no = (ply + 1) // 2
        rank_desc = f"ranked {rank}/{n_legal}"
        pct_desc = f"{percentile * 100:.0f}% of moves beaten" if percentile is not None else ""
        drop_desc = ""
        if eval_drop is not None:
            if eval_drop > 0.15:
                drop_desc = f"a significant blunder (eval drop {eval_drop:+.2f})"
            elif eval_drop > 0.07:
                drop_desc = f"a mistake (eval drop {eval_drop:+.2f})"
            elif eval_drop is not None and eval_drop < -0.05:
                drop_desc = "actually improved the position"

        alt_desc = ""
        if engine_top:
            alts = ", ".join(f"{s} ({v:+.2f})" for s, v in engine_top[:3])
            alt_desc = f"Engine alternatives: {alts}."

        tactic_desc = ""
        if tactics:
            parts = []
            if tactics.get("pins"):
                parts.append(f"pins on {', '.join(tactics['pins'])}")
            if tactics.get("forks"):
                for fk in tactics["forks"]:
                    parts.append(f"fork by {fk['attacker']}")
            if tactics.get("overloaded_squares"):
                parts.append(f"overloaded pieces on {', '.join(tactics['overloaded_squares'])}")
            if parts:
                tactic_desc = f"Tactical context: {'; '.join(parts)}."

        prompt = (
            f"You are a club chess coach. Briefly comment on move {move_no} ({side} played {san}). "
            f"It was {rank_desc}" + (f", {pct_desc}" if pct_desc else "") + ". "
            + (f"This was {drop_desc}. " if drop_desc else "")
            + (f"{alt_desc} " if alt_desc else "")
            + (f"{tactic_desc} " if tactic_desc else "")
            + "Write 1-2 sentences of practical coaching advice grounded in this data. "
            "Be specific about what the player should consider. Do not invent information."
        )
        return self.narrate(prompt)

    def narrate_player_profile(
        self,
        game: chess.pgn.Game,
        stats: dict[str, Any],
        side: str,
        elo_estimate: dict[str, Any] | None,
    ) -> str:
        h = game.headers
        name = h.get("White", "White") if side == "white" else h.get("Black", "Black")
        s = stats.get(side) or {}
        nm = s.get("moves_played") or 0

        blunder_rate = len(s.get("blunders") or []) / max(nm, 1)
        mistake_rate = len(s.get("mistakes") or []) / max(nm, 1)
        n_best = len(s.get("best_moves") or [])

        est_elo = elo_estimate.get("estimated_elo") if elo_estimate else None
        est_range = elo_estimate.get("confidence_range") if elo_estimate else None

        return self.narrate(player_profile_prompt(
            name=name,
            side=side,
            elo_estimate=est_elo,
            elo_range=est_range,
            avg_percentile=s.get("avg_move_percentile"),
            agreement_top1=s.get("agreement_top1"),
            blunder_rate=blunder_rate,
            mistake_rate=mistake_rate,
            n_best_moves=n_best,
            n_moves=nm,
        ))
