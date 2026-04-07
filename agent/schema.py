"""Schema definitions for the chess coach Pydantic-AI agent.

``GameContext`` carries all data produced by ``CaseTutor.analyse_game()``
plus PGN metadata.  ``CoachingSession`` wraps it as the ``deps_type``
passed to every tool registered on the agent.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class GameContext(BaseModel):
    """All data needed to coach a single analysed game."""

    # Full output from CaseTutor.analyse_game()
    game_stats: dict[str, Any]

    # Sequential FENs (len = n_positions + 1)
    fens: list[str]

    # UCI strings played (len = n_moves)
    moves_uci: list[str]

    # PGN header metadata
    white_player: str = "White"
    black_player: str = "Black"
    elo_white: Optional[int] = None
    elo_black: Optional[int] = None
    result: str = "*"
    opening_name: str = ""
    eco: str = ""

    # ELO estimates from CaseTutor.estimate_elo() — None when unavailable
    elo_estimate_white: Optional[dict[str, Any]] = None
    elo_estimate_black: Optional[dict[str, Any]] = None

    # Optional theoretical analysis (from analyze_theoretical)
    theoretical: Optional[dict[str, Any]] = None


class CoachingSession(BaseModel):
    """Pydantic-AI ``deps_type`` for the chess coach agent."""

    context: GameContext
