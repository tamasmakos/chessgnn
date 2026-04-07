"""Chess coach Pydantic-AI agent.

The agent uses Groq (Llama-3.3-70B) and has access to six tools that
pull structured analytics data from ``CoachingSession.context``.

For one-shot coaching reports, prefer ``agent.narrator.GameNarrator``
which calls the LLM directly section-by-section.  This agent is
designed for future interactive Q&A (``coach.py --interactive``).
"""

import os

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.groq import GroqModel
    _PYDANTIC_AI = True
except ImportError:
    Agent = None  # type: ignore[assignment,misc]
    GroqModel = None  # type: ignore[assignment,misc]
    _PYDANTIC_AI = False

from .schema import CoachingSession
from .tools import (
    explain_critical_moves,
    get_game_summary,
    get_move_detail,
    get_opening_context,
    get_piece_activity,
    get_player_profile,
)

_SYSTEM_PROMPT = """\
You are a chess coach analysing a completed game on behalf of the players.

Coaching style:
- Clear, practical, encouraging — suitable for club players (ELO 800–2000).
- Focus on patterns a player can act on: piece activity, pawn structure,
  king safety, tactical themes (pins, forks, overloaded defenders).
- ALWAYS ground your claims in data returned by your tools.
  Never invent chess moves, positions, or statistics not returned by a tool.
- When asked about a specific move, call get_move_detail(ply) first.
- When asked for an overview, call get_game_summary() first.
- Keep coaching paragraphs concise: 2–4 sentences per section.
- Translate numbers into chess language: say "the model preferred Nf3 here"
  rather than "rank=1".
- End each section with one actionable takeaway when possible.

Tone: honest about mistakes, optimistic about improvement.
"""

if _PYDANTIC_AI:
    model = GroqModel(
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    agent: "Agent[CoachingSession, str]" = Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        deps_type=CoachingSession,
    )
    agent.tool(get_game_summary)
    agent.tool(get_opening_context)
    agent.tool(explain_critical_moves)
    agent.tool(get_piece_activity)
    agent.tool(get_player_profile)
    agent.tool(get_move_detail)
else:
    agent = None  # type: ignore[assignment]
