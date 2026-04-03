"""Helpers for loading games from the Lichess API."""

from __future__ import annotations

import io
from typing import Final
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import chess.pgn

_DEFAULT_BASE_URL: Final[str] = "https://lichess.org"
_DEFAULT_TIMEOUT: Final[float] = 15.0
_USER_AGENT: Final[str] = "chess-graph-neural-network/lichess-analytics"


def normalise_lichess_game_id(game_id_or_url: str) -> str:
    """Extract an 8-character Lichess game id from a raw id or URL."""
    candidate = (game_id_or_url or "").strip()
    if not candidate:
        raise ValueError("Lichess game id must not be empty.")

    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.netloc not in {"lichess.org", "www.lichess.org"}:
            raise ValueError("Expected a lichess.org URL.")
        parts = [part for part in parsed.path.split("/") if part]
        if not parts:
            raise ValueError("Lichess URL does not contain a game id.")
        if parts[:2] == ["game", "export"] and len(parts) >= 3:
            candidate = parts[2]
        elif parts[:4] == ["api", "board", "game", "stream"] and len(parts) >= 5:
            candidate = parts[4]
        else:
            candidate = parts[0]

    candidate = candidate.strip("/")
    if len(candidate) != 8 or not candidate.isalnum():
        raise ValueError(
            "Lichess game id must be an 8-character alphanumeric id or URL."
        )
    return candidate


def build_lichess_export_url(
    game_id_or_url: str,
    *,
    base_url: str = _DEFAULT_BASE_URL,
) -> str:
    """Return the PGN export endpoint for a Lichess game."""
    game_id = normalise_lichess_game_id(game_id_or_url)
    query = urlencode(
        {
            "moves": "true",
            "tags": "true",
            "clocks": "false",
            "evals": "false",
            "pgnInJson": "false",
        }
    )
    return f"{base_url.rstrip('/')}/game/export/{game_id}?{query}"


def fetch_lichess_pgn(
    game_id_or_url: str,
    *,
    token: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    base_url: str = _DEFAULT_BASE_URL,
) -> str:
    """Fetch raw PGN text for a Lichess game."""
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": _USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        build_lichess_export_url(game_id_or_url, base_url=base_url),
        headers=headers,
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        pgn_text = response.read().decode(charset)

    if not pgn_text.strip():
        raise ValueError("Lichess API returned an empty PGN response.")
    return pgn_text


def read_lichess_game(
    game_id_or_url: str,
    *,
    token: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    base_url: str = _DEFAULT_BASE_URL,
) -> chess.pgn.Game:
    """Fetch and parse a Lichess game into a python-chess PGN object."""
    game = chess.pgn.read_game(
        io.StringIO(
            fetch_lichess_pgn(
                game_id_or_url,
                token=token,
                timeout=timeout,
                base_url=base_url,
            )
        )
    )
    if game is None:
        raise ValueError("Unable to parse PGN returned by the Lichess API.")
    return game
