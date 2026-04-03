import io

import chess.pgn
import pytest

from chessgnn.lichess_api import (
    build_lichess_export_url,
    fetch_lichess_pgn,
    normalise_lichess_game_id,
    read_lichess_game,
)

_SAMPLE_PGN = """\
[Event "Rated Classical game"]
[Site "https://lichess.org/j1dkb5dw"]
[White "BFG9k"]
[Black "mamalak"]
[Result "1-0"]

1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5
8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0
"""


class _FakeHeaders:
    @staticmethod
    def get_content_charset():
        return "utf-8"


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")
        self.headers = _FakeHeaders()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_normalise_lichess_game_id_accepts_raw_id() -> None:
    assert normalise_lichess_game_id("j1dkb5dw") == "j1dkb5dw"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://lichess.org/j1dkb5dw", "j1dkb5dw"),
        ("https://lichess.org/j1dkb5dw/black", "j1dkb5dw"),
        ("https://lichess.org/game/export/j1dkb5dw", "j1dkb5dw"),
        ("https://lichess.org/api/board/game/stream/j1dkb5dw", "j1dkb5dw"),
    ],
)
def test_normalise_lichess_game_id_accepts_lichess_urls(
    value: str,
    expected: str,
) -> None:
    assert normalise_lichess_game_id(value) == expected


def test_normalise_lichess_game_id_rejects_non_lichess_url() -> None:
    with pytest.raises(ValueError, match="lichess.org URL"):
        normalise_lichess_game_id("https://example.com/j1dkb5dw")


def test_build_lichess_export_url_uses_export_endpoint() -> None:
    url = build_lichess_export_url("https://lichess.org/j1dkb5dw")
    assert url.startswith("https://lichess.org/game/export/j1dkb5dw?")
    assert "moves=true" in url
    assert "tags=true" in url
    assert "pgnInJson=false" in url


def test_fetch_lichess_pgn_includes_authorization_header(monkeypatch) -> None:
    captured = {}

    def _fake_urlopen(request, timeout):
        captured["accept"] = request.headers["Accept"]
        captured["authorization"] = request.headers["Authorization"]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _FakeResponse(_SAMPLE_PGN)

    monkeypatch.setattr("chessgnn.lichess_api.urlopen", _fake_urlopen)

    pgn = fetch_lichess_pgn("j1dkb5dw", token="secret", timeout=3.5)

    assert "[Site \"https://lichess.org/j1dkb5dw\"]" in pgn
    assert captured["accept"] == "application/x-chess-pgn"
    assert captured["authorization"] == "Bearer secret"
    assert captured["url"].startswith("https://lichess.org/game/export/j1dkb5dw?")
    assert captured["timeout"] == 3.5


def test_read_lichess_game_parses_pgn(monkeypatch) -> None:
    monkeypatch.setattr(
        "chessgnn.lichess_api.urlopen",
        lambda request, timeout: _FakeResponse(_SAMPLE_PGN),
    )

    game = read_lichess_game("j1dkb5dw")

    assert isinstance(game, chess.pgn.Game)
    assert game.headers["White"] == "BFG9k"
    assert list(game.mainline_moves())
