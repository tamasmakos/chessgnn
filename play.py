"""Interactive browser-based chess UI for the ChessGNN searchless engine.

Serves a self-contained single-page app over a local HTTP server.  The human
plays against the GNN engine; the sidebar shows the engine's top-5 suggestions,
win probability, and uncertainty after every half-move.

No Stockfish, no calibration, no new pip dependencies.

Usage
-----
    python play.py
    python play.py --model output/gateau_distilled.pt --color black --port 8765
"""

import argparse
import json
import logging
import math
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import chess
import chess.svg
import torch

from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel
from tutor import CaseTutor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "output/gateau_distilled.pt"
DEFAULT_PORT = 8765
BOARD_SVG_SIZE = 480
TOP_N = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("output/play.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(model_path: str, device: torch.device) -> GATEAUChessModel:
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

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
    logger.info(
        "Loaded GATEAUChessModel from %s  (hidden=%d, layers=%d)",
        model_path,
        hidden_channels,
        num_layers,
    )
    return model


# ---------------------------------------------------------------------------
# Game Session
# ---------------------------------------------------------------------------


class GameSession:
    """Holds all mutable game state for one browser session."""

    def __init__(self, tutor: CaseTutor, player_color: chess.Color) -> None:
        self._tutor = tutor
        self.player_color = player_color
        self.board = chess.Board()
        self.last_move: chess.Move | None = None
        self.selected_square: int | None = None
        self.rankings: list[tuple[chess.Move, float]] = []
        self.win_prob: float = 50.0
        self.uncertainty: float = 0.0
        self._refresh_rankings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_game(self, player_color: chess.Color) -> None:
        self.player_color = player_color
        self.board = chess.Board()
        self.last_move = None
        self.selected_square = None
        self._tutor.reset()
        self._refresh_rankings()

    def select_square(self, square: int | None) -> None:
        self.selected_square = square

    def apply_move(self, move: chess.Move) -> None:
        self.board.push(move)
        self.last_move = move
        self.selected_square = None
        self._tutor.update_state(self.board.fen())
        self._refresh_rankings()

    def engine_play(self) -> chess.Move | None:
        """Pick the engine's best move and apply it. Returns the move."""
        if not self.rankings:
            return None
        best_move = self.rankings[0][0]
        self.apply_move(best_move)
        return best_move

    def legal_dests_for(self, square: int) -> list[int]:
        """Return destination squares for all legal moves from *square*."""
        return [
            m.to_square
            for m in self.board.legal_moves
            if m.from_square == square
        ]

    def get_state_dict(self) -> dict:
        board = self.board
        is_player_turn = board.turn == self.player_color
        game_over = board.is_game_over()

        # Highlighted squares (legal destinations of selected piece)
        highlight_squares = chess.SquareSet()
        if self.selected_square is not None and not game_over:
            for sq in self.legal_dests_for(self.selected_square):
                highlight_squares.add(sq)

        # Green arrow pointing to engine's best suggestion
        arrows: list[chess.svg.Arrow] = []
        if self.rankings and not game_over:
            best = self.rankings[0][0]
            arrows.append(
                chess.svg.Arrow(best.from_square, best.to_square, color="#00aa44")
            )

        svg_str = chess.svg.board(
            board,
            size=BOARD_SVG_SIZE,
            flipped=(self.player_color == chess.BLACK),
            lastmove=self.last_move,
            squares=highlight_squares,
            arrows=arrows,
        )

        # Build ranking rows (from perspective of side to move)
        ranking_rows: list[dict] = []
        for move, score in self.rankings[:TOP_N]:
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            ranking_rows.append({"uci": move.uci(), "san": san, "score": round(score, 1)})

        # Legal moves for the side to move (for client-side validation)
        legal_uci = [m.uci() for m in board.legal_moves]

        result_str = None
        if game_over:
            outcome = board.outcome()
            if outcome is not None:
                if outcome.winner is None:
                    result_str = "Draw"
                elif outcome.winner == chess.WHITE:
                    result_str = "White wins"
                else:
                    result_str = "Black wins"
            else:
                result_str = "Game over"

        return {
            "svg": svg_str,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "player_color": "white" if self.player_color == chess.WHITE else "black",
            "is_player_turn": is_player_turn,
            "rankings": ranking_rows,
            "win_prob": round(self.win_prob, 1),
            "uncertainty": round(self.uncertainty, 3),
            "game_over": game_over,
            "result": result_str,
            "last_move_uci": self.last_move.uci() if self.last_move else None,
            "legal_moves_uci": legal_uci,
            "selected_square": chess.square_name(self.selected_square)
                if self.selected_square is not None
                else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_rankings(self) -> None:
        if self.board.is_game_over():
            self.rankings = []
            self.win_prob = 50.0
            self.uncertainty = 0.0
            return
        best_move, best_prob, ranking, uncertainty = self._tutor.recommend_move(
            self.board.fen()
        )
        self.rankings = ranking if ranking else []
        self.win_prob = best_prob if best_prob is not None else 50.0
        self.uncertainty = uncertainty if uncertainty is not None else 0.0


# ---------------------------------------------------------------------------
# HTML page (embedded)
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChessGNN Explorer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; }

  header {
    background: #16213e;
    border-bottom: 2px solid #0f3460;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  header h1 { font-size: 1.4rem; color: #e94560; letter-spacing: 1px; }
  header span { color: #888; font-size: 0.85rem; }

  .btn {
    padding: 7px 14px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn-white { background: #f5f5f5; color: #222; }
  .btn-black { background: #333; color: #f5f5f5; border: 1px solid #555; }
  .btn-group label { color: #aaa; font-size: 0.8rem; margin-right: 6px; }

  main {
    display: flex;
    gap: 24px;
    padding: 24px;
    flex: 1;
    align-items: flex-start;
    justify-content: center;
    flex-wrap: wrap;
  }

  #board-container {
    position: relative;
    width: 480px;
    height: 480px;
    flex-shrink: 0;
    border-radius: 4px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    cursor: pointer;
    user-select: none;
  }
  #board-container svg { display: block; border-radius: 4px; }

  /* Transparent overlay squares */
  #sq-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 480px; height: 480px;
    pointer-events: none;
  }
  #sq-overlay rect {
    pointer-events: all;
    fill: transparent;
    cursor: pointer;
  }
  #sq-overlay rect.selected { fill: rgba(100, 200, 100, 0.35); }
  #sq-overlay rect.dest     { fill: rgba(100, 180, 255, 0.30); }

  aside {
    width: 280px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 240px;
  }

  .card {
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 14px 16px;
  }
  .card h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 10px; }

  /* Status */
  #status-text {
    font-size: 1rem;
    font-weight: 600;
    padding: 8px 12px;
    border-radius: 6px;
    text-align: center;
    background: #0f3460;
    color: #e0e0e0;
    min-height: 2.2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.3s;
  }
  #status-text.your-turn   { background: #1a5e3e; color: #7fff9f; }
  #status-text.engine-turn { background: #3e1a1a; color: #ff9f7f; }
  #status-text.game-over   { background: #3e2a00; color: #ffd080; }

  /* Win prob */
  .prob-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .prob-label { font-size: 0.8rem; color: #aaa; width: 80px; }
  .prob-bar-outer { flex: 1; height: 10px; background: #0f3460; border-radius: 5px; overflow: hidden; }
  .prob-bar-inner { height: 100%; border-radius: 5px; transition: width 0.4s; }
  .prob-bar-inner.high { background: #2ecc71; }
  .prob-bar-inner.mid  { background: #f39c12; }
  .prob-bar-inner.low  { background: #e74c3c; }
  .prob-value { font-size: 0.85rem; font-weight: 600; min-width: 36px; text-align: right; }

  /* Uncertainty */
  .unc-row { display: flex; align-items: center; gap: 8px; }
  .unc-bar-inner { height: 8px; border-radius: 4px; background: #9b59b6; transition: width 0.4s; }

  /* Rankings table */
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  thead th { color: #888; font-weight: 600; text-align: left; padding: 2px 4px; font-size: 0.75rem; border-bottom: 1px solid #0f3460; }
  tbody tr { transition: background 0.15s; }
  tbody tr:hover { background: rgba(255,255,255,0.05); }
  tbody tr.best-move { background: rgba(0,170,68,0.12); }
  td { padding: 5px 4px; }
  td.rank { color: #888; font-size: 0.75rem; width: 20px; }
  td.san  { font-weight: 700; color: #c8e0ff; }
  td.score { text-align: right; width: 46px; color: #e0e0e0; }
  td .score-bar-outer { height: 6px; background: #0f3460; border-radius: 3px; overflow: hidden; margin-top: 2px; }
  td .score-bar-inner { height: 100%; border-radius: 3px; background: #3498db; }
  td.move-uci { color: #666; font-size: 0.75rem; font-family: monospace; }

  /* Promotion dialog */
  dialog {
    background: #16213e;
    border: 2px solid #0f3460;
    border-radius: 10px;
    padding: 24px;
    color: #e0e0e0;
    text-align: center;
  }
  dialog::backdrop { background: rgba(0,0,0,0.6); }
  dialog h3 { margin-bottom: 16px; font-size: 1.1rem; }
  .promo-buttons { display: flex; gap: 12px; justify-content: center; }
  .promo-btn {
    width: 56px; height: 56px;
    font-size: 2rem;
    background: #0f3460;
    border: 2px solid #3498db;
    border-radius: 8px;
    color: #fff;
    cursor: pointer;
    transition: background 0.15s;
  }
  .promo-btn:hover { background: #3498db; }

  #loading-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.35);
    z-index: 100;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    color: #fff;
  }
  #loading-overlay.active { display: flex; }
</style>
</head>
<body>

<header>
  <h1>ChessGNN Explorer</h1>
  <span>Searchless engine &bull; GNN</span>
  <div class="btn-group" style="margin-left:auto; display:flex; align-items:center; gap:8px;">
    <label>New game as:</label>
    <button class="btn btn-white" onclick="newGame('white')">&#9812; White</button>
    <button class="btn btn-black" onclick="newGame('black')">&#9818; Black</button>
  </div>
</header>

<main>
  <div id="board-container">
    <div id="board-svg"></div>
    <svg id="sq-overlay" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>

  <aside>
    <div id="status-text">Loading…</div>

    <div class="card">
      <h2>Engine Assessment</h2>
      <div class="prob-row">
        <span class="prob-label">Win prob</span>
        <div class="prob-bar-outer"><div id="win-bar" class="prob-bar-inner mid" style="width:50%"></div></div>
        <span id="win-pct" class="prob-value">50%</span>
      </div>
      <div class="unc-row">
        <span class="prob-label" style="font-size:0.8rem;color:#aaa;">Uncertainty</span>
        <div class="prob-bar-outer"><div id="unc-bar" class="unc-bar-inner" style="width:0%"></div></div>
        <span id="unc-val" class="prob-value" style="color:#9b59b6;">0.0</span>
      </div>
    </div>

    <div class="card">
      <h2>Top Suggestions</h2>
      <table>
        <thead><tr><th>#</th><th>Move</th><th>UCI</th><th>Win%</th></tr></thead>
        <tbody id="ranking-body"></tbody>
      </table>
    </div>
  </aside>
</main>

<!-- Promotion dialog -->
<dialog id="promo-dialog">
  <h3>Promote pawn to…</h3>
  <div class="promo-buttons">
    <button class="promo-btn" onclick="choosePromo('q')">♛</button>
    <button class="promo-btn" onclick="choosePromo('r')">♜</button>
    <button class="promo-btn" onclick="choosePromo('b')">♝</button>
    <button class="promo-btn" onclick="choosePromo('n')">♞</button>
  </div>
</dialog>

<div id="loading-overlay" class="active">Loading model…</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
const MARGIN = 15;          // chess.svg board margin in px
const SIZE   = 480;
const SQ     = (SIZE - 2 * MARGIN) / 8;   // ≈ 56.25 px per square

let state       = null;   // last /api/state response
let selectedSq  = null;   // algebraic (e.g. "e2"), or null
let pendingFrom = null;   // awaiting promotion choice
let pendingTo   = null;

// ── Helpers ────────────────────────────────────────────────────────────────

function sqNameToXY(sqName, flipped) {
  // sqName: "e4"  ->  file 4 (0=a), rank 3 (0=1)
  const file = sqName.charCodeAt(0) - 97;   // 0–7
  const rank = parseInt(sqName[1]) - 1;     // 0–7

  let col = flipped ? (7 - file) : file;
  let row = flipped ? rank       : (7 - rank);

  const x = MARGIN + col * SQ;
  const y = MARGIN + row * SQ;
  return { x, y, w: SQ, h: SQ };
}

function legalDestsFrom(sqName) {
  if (!state) return [];
  const prefix = sqName;
  return state.legal_moves_uci
    .filter(uci => uci.slice(0,2) === prefix)
    .map(uci => uci.slice(2,4));
}

function isOwnPiece(sqName) {
  // Check if the square contains the player's piece.
  // We do this by asking: does any legal move originate from sqName?
  if (!state || state.game_over || !state.is_player_turn) return false;
  return state.legal_moves_uci.some(uci => uci.slice(0,2) === sqName);
}

function isPawnPromotion(fromSq, toSq) {
  if (!state) return false;
  // Check if there's a promotion move in legal_moves for this from+to
  return state.legal_moves_uci.some(
    uci => uci.length === 5 && uci.slice(0,2) === fromSq && uci.slice(2,4) === toSq
  );
}

// ── Board overlay ──────────────────────────────────────────────────────────

function buildOverlay(flipped) {
  const svg = document.getElementById('sq-overlay');
  svg.innerHTML = '';

  const files = 'abcdefgh';
  const ranks  = '12345678';
  const dests  = selectedSq ? legalDestsFrom(selectedSq) : [];

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const fileIdx = flipped ? (7 - col) : col;
      const rankIdx = flipped ? row       : (7 - row);
      const sqName  = files[fileIdx] + ranks[rankIdx];

      const x = MARGIN + col * SQ;
      const y = MARGIN + row * SQ;

      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', x);
      rect.setAttribute('y', y);
      rect.setAttribute('width',  SQ);
      rect.setAttribute('height', SQ);
      rect.setAttribute('data-sq', sqName);

      if (sqName === selectedSq)            rect.classList.add('selected');
      else if (dests.includes(sqName))      rect.classList.add('dest');

      rect.addEventListener('click', () => handleSquareClick(sqName));
      svg.appendChild(rect);
    }
  }
}

// ── Click handling ─────────────────────────────────────────────────────────

async function handleSquareClick(sqName) {
  if (!state || state.game_over || !state.is_player_turn) return;

  if (selectedSq === null) {
    // First click: select own piece
    if (isOwnPiece(sqName)) {
      selectedSq = sqName;
      buildOverlay(state.player_color === 'black');
    }
    return;
  }

  // Second click
  if (sqName === selectedSq) {
    // Deselect
    selectedSq = null;
    buildOverlay(state.player_color === 'black');
    return;
  }

  const dests = legalDestsFrom(selectedSq);
  if (dests.includes(sqName)) {
    // Valid destination
    if (isPawnPromotion(selectedSq, sqName)) {
      pendingFrom = selectedSq;
      pendingTo   = sqName;
      document.getElementById('promo-dialog').showModal();
    } else {
      await sendMove(selectedSq + sqName);
    }
  } else if (isOwnPiece(sqName)) {
    // Re-select a different own piece
    selectedSq = sqName;
    buildOverlay(state.player_color === 'black');
  } else {
    selectedSq = null;
    buildOverlay(state.player_color === 'black');
  }
}

async function choosePromo(piece) {
  document.getElementById('promo-dialog').close();
  if (pendingFrom && pendingTo) {
    await sendMove(pendingFrom + pendingTo + piece);
    pendingFrom = pendingTo = null;
  }
}

// ── API calls ──────────────────────────────────────────────────────────────

async function sendMove(uci) {
  selectedSq = null;
  showLoading(true);
  try {
    const res = await fetch('/api/move', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ uci }),
    });
    const data = await res.json();
    if (res.ok) {
      applyState(data);
    } else {
      console.error('Move error:', data.error);
    }
  } catch (e) {
    console.error('sendMove failed:', e);
  } finally {
    showLoading(false);
  }
}

async function newGame(color) {
  selectedSq = null;
  showLoading(true);
  try {
    const res = await fetch('/api/new', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ color }),
    });
    const data = await res.json();
    if (res.ok) applyState(data);
  } catch (e) {
    console.error('newGame failed:', e);
  } finally {
    showLoading(false);
  }
}

async function loadState() {
  try {
    const res  = await fetch('/api/state');
    const data = await res.json();
    applyState(data);
  } catch (e) {
    console.error('loadState failed:', e);
  }
}

// ── Rendering ──────────────────────────────────────────────────────────────

function applyState(s) {
  state = s;
  const flipped = s.player_color === 'black';

  // Board SVG
  document.getElementById('board-svg').innerHTML = s.svg;

  // Overlay
  buildOverlay(flipped);

  // Status
  const statusEl = document.getElementById('status-text');
  if (s.game_over) {
    statusEl.textContent = s.result || 'Game over';
    statusEl.className   = 'game-over';
  } else if (s.is_player_turn) {
    statusEl.textContent = 'Your turn (' + s.player_color + ')';
    statusEl.className   = 'your-turn';
  } else {
    statusEl.textContent = 'Engine thinking…';
    statusEl.className   = 'engine-turn';
  }

  // Win prob bar
  const wp = s.win_prob;
  const bar = document.getElementById('win-bar');
  bar.style.width = wp + '%';
  bar.className = 'prob-bar-inner ' + (wp >= 60 ? 'high' : wp >= 40 ? 'mid' : 'low');
  document.getElementById('win-pct').textContent = wp.toFixed(1) + '%';

  // Uncertainty bar
  const unc = s.uncertainty;
  document.getElementById('unc-bar').style.width = (unc * 100) + '%';
  document.getElementById('unc-val').textContent  = unc.toFixed(2);

  // Rankings
  const tbody = document.getElementById('ranking-body');
  tbody.innerHTML = '';
  s.rankings.forEach((row, i) => {
    const tr = document.createElement('tr');
    if (i === 0) tr.classList.add('best-move');

    const scoreBarPct = Math.min(100, Math.max(0, row.score));

    tr.innerHTML = `
      <td class="rank">${i+1}</td>
      <td class="san">${row.san}</td>
      <td class="move-uci">${row.uci}</td>
      <td class="score">
        ${row.score.toFixed(1)}%
        <div class="score-bar-outer"><div class="score-bar-inner" style="width:${scoreBarPct}%"></div></div>
      </td>`;
    tbody.appendChild(tr);
  });
}

function showLoading(on) {
  document.getElementById('loading-overlay').classList.toggle('active', on);
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadState();
  showLoading(false);
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------


class _PlayHandler(BaseHTTPRequestHandler):
    """Serves the UI and handles JSON API requests for one local user."""

    session: "GameSession"   # set by run()

    # ------------------------------------------------------------------
    def log_message(self, format: str, *args) -> None:  # type: ignore[override]
        logger.debug(format, *args)

    # ------------------------------------------------------------------
    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_html(_HTML)
        elif path == "/api/state":
            self._send_json(self.session.get_state_dict())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        body = self._read_json_body()
        if body is None:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        if path == "/api/move":
            self._handle_move(body)
        elif path == "/api/new":
            self._handle_new(body)
        else:
            self.send_error(404)

    # ------------------------------------------------------------------
    def _handle_move(self, body: dict) -> None:
        uci_str = body.get("uci", "")
        session = self.session
        board = session.board

        if session.board.is_game_over():
            self._send_json({"error": "Game is over"}, 400)
            return

        if board.turn != session.player_color:
            self._send_json({"error": "Not your turn"}, 400)
            return

        try:
            move = chess.Move.from_uci(uci_str)
        except (ValueError, chess.InvalidMoveError):
            self._send_json({"error": f"Invalid UCI: {uci_str!r}"}, 400)
            return

        if move not in board.legal_moves:
            self._send_json({"error": f"Illegal move: {uci_str!r}"}, 400)
            return

        session.apply_move(move)
        logger.info("Human played %s", uci_str)

        # Engine responds if the game continues
        if not board.is_game_over() and board.turn != session.player_color:
            engine_move = session.engine_play()
            if engine_move:
                logger.info("Engine played %s", engine_move.uci())

        self._send_json(session.get_state_dict())

    def _handle_new(self, body: dict) -> None:
        color_str = body.get("color", "white").lower()
        player_color = chess.WHITE if color_str == "white" else chess.BLACK
        self.session.new_game(player_color)
        logger.info("New game: player is %s", color_str)

        # If player chose black, engine plays first as white
        if player_color == chess.BLACK:
            self.session.engine_play()

        self._send_json(self.session.get_state_dict())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChessGNN interactive browser UI")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to GATEAUChessModel checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string, e.g. cpu or cuda (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Local port for the HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Side for the human player (default: %(default)s)",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = _load_model(args.model, device)
    tutor = CaseTutor(model, device)

    player_color = chess.WHITE if args.color == "white" else chess.BLACK
    session = GameSession(tutor, player_color)

    # If playing as black, engine goes first
    if player_color == chess.BLACK:
        session.engine_play()
        logger.info("Engine opened as White")

    # Bind session to handler class (single global session, single-user server)
    _PlayHandler.session = session

    server = HTTPServer(("127.0.0.1", args.port), _PlayHandler)
    url = f"http://127.0.0.1:{args.port}/"
    logger.info("Serving at %s  — press Ctrl-C to stop", url)
    print(f"\n  Open in browser: {url}\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\nServer stopped.")


if __name__ == "__main__":
    run(_parse_args())
