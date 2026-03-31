import math
import torch
import chess
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.calibration import TemperatureScaler

# Move-quality thresholds (win-probability units, i.e. [0, 1] scale)
_BLUNDER_THRESHOLD = 0.15   # eval drop classifying a move as a blunder
_MISTAKE_THRESHOLD = 0.075  # …as a mistake

# Centre squares used for pressure metrics
_CENTER_SQUARES = frozenset({chess.D4, chess.D5, chess.E4, chess.E5})
_EXT_CENTER_SQUARES = frozenset({          # expanded centre: c3–f6
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
})


class CaseTutor:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self._use_q_head = hasattr(model, 'forward_with_q')
        self.builder = ChessGraphBuilder(
            use_move_edges=self._use_q_head,
            use_global_node=self._use_q_head,
        )
        self.current_hidden = None
        self._scaler: TemperatureScaler | None = None

    def set_calibration(self, scaler: TemperatureScaler | None) -> None:
        """Attach or detach a TemperatureScaler for win-probability calibration."""
        self._scaler = scaler

    def reset(self):
        """Resets the internal hidden state (New Game)."""
        self.current_hidden = None
        
    def update_state(self, fen: str):
        """
        Advances the internal hidden state with the played moves.
        Call this AFTER a move is committed to the board.
        No-op for models without a recurrent state (e.g. GATEAUChessModel).
        """
        if not hasattr(self.model, 'forward_step'):
            return
        graph = self.builder.fen_to_graph(fen).to(self.device)
        with torch.no_grad():
             _, self.current_hidden = self.model.forward_step(graph, self.current_hidden)

    def recommend_move(self, fen: str, user_elo: int | None = None, explain: bool = False):
        """
        Returns the best move for the current position.

        Uses a single forward pass via the Q-head when the model supports it
        (GATEAUChessModel), otherwise falls back to a per-successor rollout.

        Parameters
        ----------
        fen : str
            Current board position in FEN notation.
        user_elo : int | None
            Player ELO rating.  When provided, the model's value head is
            conditioned on this ELO so that win-probability estimates match
            that skill level.  1500 ≈ average club player; 2800 ≈ elite GM.
            ``None`` defaults to 3000 (Stockfish / perfect-play perspective).
        explain : bool
            When True, a fifth element ``internals`` dict is appended to the
            return tuple.  Contains plot-ready arrays describing what the model
            "sees" at this position::

                {
                  "win_prob_raw":  float,          # tanh value head output [-1, 1]
                  "q_distribution": [              # all moves, best-first
                    {"uci": str, "prob": float, "from": str, "to": str}, ...
                  ],
                  "top_moves": [...],              # top-5 slice of q_distribution
                  "dest_heatmap": [[float]*8]*8,   # [rank][file], max prob landing here
                  "src_heatmap":  [[float]*8]*8,   # [rank][file], max prob leaving here
                  "elo_diff": {                    # None when elo_norm == 0.5
                    "low_top5":  [{"uci": str, "prob": float}, ...],  # 1500-Elo view
                    "high_top5": [{"uci": str, "prob": float}, ...],  # current-Elo view
                  },
                }

            The heatmaps use rank-0 = rank-1 (a1 side), file-0 = a-file.
            If ``explain=False`` (default) only four values are returned so
            existing callers are unaffected.

        Returns
        -------
        best_move : chess.Move or None
        best_prob : float
            Win probability for the side to move (0–100 %).  Calibrated if a
            TemperatureScaler has been attached via ``set_calibration``.
        ranking : list of (chess.Move, float)
            All legal moves sorted best-first.
        uncertainty : float
            Normalised entropy of the Q-score distribution (0 = certain, 1 =
            uniform).  Always 0.0 on the rollout path.
        internals : dict  (only present when ``explain=True``)
        """
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            if explain:
                return None, 0.0, [], 0.0, {}
            return None, 0.0, [], 0.0

        elo_norm = min(max(user_elo, 0), 3000) / 3000.0 if user_elo is not None else 1.0

        if self._use_q_head:
            return self._recommend_q(board, legal_moves, fen, elo_norm=elo_norm, explain=explain)
        result = self._recommend_rollout(board, legal_moves)
        if explain:
            return *result, {}
        return result

    # ------------------------------------------------------------------
    # Private inference paths
    # ------------------------------------------------------------------

    def _recommend_q(
        self,
        board: chess.Board,
        legal_moves: list,
        fen: str,
        elo_norm: float = 1.0,
        explain: bool = False,
    ):
        """Single-pass move ranking via Q-head."""
        graph = self.builder.fen_to_graph(fen).to(self.device)
        with torch.no_grad():
            value, q_scores, _ = self.model.forward_with_q(graph, elo_norm=elo_norm)

        q_list = q_scores.cpu().tolist()
        if len(q_list) != len(legal_moves):
            # Safety fallback (should not normally occur)
            result = self._recommend_rollout(board, legal_moves)
            if explain:
                return *result, {}
            return result

        # Normalised entropy: H / log(M), range [0, 1]
        m = len(q_list)
        probs = torch.softmax(q_scores, dim=0)
        log_m = math.log(m) if m > 1 else 1.0
        entropy = float(-torch.sum(probs * torch.log(probs.clamp(min=1e-9))).item())
        uncertainty = float(entropy / log_m)

        is_white_turn = board.turn == chess.WHITE
        move_scores = [
            (move, (math.tanh(q) + 1) / 2 * 100)
            for move, q in zip(legal_moves, q_list)
        ]

        if is_white_turn:
            move_scores.sort(key=lambda x: x[1], reverse=True)
            best_move, best_prob = move_scores[0]
        else:
            move_scores.sort(key=lambda x: x[1])
            best_move = move_scores[0][0]
            best_prob = 100.0 - move_scores[0][1]

        if self._scaler is not None:
            best_prob = self._scaler.calibrate(best_prob / 100.0) * 100.0

        if explain:
            internals = self._build_explain(
                legal_moves, q_scores, value, graph, elo_norm
            )
            return best_move, best_prob, move_scores, uncertainty, internals

        return best_move, best_prob, move_scores, uncertainty

    def _build_explain(
        self,
        legal_moves: list,
        q_scores: torch.Tensor,
        value: torch.Tensor,
        graph,
        elo_norm: float,
    ) -> dict:
        """Build a plot-ready internals dict from a completed forward pass.

        All tensors are detached and converted to plain Python scalars/lists
        so the returned dict is safe to serialise to JSON or pass directly to
        matplotlib / plotly without further processing.
        """
        probs = torch.softmax(q_scores.detach(), dim=0).cpu()  # [M]
        prob_list = probs.tolist()

        # ---- q_distribution: all moves sorted best → worst ----
        indexed = sorted(
            enumerate(zip(legal_moves, prob_list)),
            key=lambda x: -x[1][1],
        )
        q_distribution = [
            {
                "uci":  move.uci(),
                "prob": round(p, 6),
                "from": chess.square_name(move.from_square),
                "to":   chess.square_name(move.to_square),
            }
            for _, (move, p) in indexed
        ]
        top_moves = q_distribution[:5]

        # ---- 8×8 heatmaps ----
        dest_heatmap = [[0.0] * 8 for _ in range(8)]
        src_heatmap  = [[0.0] * 8 for _ in range(8)]
        for move, p in zip(legal_moves, prob_list):
            dr, df = chess.square_rank(move.to_square),   chess.square_file(move.to_square)
            sr, sf = chess.square_rank(move.from_square), chess.square_file(move.from_square)
            if p > dest_heatmap[dr][df]:
                dest_heatmap[dr][df] = round(p, 6)
            if p > src_heatmap[sr][sf]:
                src_heatmap[sr][sf] = round(p, 6)

        # ---- ELO diff: compare low-elo (1500) vs current elo ----
        elo_diff: dict | None = None
        low_elo_norm = 0.5  # ≈ 1500 Elo
        if abs(elo_norm - low_elo_norm) > 0.05:
            with torch.no_grad():
                _, q_low, _ = self.model.forward_with_q(graph, elo_norm=low_elo_norm)
            probs_low  = torch.softmax(q_low.detach(), dim=0).cpu().tolist()
            probs_high = prob_list  # elo_norm is "high" here

            def _top5(prob_source: list) -> list:
                ranked = sorted(
                    zip(legal_moves, prob_source),
                    key=lambda x: -x[1],
                )
                return [{"uci": m.uci(), "prob": round(p, 6)} for m, p in ranked[:5]]

            elo_diff = {
                "low_top5":  _top5(probs_low),
                "high_top5": _top5(probs_high),
            }

        return {
            "win_prob_raw":   round(float(value.item()), 6),
            "q_distribution": q_distribution,
            "top_moves":      top_moves,
            "dest_heatmap":   dest_heatmap,
            "src_heatmap":    src_heatmap,
            "elo_diff":       elo_diff,
        }

    @staticmethod
    def _extract_scalar(step_output) -> float:
        """
        Extract a [-1, 1] scalar from a forward_step return value.

        GATEAUChessModel returns a Tensor [1, 1].
        STHGATLikeModel returns (win_logits [3], mat [1], dom [1]); convert
        to a scalar via softmax(white) - softmax(black).
        """
        if isinstance(step_output, torch.Tensor):
            return step_output.item()
        # Legacy tuple: (win_logits [3], mat, dom)
        win_logits = step_output[0]
        probs = torch.softmax(win_logits, dim=0)
        return (probs[0] - probs[2]).item()

    def _recommend_rollout(self, board: chess.Board, legal_moves: list):
        """Per-successor rollout using forward_step (legacy path)."""
        move_scores = []
        is_white_turn = board.turn == chess.WHITE

        for move in legal_moves:
            board.push(move)
            next_fen = board.fen()

            graph = self.builder.fen_to_graph(next_fen)
            graph = graph.to(self.device)

            with torch.no_grad():
                step_out, _ = self.model.forward_step(graph, self.current_hidden)
                raw_score = self._extract_scalar(step_out)

            white_win_prob = (raw_score + 1) / 2 * 100
            move_scores.append((move, white_win_prob))
            board.pop()

        if is_white_turn:
            move_scores.sort(key=lambda x: x[1], reverse=True)
            best_move = move_scores[0][0]
            best_prob = move_scores[0][1]
        else:
            move_scores.sort(key=lambda x: x[1], reverse=False)
            best_move = move_scores[0][0]
            best_prob = 100.0 - move_scores[0][1]

        if self._scaler is not None:
            best_prob = self._scaler.calibrate(best_prob / 100.0) * 100.0

        return best_move, best_prob, move_scores, 0.0

    # ------------------------------------------------------------------
    # Post-game analytics
    # ------------------------------------------------------------------

    def analyse_game(
        self,
        fens: list[str],
        moves_uci: list[str] | None = None,
        elo_white: int | None = None,
        elo_black: int | None = None,
    ) -> dict:
        """Comprehensive post-game analysis.

        Runs a single GRU-threaded forward pass through every position in the
        game, extracting per-position evaluation, uncertainty, and topological
        graph metrics, then aggregates them into game-level statistics.

        Parameters
        ----------
        fens : list[str]
            FEN for every board position, starting from the initial setup.
            ``fens[i]`` is the position **before** ``moves_uci[i]`` is played.
        moves_uci : list[str], optional
            UCI strings of the moves played (length = len(fens) − 1).
            Required for per-move quality statistics (rank, percentile,
            blunders, ELO estimation).
        elo_white, elo_black : int, optional
            Player ELO ratings used to condition the value head.  1500 used
            when not provided.

        Returns
        -------
        dict — all values are plain Python types (JSON-safe).

        Trajectory keys (one entry per position analysed):

        * ``eval_trajectory``            — value-head output in [−1, 1], white perspective.
          Positive = white ahead. Convert to % via ``(v + 1) / 2 * 100``.
        * ``uncertainty_trajectory``     — normalised Shannon entropy of Q distribution [0, 1].
          1 = model has no preference (all moves equally likely).
        * ``legal_moves_trajectory``     — branching factor (# legal moves) per position.
        * ``piece_count_trajectory``     — pieces remaining on board per position.
        * ``center_pressure_trajectory`` — sum of Q-softmax probability on {d4,d5,e4,e5}.
        * ``q_gini_trajectory``          — Gini coefficient of Q-softmax distribution.
          1 = all probability on one move (clear best move). 0 = uniform.

        Graph / topological aggregations:

        * ``accumulated_dest_heatmap``   — [8][8] normalised, indexed [rank][file].
          Rank 0 = white's back rank (1st), file 0 = a-file.
          Captures where both sides most wanted to move pieces.
        * ``accumulated_src_heatmap``    — [8][8] same for source squares.
          Shows which pieces were most "activated" by the model.
        * ``white_territory``            — fraction of dest mass on ranks 5–8.
        * ``black_territory``            — fraction of dest mass on ranks 1–4.
        * ``avg_branching``              — mean legal-move count (positional complexity).
        * ``avg_piece_count``
        * ``avg_complexity``             — mean uncertainty (tactical richness proxy).
        * ``game_sharpness``             — std of eval trajectory (how volatile the game was).
        * ``decisiveness``               — monotonicity score [0, 1].
          1 = eval moved steadily in one direction. 0 = constantly reversed.

        Move-quality keys (only when ``moves_uci`` is provided):

        * ``move_ranks``                 — 1-indexed rank of each played move.
        * ``move_percentiles``           — fraction of legal moves beaten [0, 1].
          1.0 = best move played. 0.0 = worst move played.
        * ``eval_drops``                 — win-prob change from the mover's perspective
          (positive = advantage lost, negative = advantage gained).
        * ``white`` / ``black``          — per-side dict with:
            - ``moves_played``, ``avg_move_rank``, ``avg_move_percentile``
            - ``agreement_top1``, ``agreement_top3``
            - ``blunders`` (eval_drop > 0.15), ``mistakes`` (> 0.075),
              ``best_moves`` (< −0.075)
            - ``avg_uncertainty_faced``
        """
        if not self._use_q_head:
            raise RuntimeError(
                "analyse_game requires a model with forward_with_q (GATEAUChessModel)"
            )

        elo_w = min(max(elo_white or 1500, 0), 3000) / 3000.0
        elo_b = min(max(elo_black or 1500, 0), 3000) / 3000.0

        # Per-position collected data
        eval_traj: list[float] = []
        unc_traj:  list[float] = []
        legal_traj: list[int]  = []
        piece_traj: list[int]  = []
        center_traj: list[float] = []
        gini_traj:   list[float] = []
        # Q distributions for move attribution: sorted (uci, prob) best-first
        q_dists: list[list[tuple[str, float]]] = []
        # Accumulated heatmaps (raw, normalised later)
        acc_dest = [[0.0] * 8 for _ in range(8)]
        acc_src  = [[0.0] * 8 for _ in range(8)]

        cache = None
        for fen in fens:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            M = len(legal_moves)
            if M == 0:
                break  # terminal position

            elo_norm = elo_w if board.turn == chess.WHITE else elo_b
            graph = self.builder.fen_to_graph(fen).to(self.device)
            with torch.no_grad():
                value, q_scores, _, cache = self.model.forward_with_q(
                    graph, cache=cache, elo_norm=elo_norm, return_cache=True
                )

            eval_traj.append(round(float(value.item()), 6))

            probs = torch.softmax(q_scores.detach(), dim=0).cpu()
            prob_list: list[float] = probs.tolist()

            # Normalised Shannon entropy → uncertainty
            log_m = math.log(M) if M > 1 else 1.0
            entropy = float(-torch.sum(probs * torch.log(probs.clamp(1e-9))).item())
            unc_traj.append(round(entropy / log_m, 6))

            legal_traj.append(M)
            piece_traj.append(len(board.piece_map()))

            # Q distribution sorted best-first (for move attribution)
            q_dists.append(
                sorted(
                    [(m.uci(), p) for m, p in zip(legal_moves, prob_list)],
                    key=lambda x: -x[1],
                )
            )

            # Centre pressure: Q-prob mass on {d4, d5, e4, e5}
            center_press = sum(
                p for m, p in zip(legal_moves, prob_list)
                if m.to_square in _CENTER_SQUARES
            )
            center_traj.append(round(center_press, 6))

            # Gini coefficient of Q-softmax (0 = uniform, 1 = one dominant move)
            s_p = sorted(prob_list)
            if M > 1:
                cum = sum((k + 1) * v for k, v in enumerate(s_p))
                gini = (2 * cum) / (M * sum(s_p)) - (M + 1) / M
            else:
                gini = 0.0
            gini_traj.append(round(gini, 6))

            # Accumulate spatial heatmaps
            for move, p in zip(legal_moves, prob_list):
                dr = chess.square_rank(move.to_square)
                df = chess.square_file(move.to_square)
                sr = chess.square_rank(move.from_square)
                sf = chess.square_file(move.from_square)
                acc_dest[dr][df] += p
                acc_src[sr][sf]  += p

        n = len(eval_traj)

        # Normalise heatmaps to [0, 1]
        max_d = max(max(row) for row in acc_dest) or 1.0
        max_s = max(max(row) for row in acc_src)  or 1.0
        norm_dest = [[round(v / max_d, 6) for v in row] for row in acc_dest]
        norm_src  = [[round(v / max_s, 6) for v in row] for row in acc_src]

        # ------------------------------------------------------------------
        # Move attribution
        # ------------------------------------------------------------------
        move_ranks:       list[int | None]   = []
        move_percentiles: list[float | None] = []
        eval_drops:       list[float | None] = []
        white_moves: list[dict] = []
        black_moves: list[dict] = []

        if moves_uci is not None:
            for i, uci in enumerate(moves_uci):
                if i >= n:
                    break
                board = chess.Board(fens[i])
                is_white = board.turn == chess.WHITE

                # Rank of played move in Q distribution (1 = best)
                q_dist = q_dists[i]
                rank = next((r + 1 for r, (u, _) in enumerate(q_dist) if u == uci), None)
                M_i = len(q_dist)
                pct = round((M_i - rank) / (M_i - 1), 6) if (rank is not None and M_i > 1) else None

                # Eval drop from the mover's perspective
                # White: drop = how much white's win prob fell after this move
                # Black: drop = how much black's win prob fell (= white's prob rising)
                if i + 1 < n:
                    delta = eval_traj[i + 1] - eval_traj[i]
                    drop = round(-delta if is_white else delta, 6)
                else:
                    drop = None

                move_ranks.append(rank)
                move_percentiles.append(pct)
                eval_drops.append(drop)

                entry = {"move_no": i + 1, "uci": uci,
                         "rank": rank, "percentile": pct, "eval_drop": drop}
                (white_moves if is_white else black_moves).append(entry)

        # ------------------------------------------------------------------
        # Per-side aggregations
        # ------------------------------------------------------------------
        def _side_stats(side_moves: list[dict], unc_list: list[float]) -> dict:
            nm = len(side_moves)
            if nm == 0:
                return {"moves_played": 0}
            ranks = [m["rank"] for m in side_moves if m["rank"] is not None]
            pcts  = [m["percentile"] for m in side_moves if m["percentile"] is not None]
            blunders  = [m for m in side_moves
                         if m["eval_drop"] is not None and m["eval_drop"] > _BLUNDER_THRESHOLD]
            mistakes  = [m for m in side_moves
                         if m["eval_drop"] is not None
                         and _MISTAKE_THRESHOLD < m["eval_drop"] <= _BLUNDER_THRESHOLD]
            best_moves = [m for m in side_moves
                          if m["eval_drop"] is not None and m["eval_drop"] < -_MISTAKE_THRESHOLD]
            return {
                "moves_played":          nm,
                "avg_move_rank":         round(sum(ranks) / len(ranks), 3) if ranks else None,
                "avg_move_percentile":   round(sum(pcts) / len(pcts), 4) if pcts else None,
                "agreement_top1":        round(sum(1 for r in ranks if r == 1) / len(ranks), 4) if ranks else None,
                "agreement_top3":        round(sum(1 for r in ranks if r <= 3) / len(ranks), 4) if ranks else None,
                "blunders":              blunders,
                "mistakes":              mistakes,
                "best_moves":            best_moves,
                "avg_uncertainty_faced": round(sum(unc_list) / len(unc_list), 4) if unc_list else None,
            }

        white_unc = [unc_traj[i] for i in range(n)
                     if chess.Board(fens[i]).turn == chess.WHITE]
        black_unc = [unc_traj[i] for i in range(n)
                     if chess.Board(fens[i]).turn == chess.BLACK]

        # ------------------------------------------------------------------
        # Game-level topology metrics
        # ------------------------------------------------------------------
        avg_complexity  = round(sum(unc_traj) / n, 4) if n else 0.0
        avg_piece_count = round(sum(piece_traj) / n, 2) if n else 0.0
        avg_branching   = round(sum(legal_traj) / n, 2) if n else 0.0
        game_sharpness  = round(float(torch.tensor(eval_traj).std().item()), 4) if n > 1 else 0.0

        # Decisiveness: fraction of adjacent eval-pairs that share a direction
        # (no sign flip = perfectly decisive = 1.0)
        decisiveness = 1.0
        if n >= 3:
            diffs = [eval_traj[i + 1] - eval_traj[i] for i in range(n - 1)]
            sign_flips = sum(
                1 for i in range(len(diffs) - 1)
                if diffs[i] * diffs[i + 1] < 0
            )
            decisiveness = round(1.0 - sign_flips / max(len(diffs) - 1, 1), 4)

        # Territory: share of accumulated dest probability by board half
        total_dest_mass = sum(acc_dest[r][f] for r in range(8) for f in range(8)) or 1.0
        white_territory = round(
            sum(acc_dest[r][f] for r in range(4, 8) for f in range(8)) / total_dest_mass, 4
        )
        black_territory = round(
            sum(acc_dest[r][f] for r in range(0, 4) for f in range(8)) / total_dest_mass, 4
        )

        return {
            "n_positions": n,
            "n_moves":     len(moves_uci) if moves_uci is not None else max(n - 1, 0),

            # Per-position trajectories
            "eval_trajectory":             eval_traj,
            "uncertainty_trajectory":      unc_traj,
            "legal_moves_trajectory":      legal_traj,
            "piece_count_trajectory":      piece_traj,
            "center_pressure_trajectory":  center_traj,
            "q_gini_trajectory":           gini_traj,

            # Move attribution (None when moves_uci not provided)
            "move_ranks":       move_ranks or None,
            "move_percentiles": move_percentiles or None,
            "eval_drops":       eval_drops or None,

            # Per-side
            "white": _side_stats(white_moves, white_unc) if moves_uci else None,
            "black": _side_stats(black_moves, black_unc) if moves_uci else None,

            # Game-level
            "avg_complexity":   avg_complexity,
            "decisiveness":     decisiveness,
            "avg_piece_count":  avg_piece_count,
            "avg_branching":    avg_branching,
            "game_sharpness":   game_sharpness,

            # Spatial / topological
            "accumulated_dest_heatmap": norm_dest,
            "accumulated_src_heatmap":  norm_src,
            "white_territory":   white_territory,
            "black_territory":   black_territory,
        }

    def estimate_elo(self, game_stats: dict, side: str = "white") -> dict:
        """Heuristic ELO estimate from :meth:`analyse_game` output.

        Uses a two-factor formula calibrated against published relationships
        between move agreement with strong engines and player ratings
        (Guid & Bratko 2006; lichess accuracy studies).  Requires
        ``moves_uci`` to have been passed to ``analyse_game``.

        Confidence interval is ±300 — this is single-game estimation and
        inherently noisy.  Aggregate over multiple games for tighter bounds.

        Parameters
        ----------
        game_stats : dict
            Output of :meth:`analyse_game`.
        side : str
            ``"white"`` or ``"black"``.

        Returns
        -------
        dict with keys:

        * ``estimated_elo``    – rounded ELO estimate
        * ``confidence_range`` – ``(low, high)`` tuple, ±300
        * ``features``         – input features used in the formula
        * ``note``             – disclaimer string

        Formula
        -------
        Two signals are blended:

        1. **Move percentile** — fraction of legal moves the played move
           beats, averaged over the game.  Calibration anchors:
           pct 0.75 ↔ 1200 Elo, 0.82 ↔ 1600, 0.88 ↔ 2000, 0.93 ↔ 2400.
        2. **Top-1 agreement rate** — fraction of moves that matched the
           model's first choice.  Calibration: 25 % ↔ 1400, 50 % ↔ 2400.

        Penalties: −400 per blunder-rate point, −150 per mistake-rate point.
        """
        s = game_stats.get(side)
        if s is None or s.get("avg_move_percentile") is None:
            return {
                "estimated_elo": None,
                "reason": "No move attribution data — pass moves_uci to analyse_game.",
            }

        pct          = s["avg_move_percentile"]
        top1_rate    = s["agreement_top1"] or 0.0
        top3_rate    = s["agreement_top3"] or 0.0
        nm           = s["moves_played"]
        blunder_rate = len(s["blunders"]) / max(nm, 1)
        mistake_rate = len(s["mistakes"]) / max(nm, 1)

        # Signal 1: move percentile → ELO (linear fit through calibration anchors)
        elo_pct  = (pct - 0.75) / 0.18 * 1200.0 + 1200.0
        # Signal 2: top-1 agreement → ELO
        elo_top1 = (top1_rate - 0.25) / 0.25 * 1000.0 + 1400.0

        # Blend: percentile is more stable over a full game
        estimated  = 0.65 * elo_pct + 0.35 * elo_top1
        estimated -= blunder_rate * 400.0
        estimated -= mistake_rate * 150.0
        estimated  = float(max(400, min(3200, estimated)))

        return {
            "estimated_elo":   round(estimated),
            "confidence_range": (round(estimated - 300), round(estimated + 300)),
            "features": {
                "avg_move_percentile": round(pct, 4),
                "agreement_top1":      round(top1_rate, 4),
                "agreement_top3":      round(top3_rate, 4),
                "blunder_rate":        round(blunder_rate, 4),
                "mistake_rate":        round(mistake_rate, 4),
                "n_moves":             nm,
            },
            "note": (
                "Heuristic estimate calibrated against engine-agreement research. "
                "Single-game accuracy ≈ ±300 Elo. Aggregate over more games for tighter bounds."
            ),
        }


