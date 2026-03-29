import math
import torch
import chess
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.calibration import TemperatureScaler


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

    def recommend_move(self, fen: str):
        """
        Returns the best move for the current position.

        Uses a single forward pass via the Q-head when the model supports it
        (GATEAUChessModel), otherwise falls back to a per-successor rollout.

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
        """
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None, 0.0, [], 0.0

        if self._use_q_head:
            return self._recommend_q(board, legal_moves, fen)
        return self._recommend_rollout(board, legal_moves)

    # ------------------------------------------------------------------
    # Private inference paths
    # ------------------------------------------------------------------

    def _recommend_q(self, board: chess.Board, legal_moves: list, fen: str):
        """Single-pass move ranking via Q-head."""
        graph = self.builder.fen_to_graph(fen).to(self.device)
        with torch.no_grad():
            _, q_scores, _ = self.model.forward_with_q(graph)

        q_list = q_scores.cpu().tolist()
        if len(q_list) != len(legal_moves):
            # Safety fallback (should not normally occur)
            return self._recommend_rollout(board, legal_moves)

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

        return best_move, best_prob, move_scores, uncertainty

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
