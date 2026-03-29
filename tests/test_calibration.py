"""Tests for chessgnn/calibration.py — TemperatureScaler, reliability_diagram,
and the CaseTutor uncertainty field."""

import json
import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from chessgnn.calibration import TemperatureScaler, reliability_diagram


def _make_overconfident_data(n: int = 500, seed: int = 42):
    """Create synthetic overconfident predictions and soft targets.

    True probabilities are drawn from Beta(2,2) (peaked around 0.5).
    Predictions amplify the logit by 2x, making them too extreme (overconfident).
    Temperature scaling with T>1 should bring them closer to the targets.
    """
    rng = np.random.default_rng(seed)
    true_p = rng.beta(2, 2, size=n)
    eps = 1e-7
    true_p = np.clip(true_p, eps, 1 - eps)
    logits = np.log(true_p / (1.0 - true_p))
    # Amplify logit to simulate overconfidence
    overconfident_logits = logits * 2.5
    preds = 1.0 / (1.0 + np.exp(-overconfident_logits))
    return preds, true_p


def _preds_to_logits(preds: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(preds, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class TestTemperatureScaler(unittest.TestCase):

    def test_perfect_calibration_leaves_t_near_one(self):
        """On perfectly calibrated data T should converge close to 1.0."""
        rng = np.random.default_rng(0)
        n = 300
        targets = rng.uniform(0.1, 0.9, size=n)
        # Generate predictions identical to targets (perfect cal) + tiny noise
        preds = np.clip(targets + rng.normal(0, 0.01, size=n), 1e-6, 1 - 1e-6)
        logits = _preds_to_logits(preds)

        scaler = TemperatureScaler()
        scaler.fit(logits, targets)

        self.assertAlmostEqual(scaler.T, 1.0, delta=0.15)

    def test_temperature_reduces_ece(self):
        """Fitting T on overconfident data should strictly reduce ECE."""
        preds, targets = _make_overconfident_data(n=600)
        logits = _preds_to_logits(preds)

        scaler = TemperatureScaler()
        ece_before = scaler.ece(preds, targets)
        scaler.fit(logits, targets)
        calibrated = np.array([scaler.calibrate(p) for p in preds])
        ece_after = scaler.ece(calibrated, targets)

        self.assertGreater(ece_before, ece_after)
        # T should be > 1 to soften overconfident predictions
        self.assertGreater(scaler.T, 1.0)

    def test_calibrate_edge_inputs_do_not_raise(self):
        """calibrate() should handle boundary probabilities gracefully."""
        scaler = TemperatureScaler(T=1.5)
        low = scaler.calibrate(0.0)
        high = scaler.calibrate(1.0)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(low, 1.0)
        self.assertGreaterEqual(high, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_calibrate_monotonic(self):
        """calibrate() must be monotonically increasing in its input."""
        scaler = TemperatureScaler(T=2.0)
        probs = np.linspace(0.01, 0.99, 50)
        calibrated = [scaler.calibrate(p) for p in probs]
        for a, b in zip(calibrated[:-1], calibrated[1:]):
            self.assertLessEqual(a, b)

    def test_save_load_roundtrip(self):
        """save() then load() should reproduce the fitted temperature exactly."""
        preds, targets = _make_overconfident_data(n=400)
        logits = _preds_to_logits(preds)

        scaler = TemperatureScaler()
        scaler.fit(logits, targets)
        original_t = scaler.T

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            scaler.save(tmp_path)
            loaded = TemperatureScaler()
            loaded.load(tmp_path)
            self.assertAlmostEqual(loaded.T, original_t, places=6)
        finally:
            os.unlink(tmp_path)

    def test_ece_perfect_model_is_zero(self):
        """ECE of a model whose predictions equal the targets should be ~0."""
        targets = np.linspace(0.1, 0.9, 100)
        ece = TemperatureScaler().ece(targets, targets)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_ece_range(self):
        """ECE must lie in [0, 1]."""
        preds, targets = _make_overconfident_data(n=200)
        ece = TemperatureScaler().ece(preds, targets)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


class TestReliabilityDiagram(unittest.TestCase):

    def test_returns_figure(self):
        """reliability_diagram() should return a matplotlib Figure."""
        import matplotlib.figure

        preds = np.linspace(0.05, 0.95, 50)
        targets = preds + np.random.default_rng(1).normal(0, 0.05, 50)
        targets = np.clip(targets, 0.0, 1.0)
        fig = reliability_diagram(preds, targets, n_bins=5)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_empty_arrays_do_not_raise(self):
        """An empty array should produce a Figure with no data points."""
        import matplotlib.figure

        fig = reliability_diagram(np.array([]), np.array([]))
        self.assertIsInstance(fig, matplotlib.figure.Figure)


class TestCaseTutorUncertainty(unittest.TestCase):

    def _make_mock_gateau(self, q_logits: list[float]):
        """Build a minimal mock model that has forward_with_q."""
        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=model)
        # forward_with_q returns (value, q_scores, move_edge_index)
        value = torch.tensor([[0.1]])
        qs = torch.tensor(q_logits, dtype=torch.float32)
        model.forward_with_q = MagicMock(return_value=(value, qs, None))
        # hasattr check for the Q-head detection
        model.__class__ = type(
            "GATEAUChessModel",
            (),
            {"forward_with_q": model.forward_with_q},
        )
        return model

    def test_uncertainty_in_range(self):
        """Entropy-based uncertainty should be in [0, 1] for any Q distribution."""
        from tutor import CaseTutor

        # 3 legal moves from starting position — mock returns 3 Q scores
        model = self._make_mock_gateau([1.0, 0.0, -1.0])

        # Patch fen_to_graph so no real graph is built
        mock_graph = MagicMock()
        mock_graph.to = MagicMock(return_value=mock_graph)

        tutor = CaseTutor.__new__(CaseTutor)
        tutor.model = model
        tutor.device = torch.device("cpu")
        tutor._use_q_head = True
        tutor._scaler = None
        tutor.current_hidden = None

        from chessgnn.graph_builder import ChessGraphBuilder
        builder = MagicMock(spec=ChessGraphBuilder)
        builder.fen_to_graph = MagicMock(return_value=mock_graph)
        tutor.builder = builder

        # Use a FEN with exactly 3 determined legal moves isn't trivial, so we
        # call _recommend_q directly with a board that has 3 legal moves.
        # Build a board with limited moves: stalemate-adjacent position
        import chess

        board = chess.Board("8/8/8/8/8/8/6P1/6K1 w - - 0 1")  # white has P+K
        legal_moves = list(board.legal_moves)
        n = len(legal_moves)

        # Re-mock with the right number of Q scores
        qs = torch.tensor([float(i) for i in range(n)], dtype=torch.float32)
        model.forward_with_q = MagicMock(return_value=(torch.tensor([[0.1]]), qs, None))

        fen = board.fen()
        best_move, best_prob, ranking, uncertainty = tutor._recommend_q(board, legal_moves, fen)

        self.assertGreaterEqual(uncertainty, 0.0)
        self.assertLessEqual(uncertainty, 1.0)

    def test_uncertainty_uniform_is_one(self):
        """When all Q scores are equal, entropy is maximal → uncertainty ≈ 1.0."""
        import math

        from tutor import CaseTutor
        import chess

        n = 5
        model = self._make_mock_gateau([0.0] * n)
        mock_graph = MagicMock()
        mock_graph.to = MagicMock(return_value=mock_graph)

        tutor = CaseTutor.__new__(CaseTutor)
        tutor.model = model
        tutor.device = torch.device("cpu")
        tutor._use_q_head = True
        tutor._scaler = None
        tutor.current_hidden = None

        from chessgnn.graph_builder import ChessGraphBuilder
        builder = MagicMock(spec=ChessGraphBuilder)
        builder.fen_to_graph = MagicMock(return_value=mock_graph)
        tutor.builder = builder

        board = chess.Board("8/8/8/8/8/8/6P1/6K1 w - - 0 1")
        legal_moves = list(board.legal_moves)
        n_moves = len(legal_moves)

        qs = torch.zeros(n_moves)
        model.forward_with_q = MagicMock(return_value=(torch.tensor([[0.0]]), qs, None))

        _, _, _, uncertainty = tutor._recommend_q(board, legal_moves, board.fen())
        self.assertAlmostEqual(uncertainty, 1.0, places=5)

    def test_rollout_path_uncertainty_is_zero(self):
        """The rollout path always returns uncertainty=0.0."""
        from tutor import CaseTutor
        import chess

        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=model)
        # Simulate rollout model (has forward_step, no forward_with_q)
        step_out = torch.tensor([[0.1]])
        hidden = torch.zeros(1, 1, 8)
        model.forward_step = MagicMock(return_value=(step_out, hidden))
        mock_graph = MagicMock()
        mock_graph.to = MagicMock(return_value=mock_graph)

        tutor = CaseTutor.__new__(CaseTutor)
        tutor.model = model
        tutor.device = torch.device("cpu")
        tutor._use_q_head = False
        tutor._scaler = None
        tutor.current_hidden = None

        from chessgnn.graph_builder import ChessGraphBuilder
        builder = MagicMock(spec=ChessGraphBuilder)
        builder.fen_to_graph = MagicMock(return_value=mock_graph)
        tutor.builder = builder

        board = chess.Board("8/8/8/8/8/8/6P1/6K1 w - - 0 1")
        legal_moves = list(board.legal_moves)

        _, _, _, uncertainty = tutor._recommend_rollout(board, legal_moves)
        self.assertEqual(uncertainty, 0.0)


class TestCaseTutorCalibration(unittest.TestCase):

    def test_set_calibration_affects_best_prob(self):
        """Attaching a TemperatureScaler with T>1 should soften the returned prob."""
        from tutor import CaseTutor
        import chess

        # Build a minimal tutor with mock Q-head model
        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=model)

        board = chess.Board("8/8/8/8/8/8/6P1/6K1 w - - 0 1")
        legal_moves = list(board.legal_moves)
        n = len(legal_moves)

        # High Q scores → confident prediction near 100 %
        qs = torch.tensor([10.0] + [0.0] * (n - 1))
        model.forward_with_q = MagicMock(return_value=(torch.tensor([[0.9]]), qs, None))

        mock_graph = MagicMock()
        mock_graph.to = MagicMock(return_value=mock_graph)

        tutor = CaseTutor.__new__(CaseTutor)
        tutor.model = model
        tutor.device = torch.device("cpu")
        tutor._use_q_head = True
        tutor._scaler = None
        tutor.current_hidden = None

        from chessgnn.graph_builder import ChessGraphBuilder
        builder = MagicMock(spec=ChessGraphBuilder)
        builder.fen_to_graph = MagicMock(return_value=mock_graph)
        tutor.builder = builder

        fen = board.fen()
        _, raw_prob, _, _ = tutor._recommend_q(board, legal_moves, fen)

        # Now attach a T=3 scaler (softens)
        scaler = TemperatureScaler(T=3.0)
        tutor.set_calibration(scaler)

        model.forward_with_q = MagicMock(return_value=(torch.tensor([[0.9]]), qs, None))
        _, cal_prob, _, _ = tutor._recommend_q(board, legal_moves, fen)

        # Calibrated probability should be closer to 50 % than the raw one
        self.assertLess(abs(cal_prob - 50.0), abs(raw_prob - 50.0))


if __name__ == "__main__":
    unittest.main()
