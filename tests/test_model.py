import pytest
import torch
import chess

from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel

STARTING_FEN = chess.STARTING_FEN
# Position with promotion available (white pawn on e7, kings far apart)
PROMOTION_FEN = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"
# Position where castling is available
CASTLING_FEN = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"


@pytest.fixture(scope="module")
def builder_full():
    return ChessGraphBuilder(use_global_node=True, use_move_edges=True)


@pytest.fixture(scope="module")
def builder_moves_only():
    return ChessGraphBuilder(use_move_edges=True)


@pytest.fixture(scope="module")
def model_full(builder_full):
    metadata = builder_full.get_metadata()
    m = GATEAUChessModel(metadata, hidden_channels=64, num_layers=2)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_no_global(builder_moves_only):
    metadata = builder_moves_only.get_metadata()
    m = GATEAUChessModel(metadata, hidden_channels=64, num_layers=2)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Value head: forward()
# ---------------------------------------------------------------------------

class TestValueHead:
    def test_value_shape(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            value = model_full(graph)
        assert value.shape == (1, 1)

    def test_value_in_tanh_range(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            value = model_full(graph)
        assert -1.0 <= value.item() <= 1.0

    def test_value_no_global_node(self, model_no_global, builder_moves_only):
        graph = builder_moves_only.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            value = model_no_global(graph)
        assert value.shape == (1, 1)
        assert -1.0 <= value.item() <= 1.0

    def test_value_scalar_is_finite(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(CASTLING_FEN)
        with torch.no_grad():
            value = model_full(graph)
        assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# Q-head: forward_with_q()
# ---------------------------------------------------------------------------

class TestQHead:
    def test_output_tuple_length(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            out = model_full.forward_with_q(graph)
        assert len(out) == 3

    def test_value_shape_from_q_path(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            value, _, _ = model_full.forward_with_q(graph)
        assert value.shape == (1, 1)
        assert -1.0 <= value.item() <= 1.0

    def test_q_scores_count_matches_legal_moves(self, model_full, builder_full):
        board = chess.Board(STARTING_FEN)
        num_moves = len(list(board.legal_moves))  # 20
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, q_scores, _ = model_full.forward_with_q(graph)
        assert q_scores.shape == (num_moves,)

    def test_move_edge_index_shape(self, model_full, builder_full):
        board = chess.Board(STARTING_FEN)
        num_moves = len(list(board.legal_moves))
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, _, move_edge_index = model_full.forward_with_q(graph)
        assert move_edge_index.shape == (2, num_moves)

    def test_q_scores_are_finite(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, q_scores, _ = model_full.forward_with_q(graph)
        assert torch.isfinite(q_scores).all()

    def test_q_scores_not_all_identical(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, q_scores, _ = model_full.forward_with_q(graph)
        assert q_scores.std().item() > 0.0

    def test_q_head_no_global_node(self, model_no_global, builder_moves_only):
        board = chess.Board(STARTING_FEN)
        num_moves = len(list(board.legal_moves))
        graph = builder_moves_only.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            value, q_scores, move_edge_index = model_no_global.forward_with_q(graph)
        assert value.shape == (1, 1)
        assert q_scores.shape == (num_moves,)
        assert move_edge_index.shape == (2, num_moves)

    def test_q_scores_promotion_position(self, model_full, builder_full):
        board = chess.Board(PROMOTION_FEN)
        num_moves = len(list(board.legal_moves))
        graph = builder_full.fen_to_graph(PROMOTION_FEN)
        with torch.no_grad():
            _, q_scores, _ = model_full.forward_with_q(graph)
        assert q_scores.shape == (num_moves,)

    def test_q_scores_castling_position(self, model_full, builder_full):
        board = chess.Board(CASTLING_FEN)
        num_moves = len(list(board.legal_moves))
        graph = builder_full.fen_to_graph(CASTLING_FEN)
        with torch.no_grad():
            _, q_scores, _ = model_full.forward_with_q(graph)
        assert q_scores.shape == (num_moves,)


# ---------------------------------------------------------------------------
# API / attribute checks
# ---------------------------------------------------------------------------

class TestModelAPI:
    def test_has_forward_with_q(self, model_full):
        assert hasattr(model_full, 'forward_with_q')

    def test_has_forward(self, model_full):
        assert hasattr(model_full, 'forward')

    def test_has_forward_step(self, model_full):
        assert hasattr(model_full, 'forward_step')

    def test_has_forward_sequence_with_q(self, model_full):
        assert hasattr(model_full, 'forward_sequence_with_q')

    def test_parameter_count_nonzero(self, model_full):
        total = sum(p.numel() for p in model_full.parameters())
        assert total > 0


# ---------------------------------------------------------------------------
# ELO conditioning
# ---------------------------------------------------------------------------

class TestELOConditioning:
    """ELO embedding changes value output but not Q-score ranking."""

    def test_elo_changes_value(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            v_sf, _, _ = model_full.forward_with_q(graph, elo_norm=1.0)
            v_low, _, _ = model_full.forward_with_q(graph, elo_norm=0.3)
        # Different ELO → different value prediction
        assert v_sf.item() != v_low.item()

    def test_elo_default_is_one(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            v_default, _, _ = model_full.forward_with_q(graph)
            v_explicit, _, _ = model_full.forward_with_q(graph, elo_norm=1.0)
        assert v_default.item() == pytest.approx(v_explicit.item(), abs=1e-6)

    def test_elo_value_in_range(self, model_full, builder_full):
        graph = builder_full.fen_to_graph(STARTING_FEN)
        for elo in [0.0, 0.3, 0.5, 0.8, 1.0]:
            with torch.no_grad():
                value, _, _ = model_full.forward_with_q(graph, elo_norm=elo)
            assert -1.0 <= value.item() <= 1.0, f"elo_norm={elo} produced value out of [-1,1]"


# ---------------------------------------------------------------------------
# forward_sequence_with_q
# ---------------------------------------------------------------------------

class TestForwardSequenceWithQ:
    """Verify sequence method chains GRU and returns correct shapes."""

    def _make_sequence(self, builder_full, fens):
        return [builder_full.fen_to_graph(f) for f in fens]

    def test_values_shape(self, model_full, builder_full):
        fens = [STARTING_FEN, CASTLING_FEN]
        graphs = self._make_sequence(builder_full, fens)
        with torch.no_grad():
            values, _, _ = model_full.forward_sequence_with_q(graphs)
        assert values.shape == (2, 1)

    def test_q_scores_list_length(self, model_full, builder_full):
        fens = [STARTING_FEN, CASTLING_FEN]
        graphs = self._make_sequence(builder_full, fens)
        with torch.no_grad():
            _, q_list, _ = model_full.forward_sequence_with_q(graphs)
        assert len(q_list) == 2

    def test_q_scores_count_matches_legal_moves(self, model_full, builder_full):
        fens = [STARTING_FEN, CASTLING_FEN]
        graphs = self._make_sequence(builder_full, fens)
        with torch.no_grad():
            _, q_list, _ = model_full.forward_sequence_with_q(graphs)
        for fen, q in zip(fens, q_list):
            expected = len(list(chess.Board(fen).legal_moves))
            assert q.shape == (expected,), f"q shape mismatch for {fen}"

    def test_values_finite(self, model_full, builder_full):
        fens = [STARTING_FEN, CASTLING_FEN]
        graphs = self._make_sequence(builder_full, fens)
        with torch.no_grad():
            values, _, _ = model_full.forward_sequence_with_q(graphs)
        assert torch.isfinite(values).all()

    def test_sequence_differs_with_elo(self, model_full, builder_full):
        """Values from different ELO norms should differ."""
        fens = [STARTING_FEN, CASTLING_FEN]
        graphs = self._make_sequence(builder_full, fens)
        with torch.no_grad():
            v_sf, _, _ = model_full.forward_sequence_with_q(graphs, elo_norm=1.0)
            v_low, _, _ = model_full.forward_sequence_with_q(graphs, elo_norm=0.4)
        assert not torch.allclose(v_sf, v_low)

    def test_single_position_sequence(self, model_full, builder_full):
        graphs = [builder_full.fen_to_graph(STARTING_FEN)]
        with torch.no_grad():
            values, q_list, edge_list = model_full.forward_sequence_with_q(graphs)
        assert values.shape == (1, 1)
        assert len(q_list) == 1
        assert len(edge_list) == 1
