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

    def test_parameter_count_nonzero(self, model_full):
        total = sum(p.numel() for p in model_full.parameters())
        assert total > 0
