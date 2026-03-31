import pytest
import chess
import torch
from chessgnn.graph_builder import ChessGraphBuilder

STARTING_FEN = chess.STARTING_FEN
# A mid-game position with captures and en-passant available:
# After 1.e4 d5 2.e5 f5 — en passant on f6 is available, castling rights intact
MIDGAME_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"


class TestChessGraphBuilderBase:
    """Tests for the default (no-flag) builder."""

    def setup_method(self):
        self.builder = ChessGraphBuilder()

    def test_piece_node_count_starting(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        # 32 pieces on the starting board
        assert data['piece'].x.shape == (32, 10)

    def test_square_node_count(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        assert data['square'].x.shape == (64, 3)

    def test_piece_feature_range(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        x = data['piece'].x
        # file and rank are normalized to [0, 1]
        assert x[:, 8].min() >= 0.0 and x[:, 8].max() <= 1.0  # file
        assert x[:, 9].min() >= 0.0 and x[:, 9].max() <= 1.0  # rank
        # color is +1 or -1
        assert set(x[:, 6].tolist()).issubset({1.0, -1.0})

    def test_fen_attribute_set(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        assert data.fen == STARTING_FEN

    def test_location_edges_shape(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['piece', 'on', 'square'].edge_index
        # One edge per piece
        assert ei.shape == (2, 32)

    def test_reverse_location_edges(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['square', 'occupied_by', 'piece'].edge_index
        assert ei.shape == (2, 32)

    def test_adjacent_edges_present(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['square', 'adjacent', 'square'].edge_index
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0

    def test_interacts_edges_attr_dims(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        attr = data['piece', 'interacts', 'piece'].edge_attr
        assert attr.shape[1] == 2  # [weight, edge_type]

    def test_ray_edges_attr_dims(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        attr = data['piece', 'ray', 'piece'].edge_attr
        assert attr.shape[1] == 2  # [dist/7, blocking]

    def test_no_global_node_by_default(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        assert 'global' not in data.x_dict

    def test_no_move_edges_by_default(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        assert ('piece', 'move', 'square') not in data.edge_index_dict


class TestGetMetadataBase:
    def test_base_metadata_node_types(self):
        builder = ChessGraphBuilder()
        node_types, _ = builder.get_metadata()
        assert set(node_types) == {'piece', 'square'}

    def test_base_metadata_edge_types(self):
        builder = ChessGraphBuilder()
        _, edge_types = builder.get_metadata()
        expected = {
            ('piece', 'on', 'square'),
            ('square', 'occupied_by', 'piece'),
            ('square', 'adjacent', 'square'),
            ('piece', 'interacts', 'piece'),
            ('piece', 'ray', 'piece'),
        }
        assert expected.issubset(set(edge_types))
        assert ('piece', 'move', 'square') not in edge_types
        assert ('global', 'global_to_piece', 'piece') not in edge_types

    def test_metadata_consistent_with_graph(self):
        builder = ChessGraphBuilder()
        data = builder.fen_to_graph(STARTING_FEN)
        node_types, edge_types = builder.get_metadata()
        for nt in node_types:
            assert nt in data.x_dict
        for et in edge_types:
            assert et in data.edge_index_dict


class TestGlobalNode:
    def setup_method(self):
        self.builder = ChessGraphBuilder(use_global_node=True)

    def test_global_node_shape(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        assert data['global'].x.shape == (1, 11)

    def test_global_node_feature_ranges(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        feat = data['global'].x[0]
        # side_to_move is +1 or -1
        assert feat[0].item() in {1.0, -1.0}
        # all remaining features are in [0, 1]
        for i in range(1, 11):
            assert 0.0 <= feat[i].item() <= 1.0, f"feature {i} out of range: {feat[i].item()}"

    def test_global_starting_features(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        feat = data['global'].x[0]
        assert feat[0].item() == 1.0   # white to move
        assert feat[1].item() == 1.0   # white kingside castling
        assert feat[2].item() == 1.0   # white queenside castling
        assert feat[3].item() == 1.0   # black kingside castling
        assert feat[4].item() == 1.0   # black queenside castling
        assert feat[5].item() == 0.0   # halfmove clock = 0
        # ELO features default to 1500/3000 = 0.5 when not provided
        assert abs(feat[9].item() - 0.5) < 1e-5   # white_elo_norm
        assert abs(feat[10].item() - 0.5) < 1e-5  # black_elo_norm

    def test_global_elo_injection(self):
        """ELO params are injected into global node features at indices 9 and 10."""
        data = self.builder.fen_to_graph(STARTING_FEN, white_elo=2400, black_elo=1200)
        feat = data['global'].x[0]
        assert abs(feat[9].item() - 2400 / 3000) < 1e-5
        assert abs(feat[10].item() - 1200 / 3000) < 1e-5

    def test_global_elo_clamp(self):
        """ELO values outside [0, 3000] are clamped."""
        data = self.builder.fen_to_graph(STARTING_FEN, white_elo=9999, black_elo=-100)
        feat = data['global'].x[0]
        assert feat[9].item() == 1.0   # clamped at 3000
        assert feat[10].item() == 0.0  # clamped at 0

    def test_global_to_piece_edges(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['global', 'global_to_piece', 'piece'].edge_index
        # One edge from global (index 0) to each of the 32 pieces
        assert ei.shape == (2, 32)
        assert (ei[0] == 0).all()  # all sources are global node 0

    def test_piece_to_global_edges(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['piece', 'piece_to_global', 'global'].edge_index
        assert ei.shape == (2, 32)
        assert (ei[1] == 0).all()  # all destinations are global node 0

    def test_global_to_square_edges(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['global', 'global_to_square', 'square'].edge_index
        assert ei.shape == (2, 64)
        assert (ei[0] == 0).all()

    def test_square_to_global_edges(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['square', 'square_to_global', 'global'].edge_index
        assert ei.shape == (2, 64)
        assert (ei[1] == 0).all()

    def test_global_metadata_node_types(self):
        node_types, edge_types = self.builder.get_metadata()
        assert 'global' in node_types

    def test_global_metadata_edge_types(self):
        _, edge_types = self.builder.get_metadata()
        expected_new = {
            ('global', 'global_to_piece', 'piece'),
            ('piece', 'piece_to_global', 'global'),
            ('global', 'global_to_square', 'square'),
            ('square', 'square_to_global', 'global'),
        }
        assert expected_new.issubset(set(edge_types))

    def test_metadata_consistent_with_graph(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        node_types, edge_types = self.builder.get_metadata()
        for nt in node_types:
            assert nt in data.x_dict
        for et in edge_types:
            assert et in data.edge_index_dict


class TestMoveEdges:
    def setup_method(self):
        self.builder = ChessGraphBuilder(use_move_edges=True)

    def test_move_edge_count_matches_legal_moves(self):
        board = chess.Board(STARTING_FEN)
        data = self.builder.fen_to_graph(STARTING_FEN)
        ei = data['piece', 'move', 'square'].edge_index
        assert ei.shape[1] == len(list(board.legal_moves))  # 20 from start

    def test_move_edge_attr_dims(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        attr = data['piece', 'move', 'square'].edge_attr
        assert attr.shape == (20, 5)

    def test_move_edge_attr_range(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        attr = data['piece', 'move', 'square'].edge_attr
        # All features should be in [0, 1]
        assert attr.min().item() >= 0.0
        assert attr.max().item() <= 1.0

    def test_no_captures_from_start(self):
        data = self.builder.fen_to_graph(STARTING_FEN)
        attr = data['piece', 'move', 'square'].edge_attr
        # Starting position has no captures
        assert (attr[:, 0] == 0.0).all()  # is_capture column

    def test_capture_flag_set(self):
        # After e4 d5, white can capture on d5
        board = chess.Board()
        board.push_san("e4")
        board.push_san("d5")
        fen = board.fen()
        data = self.builder.fen_to_graph(fen)
        attr = data['piece', 'move', 'square'].edge_attr
        # At least one capture edge should exist (exd5)
        assert (attr[:, 0] == 1.0).any()

    def test_promotion_flag_set(self):
        # White pawn on e7, black king far away on h8, white king on e1
        promotion_fen = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"
        data = self.builder.fen_to_graph(promotion_fen)
        attr = data['piece', 'move', 'square'].edge_attr
        # At least one promotion move must have is_promotion=1
        assert (attr[:, 2] == 1.0).any()

    def test_en_passant_capture_flag(self):
        data = self.builder.fen_to_graph(MIDGAME_FEN)
        attr = data['piece', 'move', 'square'].edge_attr
        # En passant move should have is_capture=1, captured_val = pawn/9
        ep_moves = attr[attr[:, 0] == 1.0]
        assert len(ep_moves) > 0

    def test_castling_flag_set(self):
        # Position where white can castle kingside
        castle_fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        data = self.builder.fen_to_graph(castle_fen)
        attr = data['piece', 'move', 'square'].edge_attr
        assert (attr[:, 3] == 1.0).any()  # at least one castling move

    def test_move_edge_not_in_metadata_by_default(self):
        builder = ChessGraphBuilder()
        _, edge_types = builder.get_metadata()
        assert ('piece', 'move', 'square') not in edge_types

    def test_move_edge_in_metadata_when_enabled(self):
        _, edge_types = self.builder.get_metadata()
        assert ('piece', 'move', 'square') in edge_types


class TestAllFlags:
    def test_both_flags_enabled(self):
        builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
        data = builder.fen_to_graph(STARTING_FEN)
        assert 'global' in data.x_dict
        assert ('piece', 'move', 'square') in data.edge_index_dict

    def test_both_flags_metadata_consistent(self):
        builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
        data = builder.fen_to_graph(STARTING_FEN)
        node_types, edge_types = builder.get_metadata()
        for nt in node_types:
            assert nt in data.x_dict, f"node type '{nt}' missing from graph"
        for et in edge_types:
            assert et in data.edge_index_dict, f"edge type {et} missing from graph"
