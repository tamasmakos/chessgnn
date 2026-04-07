"""Tests for the split Q-head architecture (engine head + human head).

Verifies:
- forward_with_q_dual returns the expected tuple sizes and tensor shapes
- engine head (q_head) and human head (human_q_head) have distinct parameters
- forward_sequence_with_q_dual returns a 5-tuple with separate engine/human lists
- Both heads produce valid finite scores
- human head uses separate ELO conditioning (outputs differ from engine head)
"""
import pytest
import torch
import chess

from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel

STARTING_FEN = chess.STARTING_FEN
MIDGAME_FEN   = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


@pytest.fixture(scope="module")
def builder():
    return ChessGraphBuilder(use_global_node=True, use_move_edges=True)


@pytest.fixture(scope="module")
def model(builder):
    meta = builder.get_metadata()
    m = GATEAUChessModel(meta, hidden_channels=64, num_layers=2)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# forward_with_q_dual — single-position dual-head inference
# ---------------------------------------------------------------------------

class TestForwardWithQDual:
    def test_model_has_dual_attribute(self, model):
        assert hasattr(model, "forward_with_q_dual"), (
            "GATEAUChessModel must expose forward_with_q_dual"
        )

    def test_model_has_human_q_head(self, model):
        assert hasattr(model, "human_q_head"), (
            "GATEAUChessModel must have a human_q_head attribute"
        )

    def test_base_tuple_length(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            out = model.forward_with_q_dual(graph)
        # (value, q_engine, q_human, move_edge_index)
        assert len(out) == 4

    def test_tuple_length_with_cache(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            out = model.forward_with_q_dual(graph, return_cache=True)
        # (value, q_engine, q_human, move_edge_index, cache)
        assert len(out) == 5

    def test_tuple_length_with_embeddings(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            out = model.forward_with_q_dual(graph, return_embeddings=True)
        # (value, q_engine, q_human, move_edge_index, x_dict)
        assert len(out) == 5

    def test_tuple_length_with_cache_and_embeddings(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            out = model.forward_with_q_dual(
                graph, return_cache=True, return_embeddings=True
            )
        # (value, q_engine, q_human, move_edge_index, cache, x_dict)
        assert len(out) == 6

    def test_value_shape(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            value, _, _, _ = model.forward_with_q_dual(graph)
        assert value.shape == (1, 1)

    def test_value_in_tanh_range(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            value, _, _, _ = model.forward_with_q_dual(graph)
        assert -1.0 <= value.item() <= 1.0

    def test_q_engine_shape_matches_legal_moves(self, model, builder):
        board = chess.Board(STARTING_FEN)
        n_legal = len(list(board.legal_moves))
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, q_engine, _, _ = model.forward_with_q_dual(graph)
        assert q_engine.shape == (n_legal,)

    def test_q_human_shape_matches_legal_moves(self, model, builder):
        board = chess.Board(STARTING_FEN)
        n_legal = len(list(board.legal_moves))
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, _, q_human, _ = model.forward_with_q_dual(graph)
        assert q_human.shape == (n_legal,)

    def test_q_engine_finite(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, q_engine, _, _ = model.forward_with_q_dual(graph)
        assert torch.isfinite(q_engine).all()

    def test_q_human_finite(self, model, builder):
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, _, q_human, _ = model.forward_with_q_dual(graph)
        assert torch.isfinite(q_human).all()

    def test_move_edge_index_shape(self, model, builder):
        board = chess.Board(STARTING_FEN)
        n_legal = len(list(board.legal_moves))
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, _, _, move_edge_index = model.forward_with_q_dual(graph)
        assert move_edge_index.shape == (2, n_legal)

    def test_midgame_position(self, model, builder):
        board = chess.Board(MIDGAME_FEN)
        n_legal = len(list(board.legal_moves))
        graph = builder.fen_to_graph(MIDGAME_FEN)
        with torch.no_grad():
            value, q_engine, q_human, _ = model.forward_with_q_dual(graph)
        assert q_engine.shape == (n_legal,)
        assert q_human.shape == (n_legal,)
        assert torch.isfinite(q_engine).all()
        assert torch.isfinite(q_human).all()


# ---------------------------------------------------------------------------
# Independent parameters: engine head vs human head
# ---------------------------------------------------------------------------

class TestHeadIndependence:
    def test_q_head_and_human_q_head_are_distinct_modules(self, model):
        """The two heads must not share the same Python object."""
        assert model.q_head is not model.human_q_head

    def test_q_head_and_human_q_head_have_different_params(self, model):
        """Engine and human heads must have separate parameter tensors."""
        eng_params  = list(model.q_head.parameters())
        hum_params  = list(model.human_q_head.parameters())
        # Same number of parameter tensors (symmetric architecture)
        assert len(eng_params) == len(hum_params)
        # None of the engine tensors should be the *same object* as a human tensor
        for ep, hp in zip(eng_params, hum_params):
            assert ep.data_ptr() != hp.data_ptr(), (
                "Engine and human Q-head must not share weight storage"
            )

    def test_elo_conditioning_affects_outputs(self, model, builder):
        """When ELO conditioning differs, engine and human outputs should differ."""
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            # Engine always uses elo_norm_sf=1.0; use a low player ELO
            _, q_engine, q_human, _ = model.forward_with_q_dual(
                graph, elo_norm_sf=1.0, elo_norm_player=0.3
            )
        # With freshly initialised (random) weights, outputs should not be identical
        # (this would fail only in the pathological case of all-zero heads)
        # We test that the tensors are not byte-for-byte equal
        assert not torch.equal(q_engine, q_human)


# ---------------------------------------------------------------------------
# forward_sequence_with_q_dual — sequential (training) 5-tuple
# ---------------------------------------------------------------------------

class TestForwardSequenceWithQDual:
    def test_returns_5_tuple(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        with torch.no_grad():
            out = model.forward_sequence_with_q_dual(graphs)
        assert len(out) == 5, "Expected 5-tuple: (values_sf, values_player, q_engine_list, q_human_list, move_edge_indices)"

    def test_values_length(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        with torch.no_grad():
            values_sf, values_player, q_eng, q_hum, _ = model.forward_sequence_with_q_dual(graphs)
        assert len(values_sf) == 2
        assert len(values_player) == 2

    def test_q_engine_list_length(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        with torch.no_grad():
            _, _, q_eng, _, _ = model.forward_sequence_with_q_dual(graphs)
        assert len(q_eng) == 2

    def test_q_human_list_length(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        with torch.no_grad():
            _, _, _, q_hum, _ = model.forward_sequence_with_q_dual(graphs)
        assert len(q_hum) == 2

    def test_q_engine_per_position_shape(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        n_legal = [
            len(list(chess.Board(f).legal_moves)) for f in fens
        ]
        with torch.no_grad():
            _, _, q_eng, _, _ = model.forward_sequence_with_q_dual(graphs)
        for i, q in enumerate(q_eng):
            assert q.shape == (n_legal[i],), (
                f"Position {i}: expected ({n_legal[i]},), got {q.shape}"
            )

    def test_q_human_per_position_shape(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        n_legal = [
            len(list(chess.Board(f).legal_moves)) for f in fens
        ]
        with torch.no_grad():
            _, _, _, q_hum, _ = model.forward_sequence_with_q_dual(graphs)
        for i, q in enumerate(q_hum):
            assert q.shape == (n_legal[i],), (
                f"Position {i}: expected ({n_legal[i]},), got {q.shape}"
            )

    def test_q_engine_finite(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        with torch.no_grad():
            _, _, q_eng, _, _ = model.forward_sequence_with_q_dual(graphs)
        for q in q_eng:
            assert torch.isfinite(q).all()

    def test_q_human_finite(self, model, builder):
        fens = [STARTING_FEN, MIDGAME_FEN]
        graphs = [builder.fen_to_graph(f) for f in fens]
        with torch.no_grad():
            _, _, _, q_hum, _ = model.forward_sequence_with_q_dual(graphs)
        for q in q_hum:
            assert torch.isfinite(q).all()
