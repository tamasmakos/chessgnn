"""
Tests for the three temporal modes of GATEAUChessModel:
  - "none"       (static spatial-only)
  - "global_gru" (GRU over graph-level pooling)
  - "node_gru"   (GRUCell per node type)

Also covers KVCache, forward_step(), and forward_sequence().
"""
import pytest
import torch
import chess

from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel, KVCache

STARTING_FEN = chess.STARTING_FEN
# Short game sequence: start, after e4, after e4-d5
_FENS = [
    chess.STARTING_FEN,
    chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").fen(),
    chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2").fen(),
]

ALL_MODES = ["none", "global_gru", "node_gru"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def builder():
    return ChessGraphBuilder(use_global_node=True, use_move_edges=True)


@pytest.fixture(scope="module")
def builder_base():
    """No global node, no move edges — smallest metadata."""
    return ChessGraphBuilder()


def make_model(builder, mode: str, hidden: int = 48, layers: int = 2) -> GATEAUChessModel:
    metadata = builder.get_metadata()
    m = GATEAUChessModel(metadata, hidden_channels=hidden, num_layers=layers, temporal_mode=mode)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_default_fields_none(self):
        c = KVCache()
        assert c.global_h is None
        assert c.piece_h is None
        assert c.square_h is None

    def test_can_assign_tensors(self):
        h = torch.zeros(1, 1, 32)
        c = KVCache(global_h=h)
        assert c.global_h is h


# ---------------------------------------------------------------------------
# Invalid temporal_mode
# ---------------------------------------------------------------------------

class TestInvalidMode:
    def test_raises_on_bad_mode(self, builder):
        with pytest.raises(ValueError, match="temporal_mode"):
            GATEAUChessModel(builder.get_metadata(), temporal_mode="bad_mode")


# ---------------------------------------------------------------------------
# forward() — value shape and range, all modes
# ---------------------------------------------------------------------------

class TestForwardAllModes:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_value_shape(self, builder, mode):
        model = make_model(builder, mode)
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            v = model(graph)
        assert v.shape == (1, 1)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_value_in_tanh_range(self, builder, mode):
        model = make_model(builder, mode)
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            v = model(graph)
        assert -1.0 <= v.item() <= 1.0

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_value_is_finite(self, builder, mode):
        model = make_model(builder, mode)
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            v = model(graph)
        assert torch.isfinite(v).all()


# ---------------------------------------------------------------------------
# forward_step() — returns (value, KVCache), all modes
# ---------------------------------------------------------------------------

class TestForwardStep:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_returns_tuple(self, builder, mode):
        model = make_model(builder, mode)
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            out = model.forward_step(graph)
        assert isinstance(out, tuple) and len(out) == 2

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_value_shape(self, builder, mode):
        model = make_model(builder, mode)
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            v, _ = model.forward_step(graph)
        assert v.shape == (1, 1)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_cache_type(self, builder, mode):
        model = make_model(builder, mode)
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, cache = model.forward_step(graph)
        assert isinstance(cache, KVCache)

    def test_none_mode_cache_empty(self, builder):
        model = make_model(builder, "none")
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, cache = model.forward_step(graph)
        assert cache.global_h is None
        assert cache.piece_h is None
        assert cache.square_h is None

    def test_global_gru_cache_populated(self, builder):
        model = make_model(builder, "global_gru")
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, cache = model.forward_step(graph)
        assert cache.global_h is not None
        assert cache.global_h.shape == (1, 1, 48)  # [layers=1, batch=1, H]

    def test_node_gru_cache_populated(self, builder):
        model = make_model(builder, "node_gru")
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, cache = model.forward_step(graph)
        assert cache.piece_h is not None
        assert cache.square_h is not None
        assert cache.piece_h.shape[1] == 48   # hidden_channels
        assert cache.square_h.shape == (64, 48)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_cache_reuse_changes_output(self, builder, mode):
        """Passing a used cache should give different output than a fresh None."""
        model = make_model(builder, mode)
        g0 = builder.fen_to_graph(_FENS[0])
        g1 = builder.fen_to_graph(_FENS[1])
        with torch.no_grad():
            v_none, cache = model.forward_step(g1, cache=None)
            _, cache0 = model.forward_step(g0, cache=None)
            v_with_ctx, _ = model.forward_step(g1, cache=cache0)

        if mode == "none":
            # Static mode: cache is not used; outputs must be identical
            assert torch.allclose(v_none, v_with_ctx)
        else:
            # Temporal modes should differ (warm vs cold start)
            assert not torch.allclose(v_none, v_with_ctx)


# ---------------------------------------------------------------------------
# forward_sequence() — returns [T, 1], all modes
# ---------------------------------------------------------------------------

class TestForwardSequence:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_output_shape(self, builder, mode):
        model = make_model(builder, mode)
        graphs = [builder.fen_to_graph(f) for f in _FENS]
        with torch.no_grad():
            values = model.forward_sequence(graphs)
        assert values.shape == (len(_FENS), 1)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_all_finite(self, builder, mode):
        model = make_model(builder, mode)
        graphs = [builder.fen_to_graph(f) for f in _FENS]
        with torch.no_grad():
            values = model.forward_sequence(graphs)
        assert torch.isfinite(values).all()

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_all_in_tanh_range(self, builder, mode):
        model = make_model(builder, mode)
        graphs = [builder.fen_to_graph(f) for f in _FENS]
        with torch.no_grad():
            values = model.forward_sequence(graphs)
        assert values.min().item() >= -1.0
        assert values.max().item() <= 1.0

    def test_sequence_consistent_with_step(self, builder):
        """forward_sequence and repeated forward_step must produce identical outputs."""
        model = make_model(builder, "global_gru")
        graphs = [builder.fen_to_graph(f) for f in _FENS]

        with torch.no_grad():
            seq_vals = model.forward_sequence(graphs)

            step_vals = []
            cache = None
            for g in graphs:
                v, cache = model.forward_step(g, cache)
                step_vals.append(v)
            step_vals = torch.cat(step_vals, dim=0)

        assert torch.allclose(seq_vals, step_vals, atol=1e-5)

    def test_node_gru_sequence_consistent_with_step(self, builder):
        model = make_model(builder, "node_gru")
        graphs = [builder.fen_to_graph(f) for f in _FENS]

        with torch.no_grad():
            seq_vals = model.forward_sequence(graphs)

            step_vals = []
            cache = None
            for g in graphs:
                v, cache = model.forward_step(g, cache)
                step_vals.append(v)
            step_vals = torch.cat(step_vals, dim=0)

        assert torch.allclose(seq_vals, step_vals, atol=1e-5)


# ---------------------------------------------------------------------------
# forward_with_q() still works across all modes
# ---------------------------------------------------------------------------

class TestQHeadWithModes:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_q_shape(self, builder, mode):
        model = make_model(builder, mode)
        board = chess.Board(STARTING_FEN)
        num_moves = len(list(board.legal_moves))
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, q_scores, move_ei = model.forward_with_q(graph)
        assert q_scores.shape == (num_moves,)
        assert move_ei.shape == (2, num_moves)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_q_finite(self, builder, mode):
        model = make_model(builder, mode)
        graph = builder.fen_to_graph(STARTING_FEN)
        with torch.no_grad():
            _, q_scores, _ = model.forward_with_q(graph)
        assert torch.isfinite(q_scores).all()


# ---------------------------------------------------------------------------
# Backward pass (gradient flow)
# ---------------------------------------------------------------------------

class TestGradientFlow:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_backward_sequence(self, builder, mode):
        model = make_model(builder, mode)
        model.train()
        graphs = [builder.fen_to_graph(f) for f in _FENS]
        values = model.forward_sequence(graphs)
        target = torch.tensor([[0.5], [0.3], [0.4]])
        loss = torch.nn.functional.mse_loss(values, target)
        loss.backward()
        # At least one parameter must have a non-None gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
