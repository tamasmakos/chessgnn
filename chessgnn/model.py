
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
from torch.nn import ParameterDict
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones

class WeightedHGTConv(MessagePassing):
    """
    Heterogeneous Graph Transformer with Edge Weights.
    """
    def __init__(self, in_channels, out_channels, metadata, heads=1, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.d_k = out_channels // heads
        self.metadata = metadata

        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        self.a_lin = nn.ModuleDict()
        
        self.skip = ParameterDict()
        
        node_types, edge_types = metadata
        
        for nt in node_types:
            self.k_lin[nt] = Linear(in_channels, out_channels)
            self.q_lin[nt] = Linear(in_channels, out_channels)
            self.v_lin[nt] = Linear(in_channels, out_channels)
            self.a_lin[nt] = Linear(out_channels, out_channels)
            
            if in_channels != out_channels:
                self.skip[nt] = Linear(in_channels, out_channels)
            else:
                self.skip[nt] = nn.Identity()
            
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        self.relation_pri = nn.ParameterDict()
        
        for et in edge_types:
            et_str = '__'.join(et)
            self.relation_att[et_str] = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
            self.relation_msg[et_str] = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
            self.relation_pri[et_str] = nn.Parameter(torch.ones(1))
            
            glorot(self.relation_att[et_str])
            glorot(self.relation_msg[et_str])

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        out_dict = {}
        
        # Prepare Q, K, V
        k_dict, q_dict, v_dict = {}, {}, {}
        for nt, x in x_dict.items():
            k_dict[nt] = self.k_lin[nt](x).view(-1, self.heads, self.d_k)
            q_dict[nt] = self.q_lin[nt](x).view(-1, self.heads, self.d_k)
            v_dict[nt] = self.v_lin[nt](x).view(-1, self.heads, self.d_k)

        # Propagate
        for et, edge_index in edge_index_dict.items():
            src_type, _, dst_type = et
            et_str = '__'.join(et)
            
            # Edge Weights
            # If edge_weight_dict is provided and has this edge type
            edge_weight = None
            if edge_weight_dict and et in edge_weight_dict:
                edge_weight = edge_weight_dict[et] # Expecting [E, 1] or [E]
            
            out = self.propagate(edge_index, 
                                 k=k_dict[src_type], 
                                 q=q_dict[dst_type], 
                                 v=v_dict[src_type], 
                                 rel_att=self.relation_att[et_str],
                                 rel_msg=self.relation_msg[et_str],
                                 rel_pri=self.relation_pri[et_str],
                                 edge_weight=edge_weight)
                                 
            if dst_type not in out_dict:
                out_dict[dst_type] = out
            else:
                out_dict[dst_type] += out
                
        # Skip connection & Update
        for nt in out_dict:
            out_dict[nt] = self.a_lin[nt](out_dict[nt].view(-1, self.out_channels))
            out_dict[nt] += self.skip[nt](x_dict[nt])
            out_dict[nt] = F.gelu(out_dict[nt])
            
        return out_dict

    def message(self, k_j, q_i, v_j, rel_att, rel_msg, rel_pri, index, edge_weight):
        # k_j: [E, Heads, D]
        
        # Attention Score
        # (K * W_att) * Q
        k_att = torch.einsum('ehd, hdk -> ehk', k_j, rel_att)
        alpha = (k_att * q_i).sum(dim=-1) * rel_pri / math.sqrt(self.d_k) # [E, Heads]
        
        # Integrate Edge Weight into Attention
        if edge_weight is not None:
            # edge_weight: [E] or [E, 1]
            if edge_weight.dim() == 1: edge_weight = edge_weight.unsqueeze(-1)
            # alpha = alpha * (1 + lambda * w)
            # Simplified: alpha = alpha + log(w)? Or multiplicative?
            # Specification: "AttnHead ... * (1 + lambda w_st)" -> This modulates the ATTENTION VALUE (pre-softmax) or PROBABILITY?
            # Usually pre-softmax logits.
            # Let's add it to logits to bias attention.
            # alpha += edge_weight
            alpha = alpha * (1.0 + edge_weight)

        alpha = softmax(alpha, index)
        
        # Message
        # V * W_msg
        v_msg = torch.einsum('ehd, hdk -> ehk', v_j, rel_msg)
        return v_msg * alpha.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Input feature dimensions produced by ChessGraphBuilder
# ---------------------------------------------------------------------------
_NODE_INPUT_DIMS: dict[str, int] = {'piece': 15, 'square': 3, 'global': 11}

# Maximum node counts used to pre-allocate node-GRU hidden states.
# Piece count varies per position (≤ 32); square count is always 64.
_MAX_PIECES = 32
_NUM_SQUARES = 64


from dataclasses import dataclass, field


@dataclass
class KVCache:
    """
    Holds per-step hidden states for incremental online inference.

    temporal_mode="none"
        All fields are None; cache is unused.

    temporal_mode="global_gru"
        ``global_h``: Tensor [1, 1, H] — GRU hidden state.

    temporal_mode="node_gru"
        ``piece_h``: Tensor [max_pieces, H] — per-piece hidden states,
            row i is the hidden vector for piece index i in the *last* graph.
            Indices change between positions (pieces are captured / appear);
            callers must pass the ``piece_id_map`` returned by
            ``forward_step`` to correctly re-index surviving pieces.
        ``square_h``: Tensor [64, H] — per-square hidden states (stable
            indices, squares never appear or disappear).
    """
    global_h: torch.Tensor | None = None
    piece_h: torch.Tensor | None = None
    square_h: torch.Tensor | None = None


class GATEAUChessModel(nn.Module):
    """
    Edge-aware GNN with a value head V(s), Q-head Q(s,a), and three
    temporal context modes.

    Architecture
    ------------
    * Per-node-type linear input projection → hidden_channels
    * num_layers × WeightedHGTConv + LayerNorm (residual inside conv)
    * Temporal context (controlled by ``temporal_mode``):
      - ``"none"``       — no recurrence; value from spatial pooling only.
      - ``"global_gru"`` — GRU over [piece_pool ‖ square_pool] graph embedding.
      - ``"node_gru"``   — GRUCell per node type; piece and square nodes each
                           carry their own hidden state across moves.
    * Value head: temporal embedding → MLP → tanh scalar in [-1, 1]
    * Q-head: h_piece ‖ h_square ‖ edge_attr(5) → 2-layer MLP → scalar

    Parameters
    ----------
    metadata : tuple
        (node_types, edge_types) from ChessGraphBuilder.get_metadata().
    hidden_channels : int
        Width of all internal representations (default 128).
    num_layers : int
        Number of WeightedHGTConv stages (default 4).
    temporal_mode : str
        One of ``"none"``, ``"global_gru"``, ``"node_gru"`` (default ``"none"``).
    """

    _VALID_MODES = frozenset({"none", "global_gru", "node_gru"})

    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 128,
        num_layers: int = 4,
        temporal_mode: str = "none",
    ):
        if temporal_mode not in self._VALID_MODES:
            raise ValueError(f"temporal_mode must be one of {self._VALID_MODES}, got {temporal_mode!r}")

        super().__init__()
        node_types, _ = metadata
        self._node_types: list[str] = list(node_types)
        self.hidden_channels = hidden_channels
        self.temporal_mode = temporal_mode

        # Per-type input projections
        self.input_proj = nn.ModuleDict({
            nt: Linear(_NODE_INPUT_DIMS[nt], hidden_channels)
            for nt in node_types
        })

        # ELO embedding: maps normalised scalar elo ∈ [0,1] → R^elo_dim
        # Injected into value head and q head at inference time.
        self._elo_dim = 16
        self.elo_embed = nn.Sequential(
            nn.Linear(1, self._elo_dim),
            nn.GELU(),
        )

        # GNN backbone
        self.convs = nn.ModuleList([
            WeightedHGTConv(hidden_channels, hidden_channels, metadata, heads=4)
            for _ in range(num_layers)
        ])

        # LayerNorm after each conv, per node type
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            for _ in range(num_layers)
        ])

        # ----- Temporal modules (mode-specific) -----
        pool_dim = hidden_channels * len(node_types)

        if temporal_mode == "global_gru":
            # GRU input: mean-pooled [piece ‖ square ‖ …] concatenation
            self.global_gru = nn.GRU(pool_dim, hidden_channels, batch_first=True)
            value_in_dim = hidden_channels
        elif temporal_mode == "node_gru":
            # One GRUCell per node type that has spatial nodes
            self.piece_gru_cell = nn.GRUCell(hidden_channels, hidden_channels)
            self.square_gru_cell = nn.GRUCell(hidden_channels, hidden_channels)
            # After node-GRU update, pool updated nodes → value head
            value_in_dim = pool_dim
        else:  # "none"
            value_in_dim = pool_dim

        # Value head — accepts optional ELO embedding concat
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim + self._elo_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

        # Q-head: h_piece ‖ h_square ‖ edge_attr(5) ‖ elo_embed → scalar
        self.q_head = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 5 + self._elo_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, graph) -> dict:
        """Project inputs and run GNN backbone. Returns final x_dict."""
        x_dict = {nt: self.input_proj[nt](graph.x_dict[nt]) for nt in self._node_types}

        ew_dict = {}
        if ('piece', 'interacts', 'piece') in graph.edge_attr_dict:
            ew_dict[('piece', 'interacts', 'piece')] = (
                graph['piece', 'interacts', 'piece'].edge_attr[:, 0]
            )

        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, graph.edge_index_dict, ew_dict)
            x_dict = {nt: norm_dict[nt](x) for nt, x in x_dict.items()}

        return x_dict

    def _encode_batch(self, graphs: list) -> list[dict]:
        """Run GNN backbone on all graphs in one fused batched pass.

        All T graphs are stacked into a single ``Batch``, run through the GNN
        layers together (GPU sees T× more nodes), then split back into
        per-graph ``x_dict`` dicts.  This increases GPU occupancy substantially
        compared to calling :meth:`_encode` T times in a loop.

        Returns
        -------
        list of dict  — same length as *graphs*, each entry is the per-position
        x_dict produced by the GNN (equivalent to calling ``_encode`` individually).
        """
        from torch_geometric.data import Batch

        batched = Batch.from_data_list(graphs)

        x_dict = {nt: self.input_proj[nt](batched.x_dict[nt]) for nt in self._node_types}

        ew_dict = {}
        edge_key = ('piece', 'interacts', 'piece')
        if edge_key in batched.edge_attr_dict:
            ew_dict[edge_key] = batched[edge_key].edge_attr[:, 0]

        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, batched.edge_index_dict, ew_dict)
            x_dict = {nt: norm_dict[nt](x) for nt, x in x_dict.items()}

        # Split by graph index
        T = len(graphs)
        result: list[dict] = []
        batch_vecs = {nt: batched[nt].batch for nt in self._node_types}
        for t in range(T):
            result.append({nt: x_dict[nt][batch_vecs[nt] == t] for nt in self._node_types})
        return result

    def _pool_concat(self, x_dict: dict) -> torch.Tensor:
        """Mean-pool all node types and concatenate into one vector [1, pool_dim]."""
        pools = [x_dict[nt].mean(dim=0, keepdim=True) for nt in self._node_types]
        return torch.cat(pools, dim=1)

    def _apply_temporal(
        self,
        x_dict: dict,
        cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Apply the temporal module for the current step.

        Returns
        -------
        value_feat : Tensor [1, value_in_dim]
            Feature vector to feed into the value head.
        new_cache : KVCache
            Updated cache for the next call.
        """
        new_cache = KVCache()

        if self.temporal_mode == "none":
            feat = self._pool_concat(x_dict)           # [1, pool_dim]
            return feat, new_cache

        if self.temporal_mode == "global_gru":
            pool = self._pool_concat(x_dict)           # [1, pool_dim]
            h_prev = cache.global_h if cache is not None else None
            # GRU expects [batch=1, seq=1, input]
            gru_in = pool.unsqueeze(0)                 # [1, 1, pool_dim]
            gru_out, h_new = self.global_gru(gru_in, h_prev)
            feat = gru_out.squeeze(1)                  # [1, H]
            new_cache.global_h = h_new
            return feat, new_cache

        # temporal_mode == "node_gru"
        num_pieces = x_dict['piece'].shape[0]
        device = x_dict['piece'].device

        if cache is not None and cache.piece_h is not None:
            piece_h_prev = cache.piece_h[:num_pieces]  # [N_p, H] (trim/pad for captures)
            if piece_h_prev.shape[0] < num_pieces:
                # New pieces materialised (promotion) — pad with zeros
                pad = torch.zeros(num_pieces - piece_h_prev.shape[0], self.hidden_channels, device=device)
                piece_h_prev = torch.cat([piece_h_prev, pad], dim=0)
        else:
            piece_h_prev = torch.zeros(num_pieces, self.hidden_channels, device=device)

        if cache is not None and cache.square_h is not None:
            square_h_prev = cache.square_h              # [64, H]
        else:
            square_h_prev = torch.zeros(_NUM_SQUARES, self.hidden_channels, device=device)

        piece_h_new = self.piece_gru_cell(x_dict['piece'], piece_h_prev)    # [N_p, H]
        square_h_new = self.square_gru_cell(x_dict['square'], square_h_prev)  # [64, H]

        new_cache.piece_h = piece_h_new.detach()
        new_cache.square_h = square_h_new.detach()

        # Build updated x_dict for pooling (replace embeddings with GRU outputs)
        x_dict_updated = dict(x_dict)
        x_dict_updated['piece'] = piece_h_new
        x_dict_updated['square'] = square_h_new

        feat = self._pool_concat(x_dict_updated)       # [1, pool_dim]
        return feat, new_cache

    def _pool_value(self, feat: torch.Tensor, elo_norm: float = 1.0) -> torch.Tensor:
        """Pass a pre-pooled feature through the value head → [1, 1].

        Parameters
        ----------
        elo_norm : float
            Normalised ELO in [0, 1].  1.0 = perfect play (Stockfish / 3000).  
            0.5 ≈ 1500 Elo.  Passed through ``elo_embed`` and concatenated to
            ``feat`` before the value MLP.
        """
        device = feat.device
        elo_t = torch.tensor([[elo_norm]], dtype=torch.float32, device=device)  # [1, 1]
        elo_e = self.elo_embed(elo_t)                                            # [1, elo_dim]
        return self.value_head(torch.cat([feat, elo_e], dim=1))

    # ------------------------------------------------------------------
    # Public API — single position
    # ------------------------------------------------------------------

    def forward(self, graph, cache: KVCache | None = None, elo_norm: float = 1.0) -> torch.Tensor:
        """
        Returns value V(s) as a [1, 1] tensor in [-1, 1].

        For temporal modes other than ``"none"`` the cache is updated
        internally but not returned; use ``forward_step`` for incremental
        online inference where you need the updated cache.
        """
        x_dict = self._encode(graph)
        feat, _ = self._apply_temporal(x_dict, cache)
        return self._pool_value(feat, elo_norm)

    def forward_step(
        self,
        graph,
        cache: KVCache | None = None,
        elo_norm: float = 1.0,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Single-step inference returning value and the updated cache.

        Parameters
        ----------
        graph : HeteroData
            Current board position graph.
        cache : KVCache, optional
            Hidden state from the previous move. Pass ``None`` at game start.
        elo_norm : float
            Normalised ELO in [0, 1].  1.0 = perfect play.  0.5 ≈ 1500 Elo.

        Returns
        -------
        value : Tensor [1, 1]
            Win probability in [-1, 1].
        new_cache : KVCache
            Updated hidden state to pass to the next call.
        """
        x_dict = self._encode(graph)
        feat, new_cache = self._apply_temporal(x_dict, cache)
        value = self._pool_value(feat, elo_norm)
        return value, new_cache

    def forward_with_q(
        self,
        graph,
        cache: KVCache | None = None,
        elo_norm: float = 1.0,
        return_cache: bool = False,
        return_embeddings: bool = False,
    ) -> tuple:
        """
        Single-pass inference returning value and per-move Q scores.

        Parameters
        ----------
        return_cache : bool
            When True, appends the updated ``KVCache`` to the return tuple.
        return_embeddings : bool
            When True, appends the final GNN ``x_dict`` to the return tuple.
            ``x_dict['piece']`` is shaped [N_pieces, H].

        Returns
        -------
        value : Tensor [1, 1]
        q_scores : Tensor [M]
        move_edge_index : Tensor [2, M]
        new_cache : KVCache  (only when return_cache=True)
        x_dict : dict[str, Tensor]  (only when return_embeddings=True)
        """
        x_dict = self._encode(graph)
        feat, new_cache = self._apply_temporal(x_dict, cache)
        value = self._pool_value(feat, elo_norm)

        move_edge_index = graph['piece', 'move', 'square'].edge_index  # [2, M]
        move_edge_attr = graph['piece', 'move', 'square'].edge_attr    # [M, 5]

        device = feat.device
        elo_t = torch.tensor([[elo_norm]], dtype=torch.float32, device=device)
        elo_e = self.elo_embed(elo_t)                                   # [1, elo_dim]
        elo_expand = elo_e.expand(move_edge_index.shape[1], -1)        # [M, elo_dim]

        src_idx, dst_idx = move_edge_index
        h_src = x_dict['piece'][src_idx]   # [M, H]
        h_dst = x_dict['square'][dst_idx]  # [M, H]
        q_feats = torch.cat([h_src, h_dst, move_edge_attr, elo_expand], dim=1)  # [M, 2H+5+elo_dim]
        q_scores = self.q_head(q_feats).squeeze(-1)                             # [M]

        result: list = [value, q_scores, move_edge_index]
        if return_cache:
            result.append(new_cache)
        if return_embeddings:
            result.append(x_dict)
        return tuple(result)

    # ------------------------------------------------------------------
    # Public API — sequence (training)
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        graphs: list,
        elo_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        Process a sequence of positions and return value predictions for
        every step. Used during training on game histories.

        Parameters
        ----------
        graphs : list of HeteroData
            Ordered list of board positions (moves 0 … T-1).
        elo_norm : float
            Normalised ELO in [0, 1] shared across the whole game sequence.

        Returns
        -------
        values : Tensor [T, 1]
            Value prediction for each position in the sequence.
        """
        values = []
        cache: KVCache | None = None
        for graph in graphs:
            x_dict = self._encode(graph)
            feat, cache = self._apply_temporal(x_dict, cache)
            values.append(self._pool_value(feat, elo_norm))
        return torch.cat(values, dim=0)  # [T, 1]

    def forward_sequence_with_q(
        self,
        graphs: list,
        elo_norm: float = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Process a full game sequence, returning value + Q scores at every step.

        The GRU hidden state is chained across all positions so the model
        accumulates temporal context (piece stability, threat evolution, etc.).

        Parameters
        ----------
        graphs : list of HeteroData
            Ordered board positions from a single game (e.g. moves 10 … N).
        elo_norm : float
            Normalised ELO in [0, 1] shared across the game.

        Returns
        -------
        values : Tensor [T, 1]
            Value at each position in the sequence.
        q_scores_list : list of Tensor [M_t]
            Per-move Q logits at each step t; M_t may differ (captures change
            the number of legal moves).
        move_edge_indices : list of Tensor [2, M_t]
            Move edge indices at each step, matching board.legal_moves order.
        """
        values: list[torch.Tensor] = []
        q_scores_list: list[torch.Tensor] = []
        move_edge_indices: list[torch.Tensor] = []

        device = next(self.parameters()).device
        elo_t = torch.tensor([[elo_norm]], dtype=torch.float32, device=device)
        elo_e = self.elo_embed(elo_t)  # [1, elo_dim]; computed once per game

        cache: KVCache | None = None
        for graph in graphs:
            x_dict = self._encode(graph)
            feat, cache = self._apply_temporal(x_dict, cache)
            values.append(self._pool_value(feat, elo_norm))

            move_edge_index = graph['piece', 'move', 'square'].edge_index  # [2, M]
            move_edge_attr = graph['piece', 'move', 'square'].edge_attr    # [M, 5]
            m = move_edge_index.shape[1]

            if m > 0:
                elo_expand = elo_e.expand(m, -1)                          # [M, elo_dim]
                src_idx, dst_idx = move_edge_index
                h_src = x_dict['piece'][src_idx]                          # [M, H]
                h_dst = x_dict['square'][dst_idx]                        # [M, H]
                q_feats = torch.cat([h_src, h_dst, move_edge_attr, elo_expand], dim=1)
                q_scores = self.q_head(q_feats).squeeze(-1)               # [M]
            else:
                q_scores = torch.empty(0, device=device)

            q_scores_list.append(q_scores)
            move_edge_indices.append(move_edge_index)

        return torch.cat(values, dim=0), q_scores_list, move_edge_indices

    def forward_sequence_with_q_dual(
        self,
        graphs: list,
        elo_norm_sf: float = 1.0,
        elo_norm_player: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Like :meth:`forward_sequence_with_q` but decodes **two** value tensors
        in a single GNN backbone pass.

        The GNN + GRU chain runs once per position.  After accumulating all
        temporal features the value head is applied twice with different ELO
        embeddings.  This halves backbone compute for the dual-target training
        objective compared to calling :meth:`forward_sequence_with_q` twice.

        Parameters
        ----------
        graphs : list of HeteroData
        elo_norm_sf : float
            ELO for the Stockfish-perspective value (default 1.0).
        elo_norm_player : float
            ELO for the player-outcome value.

        Returns
        -------
        values_sf : Tensor [T, 1]
        values_player : Tensor [T, 1]
        q_scores_list : list of Tensor [M_t]  (uses elo_norm_sf for Q)
        move_edge_indices : list of Tensor [2, M_t]
        """
        device = next(self.parameters()).device

        elo_t_sf = torch.tensor([[elo_norm_sf]], dtype=torch.float32, device=device)
        elo_e_sf = self.elo_embed(elo_t_sf)          # [1, elo_dim]

        elo_t_pl = torch.tensor([[elo_norm_player]], dtype=torch.float32, device=device)
        elo_e_pl = self.elo_embed(elo_t_pl)          # [1, elo_dim]

        vals_sf: list[torch.Tensor] = []
        vals_pl: list[torch.Tensor] = []
        q_scores_list: list[torch.Tensor] = []
        move_edge_indices: list[torch.Tensor] = []

        # Batch-encode all T positions through the GNN in one fused pass.
        # The GRU still runs sequentially (temporal dependency), but the
        # expensive HGT convolutions now see T× more nodes on the GPU.
        all_x_dicts = self._encode_batch(graphs)

        cache: KVCache | None = None
        for t, (graph, x_dict) in enumerate(zip(graphs, all_x_dicts)):
            feat, cache = self._apply_temporal(x_dict, cache)

            # Value head: two ELO conditions, one backbone
            vals_sf.append(self.value_head(torch.cat([feat, elo_e_sf], dim=1)))
            vals_pl.append(self.value_head(torch.cat([feat, elo_e_pl], dim=1)))

            move_edge_index = graph['piece', 'move', 'square'].edge_index
            move_edge_attr = graph['piece', 'move', 'square'].edge_attr
            m = move_edge_index.shape[1]

            if m > 0:
                elo_expand = elo_e_sf.expand(m, -1)
                src_idx, dst_idx = move_edge_index
                h_src = x_dict['piece'][src_idx]
                h_dst = x_dict['square'][dst_idx]
                q_feats = torch.cat([h_src, h_dst, move_edge_attr, elo_expand], dim=1)
                q_scores = self.q_head(q_feats).squeeze(-1)
            else:
                q_scores = torch.empty(0, device=device)

            q_scores_list.append(q_scores)
            move_edge_indices.append(move_edge_index)

        return (
            torch.cat(vals_sf, dim=0),
            torch.cat(vals_pl, dim=0),
            q_scores_list,
            move_edge_indices,
        )
