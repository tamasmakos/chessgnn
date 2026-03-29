
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

class RayAlignmentBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.transmissibility_net = nn.Linear(in_channels, 1)
        self.ray_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x_dict, edge_index_ray, edge_attr_ray):
        # x_dict['piece']: [N_p, D]
        # edge_index_ray: [2, E_ray]
        # edge_attr_ray: [E_ray, 2] (dist, blocking)
        
        x = x_dict['piece']
        
        # 1. Compute Transmissibility per node
        # T_v = sigmoid(W z_v + b)
        # Represents likelihood to PASS influence.
        t_val = torch.sigmoid(self.transmissibility_net(x)) # [N, 1]
        
        # 2. Propagate
        # Influence = Gate(blocking_count) * Gate(Transmissibility of blockers)?
        # For MVP: We assume direct edge exists if aligned.
        # We modulate the message by 1 / (1 + blocking_count).
        # This is a soft "inverse distance" weighting based on obstacles.
        
        # E_ray: [dist, blocking]
        blocking_count = edge_attr_ray[:, 1].unsqueeze(-1) # [E, 1]
        dist = edge_attr_ray[:, 0].unsqueeze(-1) # [E, 1]
        
        # Weight = 1 / (1 + blocking + 0.1 * dist) 
        # Favor closer and unblocked pieces.
        ray_weight = 1.0 / (1.0 + blocking_count + 0.1 * dist)

        # Apply Source Transmissibility
        # If the source piece is NOT a slider (e.g. Knight), it should not be sending rays.
        # ray_weight *= t_val[source]
        row, col = edge_index_ray
        source_transmissibility = t_val[row] 
        ray_weight = ray_weight * source_transmissibility
        
        # Message passing manually or via sparse MM?
        # Simple GAT-like: Target = Sum( Weight * Source )
        # Using sparse matrix multiplication logic or loop.
        
        # Project source
        x_proj = self.ray_proj(x)
        
        # Scatter add (Message Passing)
        row, col = edge_index_ray
        
        # Message = x_proj[source] * weight
        msg = x_proj[row] * ray_weight
        
        # Aggregate to target
        # Using a simple scatter add implementation if torch_scatter is avail, else loop (slow) or use index_add?
        # Standard GNN approach:
        out = torch.zeros_like(x)
        out.index_add_(0, col, msg)
        
        # Update piece features
        # Residual connection
        x_new = x + out
        
        # Update dict
        x_dict['piece'] = x_new
        return x_dict



class STHGATLikeModel(nn.Module):
    def __init__(self, metadata, hidden_channels=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.square_proj = Linear(3, 10) 
        self.ray_block = RayAlignmentBlock(hidden_channels) 
        
        # Stack of GNN Layers
        self.convs = nn.ModuleList()
        # First layer: input dim 10 -> hidden
        self.convs.append(WeightedHGTConv(10, hidden_channels, metadata, heads=4))
        
        # Subsequent layers: hidden -> hidden
        for _ in range(num_layers - 1):
             self.convs.append(WeightedHGTConv(hidden_channels, hidden_channels, metadata, heads=4))

        # Temporal RNN (Simplified: GRU over global graph state)
        # Input to GRU is now 2 * hidden (Piece + Square pool)
        self.global_gru = nn.GRU(hidden_channels * 2, hidden_channels, batch_first=True)
        
        # 1. Main Win Probability Head (Outputs [White, Draw, Black] logits)
        self.win_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3) 
        )
        
        # 2. Auxiliary Head: Material Balance
        self.material_head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Predicts (WhiteMat - BlackMat)
        )
        
        # 3. Auxiliary Head: Graph Dominance (PageRank Diff)
        self.dominance_head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Predicts (WhiteCent - BlackCent)
        )

    def forward(self, sequence_graphs):
        # sequence_graphs: List[HeteroData] (Length T) representing the full game
        
        # 1. Batch the entire sequence into one Super-Graph
        # This allows parallel computation of GNN over all time steps
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(sequence_graphs)
        
        # 2. Spatial Encode (Parallel)
        x_dict = batch.x_dict.copy()
        x_dict['square'] = self.square_proj(x_dict['square'])
        
        # Edge Weights
        ew_dict = {}
        if ('piece', 'interacts', 'piece') in batch.edge_attr_dict:
                ew_dict[('piece', 'interacts', 'piece')] = batch['piece', 'interacts', 'piece'].edge_attr[:, 0]

        # Multi-Layer GNN Pass
        for i, conv in enumerate(self.convs):
                x_dict = conv(x_dict, batch.edge_index_dict, ew_dict)
        
        z_dict = x_dict 

        # Ray Alignment Block (Spatial Logic)
        if ('piece', 'ray', 'piece') in batch.edge_index_dict:
                x_dict = self.ray_block(x_dict, 
                                        batch.edge_index_dict[('piece', 'ray', 'piece')],
                                        batch.edge_attr_dict[('piece', 'ray', 'piece')])

        z_dict = x_dict 

        # 3. Pooling (Graph Level)
        # We need to pool nodes back to their respective graphs in the batch.
        # Batch.batch maps nodes to batch index.
        # batch.batch_dict['piece'] -> [N_total_pieces] (indices 0 to T-1)
        
        from torch_geometric.nn import global_mean_pool
        
        piece_embeds = z_dict['piece'] 
        square_embeds = z_dict['square']
        
        # Pool Pieces
        # Some graphs might have NO pieces (rare but possible in empty board? No, kings always exist).
        # We assume batch indices exist.
        batch_index_piece = batch[ 'piece' ].batch
        p_pool = global_mean_pool(piece_embeds, batch_index_piece) # [T, Hidden]
        
        # Pool Squares
        batch_index_square = batch[ 'square' ].batch
        s_pool = global_mean_pool(square_embeds, batch_index_square) # [T, Hidden]
        
        # Concatenate
        graph_embeds = torch.cat([p_pool, s_pool], dim=1) # [T, 2 * Hidden]
        
        # 4. Temporal Process (GRU)
        # Input: [Batch=1, Seq_Len=T, Input_Dim]
        seq_tensor = graph_embeds.unsqueeze(0) 
        
        # GRU
        # output: [1, T, Hidden]
        gru_out, _ = self.global_gru(seq_tensor)
        
        # 5. Predict Win Probability for ALL steps
        # win_logits: [1, T, 3]
        win_logits = self.win_head(gru_out)
        
        # Aux: [1, T, 1]
        mat_pred = self.material_head(gru_out)
        dom_pred = self.dominance_head(gru_out)
        
        return win_logits, mat_pred, dom_pred

    def forward_step(self, graph, h_prev=None):
        """
        Runs one step of inference for minimal latency.
        Args:
            graph (HeteroData): The current board state graph.
            h_prev (Tensor, optional): Previous GRU hidden state [1, 1, Hidden].
        Returns:
            value (Tensor): Win probability logit [1, 1].
            h_new (Tensor): New GRU hidden state [1, 1, Hidden].
        """
        # 1. Spatial Encode (Single Graph)
        x_dict = graph.x_dict.copy()
        x_dict['square'] = self.square_proj(x_dict['square'])
        
        # Edge Weights
        ew_dict = {}
        if ('piece', 'interacts', 'piece') in graph.edge_attr_dict:
                 ew_dict[('piece', 'interacts', 'piece')] = graph['piece', 'interacts', 'piece'].edge_attr[:, 0]

        # Multi-Layer GNN Pass
        for i, conv in enumerate(self.convs):
            # No batch index needed for single graph, but GNN expects edge_index dict
            x_dict = conv(x_dict, graph.edge_index_dict, ew_dict)
        
        z_dict = x_dict 

        # Ray Alignment Block
        if ('piece', 'ray', 'piece') in graph.edge_index_dict:
                x_dict = self.ray_block(x_dict, 
                                        graph.edge_index_dict[('piece', 'ray', 'piece')],
                                        graph.edge_attr_dict[('piece', 'ray', 'piece')])

        z_dict = x_dict 
        
        # 2. Pooling
        from torch_geometric.nn import  global_mean_pool
        
        # For a single graph without batch attribute, we treat all nodes as batch 0
        p_pool = torch.mean(z_dict['piece'], dim=0, keepdim=True) # [1, Hidden]
        s_pool = torch.mean(z_dict['square'], dim=0, keepdim=True) # [1, Hidden]
        
        graph_embed = torch.cat([p_pool, s_pool], dim=1) # [1, 2 * Hidden]
        
        # 3. Temporal Update (GRU Cell equivalent)
        # GRU expects input [Batch, Seq, Feature] if using full GRU with batch_first=True
        # We process a sequence of length 1
        seq_tensor = graph_embed.unsqueeze(0) # [1, 1, 2*Hidden]
        
        if h_prev is None:
            # Init hidden state
             h_prev = torch.zeros(1, 1, self.hidden_channels, device=graph_embed.device)
        
        gru_out, h_new = self.global_gru(seq_tensor, h_prev)
        # gru_out: [1, 1, Hidden]
        # h_new: [1, 1, Hidden]
        
        # 4. Predict
        feat = gru_out.squeeze(0).squeeze(0) # [Hidden]
        
        win_logits = self.win_head(feat)       # [3]
        mat_pred = self.material_head(feat)    # [1]
        dom_pred = self.dominance_head(feat)   # [1]
        
        return (win_logits, mat_pred, dom_pred), h_new


# ---------------------------------------------------------------------------
# Input feature dimensions produced by ChessGraphBuilder
# ---------------------------------------------------------------------------
_NODE_INPUT_DIMS: dict[str, int] = {'piece': 10, 'square': 3, 'global': 9}

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

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

        # Q-head: h_piece ‖ h_square ‖ edge_attr(5) → scalar
        self.q_head = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 5, hidden_channels),
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

    def _pool_value(self, feat: torch.Tensor) -> torch.Tensor:
        """Pass a pre-pooled feature through the value head → [1, 1]."""
        return self.value_head(feat)

    # ------------------------------------------------------------------
    # Public API — single position
    # ------------------------------------------------------------------

    def forward(self, graph, cache: KVCache | None = None) -> torch.Tensor:
        """
        Returns value V(s) as a [1, 1] tensor in [-1, 1].

        For temporal modes other than ``"none"`` the cache is updated
        internally but not returned; use ``forward_step`` for incremental
        online inference where you need the updated cache.
        """
        x_dict = self._encode(graph)
        feat, _ = self._apply_temporal(x_dict, cache)
        return self._pool_value(feat)

    def forward_step(
        self,
        graph,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Single-step inference returning value and the updated cache.

        Parameters
        ----------
        graph : HeteroData
            Current board position graph.
        cache : KVCache, optional
            Hidden state from the previous move. Pass ``None`` at game start.

        Returns
        -------
        value : Tensor [1, 1]
            Win probability in [-1, 1].
        new_cache : KVCache
            Updated hidden state to pass to the next call.
        """
        x_dict = self._encode(graph)
        feat, new_cache = self._apply_temporal(x_dict, cache)
        value = self._pool_value(feat)
        return value, new_cache

    def forward_with_q(
        self,
        graph,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-pass inference returning value and per-move Q scores.

        Returns
        -------
        value : Tensor [1, 1]
            Win probability in [-1, 1] (tanh); +1 = white wins.
        q_scores : Tensor [M]
            Raw Q logit for each of the M legal move edges.
        move_edge_index : Tensor [2, M]
            Edge index (piece_src, square_dst) for each move, matching
            the order of board.legal_moves used by ChessGraphBuilder.
        """
        x_dict = self._encode(graph)
        feat, _ = self._apply_temporal(x_dict, cache)
        value = self._pool_value(feat)

        move_edge_index = graph['piece', 'move', 'square'].edge_index  # [2, M]
        move_edge_attr = graph['piece', 'move', 'square'].edge_attr    # [M, 5]

        src_idx, dst_idx = move_edge_index
        h_src = x_dict['piece'][src_idx]   # [M, H]
        h_dst = x_dict['square'][dst_idx]  # [M, H]
        q_feats = torch.cat([h_src, h_dst, move_edge_attr], dim=1)  # [M, 2H+5]
        q_scores = self.q_head(q_feats).squeeze(-1)                  # [M]

        return value, q_scores, move_edge_index

    # ------------------------------------------------------------------
    # Public API — sequence (training)
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        graphs: list,
    ) -> torch.Tensor:
        """
        Process a sequence of positions and return value predictions for
        every step. Used during training on game histories.

        Parameters
        ----------
        graphs : list of HeteroData
            Ordered list of board positions (moves 0 … T-1).

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
            values.append(self._pool_value(feat))
        return torch.cat(values, dim=0)  # [T, 1]
