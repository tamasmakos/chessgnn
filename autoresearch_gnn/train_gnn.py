import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters — agent entry point, change these freely
# ---------------------------------------------------------------------------

HIDDEN_DIM    = 256       # hidden width for GNN layers
NUM_LAYERS    = 5         # number of WeightedHGTConv layers
TEMPORAL_MODE = "none"    # "none" | "global_gru" | "node_gru"
LR            = 5e-3      # learning rate
WEIGHT_DECAY  = 1e-4      # AdamW weight decay
LAMBDA_Q      = 1.0       # policy (Q-head) cross-entropy loss weight
LAMBDA_V      = 0.5       # value regression loss weight
GRAD_CLIP     = 1.0       # max gradient norm (0 = disabled)
BATCH_SIZE    = 16        # positions per optimizer step (no gradient accumulation)
LOG_EVERY     = 10        # print a training log line every N steps
VAL_EVERY     = 0         # evaluate top1_agreement mid-run every N steps (0 = disabled, slow)

# ---------------------------------------------------------------------------
# Fixed training constants from prepare_gnn (do not change)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from autoresearch_gnn.prepare_gnn import (  # noqa: E402
    TIME_BUDGET,
    ELO_NORM_SF,
    TRAIN_JSONL,
    evaluate_top1_agreement,
    load_train_data,
)

# ---------------------------------------------------------------------------
# Model — WeightedHGTConv + GATEAUChessModel defined inline so you can freely
# modify the architecture without touching the production chessgnn/ package.
#
# The class interface used by prepare_gnn.evaluate_top1_agreement is:
#   model.forward_with_q(graph, elo_norm=float) -> (value[1,1], q[M], move_ei[2,M])
# ---------------------------------------------------------------------------

from torch.nn import ParameterDict
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import softmax


# Node feature dimensions produced by ChessGraphBuilder (fixed by graph contract)
_NODE_INPUT_DIMS: dict[str, int] = {"piece": 15, "square": 3, "global": 11}


class WeightedHGTConv(MessagePassing):
    """Heterogeneous Graph Transformer with Edge Weights."""

    def __init__(self, in_channels: int, out_channels: int, metadata: tuple, heads: int = 4):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.d_k = out_channels // heads

        node_types, edge_types = metadata
        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        self.a_lin = nn.ModuleDict()
        self.skip  = ParameterDict()

        for nt in node_types:
            self.k_lin[nt] = Linear(in_channels, out_channels)
            self.q_lin[nt] = Linear(in_channels, out_channels)
            self.v_lin[nt] = Linear(in_channels, out_channels)
            self.a_lin[nt] = Linear(out_channels, out_channels)
            self.skip[nt]  = (
                Linear(in_channels, out_channels) if in_channels != out_channels
                else nn.Identity()
            )

        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        self.relation_pri = nn.ParameterDict()
        from torch_geometric.nn.inits import glorot
        for et in edge_types:
            key = "__".join(et)
            self.relation_att[key] = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
            self.relation_msg[key] = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
            self.relation_pri[key] = nn.Parameter(torch.ones(1))
            glorot(self.relation_att[key])
            glorot(self.relation_msg[key])

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        k_dict, q_dict, v_dict = {}, {}, {}
        for nt, x in x_dict.items():
            k_dict[nt] = self.k_lin[nt](x).view(-1, self.heads, self.d_k)
            q_dict[nt] = self.q_lin[nt](x).view(-1, self.heads, self.d_k)
            v_dict[nt] = self.v_lin[nt](x).view(-1, self.heads, self.d_k)

        out_dict = {}
        for et, edge_index in edge_index_dict.items():
            src_type, _, dst_type = et
            key = "__".join(et)
            ew = None
            if edge_weight_dict and et in edge_weight_dict:
                ew = edge_weight_dict[et]
            out = self.propagate(
                edge_index,
                k=k_dict[src_type], q=q_dict[dst_type], v=v_dict[src_type],
                rel_att=self.relation_att[key],
                rel_msg=self.relation_msg[key],
                rel_pri=self.relation_pri[key],
                edge_weight=ew,
            )
            if dst_type not in out_dict:
                out_dict[dst_type] = out
            else:
                out_dict[dst_type] = out_dict[dst_type] + out

        for nt in out_dict:
            out_dict[nt] = self.a_lin[nt](out_dict[nt].view(-1, self.out_channels))
            out_dict[nt] = out_dict[nt] + self.skip[nt](x_dict[nt])
            out_dict[nt] = F.gelu(out_dict[nt])
        return out_dict

    def message(self, k_j, q_i, v_j, rel_att, r