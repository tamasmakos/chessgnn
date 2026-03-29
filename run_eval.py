"""Small-scale real-life evaluation script."""
import warnings
warnings.filterwarnings("ignore")

import torch
import json

from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel, STHGATLikeModel
from chessgnn.eval import Evaluator

PUZZLE_CSV   = "input/lichess_db_puzzle.csv/lichess_db_puzzle.csv"
LABELS_JSONL = "output/distillation_labels.jsonl"
N_PUZZLES    = 50
K            = 3
# STHGAT rollout is O(legal_moves) per position — limit to avoid multi-minute wait
N_LABELS_STHGAT = 30
device = torch.device("cpu")

# ── Inspect STHGATLikeModel checkpoint to get hidden_channels ─────────
ckpt_s = torch.load("output/st_hgat_model.pt", map_location=device)
# global_gru.weight_ih_l0 shape is [3*H, 2*H] → H = shape[0]//3
if "global_gru.weight_ih_l0" in ckpt_s:
    H_s = ckpt_s["global_gru.weight_ih_l0"].shape[0] // 3
else:
    # fall back: convs.0.k_lin.piece.weight shape is [H, 10]
    H_s = ckpt_s["convs.0.k_lin.piece.weight"].shape[0]
layers_s = max(int(k.split(".")[1]) for k in ckpt_s if k.startswith("convs.")) + 1
print(f"STHGATLikeModel inferred: hidden_channels={H_s}, num_layers={layers_s}")

# ── Load GATEAUChessModel ─────────────────────────────────────────────
print("Loading GATEAUChessModel …")
builder_g = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
gateau = GATEAUChessModel(
    builder_g.get_metadata(),
    hidden_channels=128,
    num_layers=4,
    temporal_mode="global_gru",
)
gateau.load_state_dict(torch.load("output/gateau_distilled.pt", map_location=device))
gateau.eval()

# ── Load STHGATLikeModel ──────────────────────────────────────────────
print("Loading STHGATLikeModel …")
builder_s = ChessGraphBuilder(use_global_node=False, use_move_edges=False)
sthgat = STHGATLikeModel(builder_s.get_metadata(), hidden_channels=H_s, num_layers=layers_s)
sthgat.load_state_dict(ckpt_s)
sthgat.eval()

ev_g = Evaluator(gateau, device=device, use_global_node=True)
ev_s = Evaluator(sthgat,  device=device, use_global_node=False)

import tempfile, os

# ── Puzzle accuracy ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Puzzle accuracy  (first {N_PUZZLES} Lichess puzzles, ~rating 1000-2500)")
print(f"{'='*55}")
rg = ev_g.evaluate_puzzles(PUZZLE_CSV, n=N_PUZZLES)
rs = ev_s.evaluate_puzzles(PUZZLE_CSV, n=N_PUZZLES)
print(f"  GATEAUChessModel : {rg['accuracy']*100:.1f}%  ({rg['solved']}/{rg['count']})")
print(f"  STHGATLikeModel  : {rs['accuracy']*100:.1f}%  ({rs['solved']}/{rs['count']})")
print(f"  Random baseline  :  ~4.5%  (1/avg_legal_moves)")

# ── Engine agreement ──────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Engine agreement  (distillation_labels.jsonl, top-{K})")
print(f"{'='*55}")
eg = ev_g.evaluate_engine_agreement(LABELS_JSONL, k=K)
es = ev_s.evaluate_engine_agreement(LABELS_JSONL_SMALL, k=K)
print(f"  GATEAU  top-1: {eg['top1_acc']*100:.1f}%  top-{K}: {eg[f'top{K}_acc']*100:.1f}%  (n={eg['count']})")
print(f"  STHGAT  top-1: {es['top1_acc']*100:.1f}%  top-{K}: {es[f'top{K}_acc']*100:.1f}%  (n={es['count']}, limited sample)")

# ── Value correlation ─────────────────────────────────────────────────
print(f"\n{'='*55}")
print("Value correlation  (vs Stockfish win-probability)")
print(f"{'='*55}")
vg = ev_g.evaluate_value_correlation(LABELS_JSONL)
vs = ev_s.evaluate_value_correlation(LABELS_JSONL_SMALL)
print(f"  GATEAU  Pearson R={vg['pearson_r']:+.4f}   Spearman rho={vg['spearman_rho']:+.4f}   (n={vg['count']})")
print(f"  STHGAT  Pearson R={vs['pearson_r']:+.4f}   Spearman rho={vs['spearman_rho']:+.4f}   (n={vs['count']}, limited sample)")

# ── Reliability diagram ───────────────────────────────────────────────
stats = ev_g.reliability_diagram(LABELS_JSONL, "output/reliability_gateau.png", n_bins=10)
print(f"\nReliability diagram saved -> output/reliability_gateau.png")
print("Bins (center | mean_pred | mean_sf_wp | count):")
for s in stats:
    bar = "#" * min(s["count"], 20)
    print(f"  {s['bin_center']:.2f}  pred={s['mean_pred']:.3f}  sf={s['mean_target']:.3f}  n={s['count']:3d}  {bar}")

# ── Summary JSON ──────────────────────────────────────────────────────
summary = {
    "gateau": {**rg, **eg, **vg},
    "sthgat": {**rs, **es, **vs},
}
with open("output/eval_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nFull summary -> output/eval_results.json")
os.unlink(LABELS_JSONL_SMALL)
