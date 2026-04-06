# ChessGNN — Searchless Chess Engine via Spatio-Temporal Heterogeneous GNN
> **The Broken Mirror Thesis: Graph-Native Intelligence**
> Modern chess analytics is trapped in a representational error. By treating Stockfish’s scalar evaluations as the ultimate ground truth, the industry has built an ecosystem that fails to capture the structural and relational reality of human chess cognition. 
> 
> *ChessGNN* is built on a fundamentally different ontology: **Graph-Native Intelligence**. Instead of brute-forcing millions of move sequences via alpha-beta search, we encode chess as it is actually perceived by human experts—as a temporal sequence of graph transformations. Pieces are nodes; attacks, defenses, and structural interactions are directed, weighted edges. By aligning our model's inductive bias (Heterogeneous Graph Transformers) with the relational reality of the board, ChessGNN moves beyond mere centipawn scores. It lays the architectural groundwork for quantifying *practical difficulty*, performing *structural fingerprinting*, and achieving powerful searchless inference in a single forward pass.


A graph neural network that encodes chess positions as typed relational graphs and ranks legal moves by win probability **in a single forward pass** — no search tree required.

Traditional engines like Stockfish rely on explicit alpha-beta search over millions of positions. AlphaZero-style CNNs treat the board as a flat 8×8 grid and still need MCTS at inference. ChessGNN asks a different question: can a GNN with rich relational inductive bias — attack edges, defense edges, ray edges, move edges — learn enough tactical and strategic knowledge to rank moves correctly without any rollout?

The answer so far: yes, with distillation from Stockfish and the architecture described below.

---

## How It Works

The architecture is divided into two distinct phases: **Structural (Spatial) Reasoning** to understand the board in a single moment, followed by **Sequential (Temporal) Context** to understand how threats and dynamics evolve across moves.

```mermaid
graph TD
    FEN[FEN String] -->|Graph Builder| Graph[Heterogeneous Graph\nPieces & Squares]
    
    subgraph Structural [1. Structural Spatial Reasoning]
        Graph --> HGT1[Weighted HGT Layer 1]
        HGT1 --> HGT2[Weighted HGT Layer 2]
        HGT2 --> HGT3[Weighted HGT Layer 3]
        HGT3 --> HGT4[Weighted HGT Layer 4]
    end

    subgraph Sequential [2. Sequential Temporal Context]
        HGT4 -->|Node Embeddings| GRU[GRU / Node-GRU Cell]
        Cache[(KV Cache\nPrevious States)] --> GRU
        GRU -->|Updates| Cache
    end

    subgraph Output [3. Output Heads]
        GRU --> Pool[Pooling]
        Pool -->|Elo Embedding| ValueHead[Value Head V(s)]
        
        GRU -->|Move Edges + Elo| QHead[Q-Head Q(s,a)]
    end
    
    ValueHead --> WinProb[Win Probability]
    QHead --> MoveLogits[Move Logits & Ranking]
```

### 1. Structural Phase: Position as a Graph

Every chess position is converted to a `HeteroData` graph by `ChessGraphBuilder.fen_to_graph()`:

| Node type | Count | Features |
|-----------|-------|----------|
| `piece` | ≤ 32 | type (one-hot 6), color, value, file, rank |
| `square` | 64 | file/7, rank/7, is_occupied |
| `global` | 1 | material balance, side to move, castling rights, game phase |

| Edge type | Semantics |
|-----------|-----------|
| `(piece, on, square)` / `(square, occupied_by, piece)` | piece location |
| `(piece, interacts, piece)` | attack / defend with piece-value weights |
| `(piece, ray, piece)` | long-range diagonal / file / rank influence |
| `(square, adjacent, square)` | Chebyshev-1 king-move adjacency |
| `(piece, move, square)` | one edge per legal move (used by Q-head) |

### 2. Structural Phase: Spatial Reasoning — Weighted HGT

`GATEAUChessModel` stacks `WeightedHGTConv` layers (default: 4). Each layer uses separate key/query/value projections per node type and separate relation matrices per edge type. Edge weights fold into attention multiplicatively:

$$\alpha = \mathrm{softmax}\!\left(\frac{(h_i W_K)(h_j W_Q)^\top}{\sqrt{d}}\right) \cdot (1 + w_{ij})$$

LayerNorm is applied after each convolution. Layer 1 captures direct attacks and defenses. Layer 2 captures secondary support chains. Layer 3+ captures complex tactics — pins, batteries, discovered attacks.

### 3. Sequential Phase: Temporal Context — Three Modes

`GATEAUChessModel` supports three temporal modes, selectable via `temporal_mode`:

| Mode | Description |
|------|-------------|
| `"none"` | No recurrence; value from spatial pooling only |
| `"global_gru"` | GRU over $[\mathrm{pool}_\mathrm{piece} \| \mathrm{pool}_\mathrm{square}]$ per step |
| `"node_gru"` | Separate `GRUCell` per node type; each piece/square carries its own hidden state |

$$h_t = \mathrm{GRU}([\mathrm{pool}_\mathrm{piece} \| \mathrm{pool}_\mathrm{square}],\; h_{t-1})$$

- **Sequence training** (`forward_sequence()`): iterates positions maintaining a `KVCache`.
- **Incremental inference** (`forward_step(graph, cache)`): O(1) per move for online play.

### 4. Output Heads

| Head | Output | Training signal |
|------|--------|-----------------|
| **Value** V(s) | tanh scalar in [−1, 1] | Stockfish win-probability (distillation) or game outcome |
| **Q-head** Q(s,a) | scalar per legal-move edge | KL divergence vs. Stockfish top-k move distribution |
| **Material** (aux) | normalized material imbalance | White − black material / 39 (legacy `STHGATLikeModel`) |
| **Dominance** (aux) | positional dominance scalar | PageRank centrality difference (legacy `STHGATLikeModel`) |

### 5. Training Loss

Two training regimes are supported:

**Baseline pretraining** (game-outcome regression on PGN, `train.py`):

$$\mathcal{L} = \mathcal{L}_\mathrm{outcome} + \lambda_1 \mathcal{L}_\mathrm{TD} + \lambda_2 \mathcal{L}_\mathrm{aux}$$

**Distillation fine-tuning** (Stockfish-labelled positions, `distill_train.py`):

$$\mathcal{L} = \lambda_V \cdot \mathcal{L}_\mathrm{value} + \lambda_Q \cdot \mathcal{L}_\mathrm{policy}$$

- **Value**: MSE between V(s) and Stockfish win probability $wp = 1/(1+e^{-cp/400})$ for each position.
- **Policy**: KL divergence between Q-head soft distribution and Stockfish top-k move scores.

Gradient accumulation is used in both regimes to compensate for batch size 1.

---

## Inference: Move Ranking Without Search

For a given position the `CaseTutor` does:

1. Run a single `forward_step()` with the cached GRU state → node and edge embeddings.
2. Read Q-scores from each `(piece, move, square)` edge.
3. Return `argmax` move + calibrated win probability + top-k ranking + uncertainty.

No rollout, no MCTS, no alpha-beta. O(1) in the number of legal moves after the GNN pass.

`recommend_move()` returns a 4-tuple `(best_move, win_prob, ranking, uncertainty)` where `uncertainty` is the normalised entropy $H/\log(M)$ of the Q-score softmax distribution over all $M$ legal moves (0 = certain, 1 = uniform).

---

## Quick Start

### Prerequisites

```
Python 3.10+
PyTorch (CUDA optional)
PyTorch Geometric
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

The Stockfish binary is included at `stockfish/src/stockfish` (Linux ELF). On Windows, point `STOCKFISH_PATH` to your local executable.

### Baseline Pretraining

Train `STHGATLikeModel` on raw game outcomes from a PGN file:

```bash
python train.py
```

Streams full game sequences, computes outcome + TD + auxiliary loss, saves to `output/st_hgat_model.pt`. All hyperparameters are `ALL_CAPS` constants at the top of the file:

| Constant | Default | Notes |
|----------|---------|-------|
| `HIDDEN_DIM` | 256 | GNN and GRU hidden size |
| `NUM_LAYERS` | 3 | `WeightedHGTConv` stack depth |
| `LR` | 0.005 | Adam learning rate |
| `ACCUMULATION_STEPS` | 16 | Effective batch size |
| `EPOCHS` | 2 | Passes over the training PGN |
| `TRAIN_GAMES` | 100 | Games to sample per epoch |

### Distillation Training

Train `GATEAUChessModel` directly on Stockfish-labelled positions using dual loss (value MSE + policy KL). Two modes are supported: **offline** (pre-generate labels once) and **online** (Stockfish labels produced in a background thread while training proceeds).

**Offline — Step 1: Generate labelled positions:**

```bash
python -m chessgnn.distillation_pipeline \
  --pgn input/lichess_db_standard_rated_2013-01.pgn \
  --out output/distillation_labels.jsonl \
  --stockfish stockfish/src/stockfish \
  --positions 10000 --depth 12 --multipv 5
```

| Flag | Default | Notes |
|------|---------|-------|
| `--positions` | 10000 | Total positions to label |
| `--depth` | 12 | Stockfish search depth (higher = slower, better labels) |
| `--multipv` | 5 | Top moves per position for policy target |
| `--min-move` | 10 | Skip opening moves (low information) |
| `--max-move` | 100 | Skip endgame tablebase territory |

**Offline — Step 2: Train:**

```bash
python distill_train.py
```

Reads the JSONL labels, builds move-edge graphs on the fly, and trains with gradient accumulation. Saves the best checkpoint (by top-1 move accuracy on the validation split) to `output/gateau_distilled.pt`. Key constants in `distill_train.py`:

| Constant | Default | Notes |
|----------|---------|-------|
| `HIDDEN_DIM` | 128 | GNN hidden size |
| `NUM_LAYERS` | 4 | `WeightedHGTConv` stack depth |
| `TEMPORAL_MODE` | `"global_gru"` | `"none"` / `"global_gru"` / `"node_gru"` |
| `LAMBDA_V` | 1.0 | Weight on value MSE loss |
| `LAMBDA_Q` | 1.0 | Weight on policy KL loss |
| `TEMPERATURE` | 1.0 | Softmax temperature for policy target |
| `VALIDATION_SPLIT` | 0.1 | Fraction of data held out for validation |
| `PRETRAIN_CHECKPOINT` | `None` | Path to load pretrained weights before distillation |

**Online — label and train simultaneously:**

`OnlineDistillationDataset` (`chessgnn/online_distillation.py`) runs a daemon producer thread that walks the PGN, calls Stockfish, and pushes ready samples into a bounded queue. The DataLoader consumes them — so Stockfish I/O and backprop overlap in time with no intermediate JSONL file required.

```bash
python run_experiment.py --config configs/tiny_online.json
```

Online mode config keys (in addition to the standard run_experiment schema):

| Key | Notes |
|-----|-------|
| `"online": true` | Enables online mode |
| `"pgn_path"` | Lichess PGN source |
| `"stockfish_path"` | Path to Stockfish binary |
| `"total_positions"` | Positions to generate per epoch |
| `"sf_depth"` | Stockfish depth (8 is fast; 12 is high quality) |
| `"multipv_k"` | Top moves per position |
| `"buffer_size"` | Queue depth (default 128) |
| `"val_jsonl"` | Fixed JSONL for validation (since training data is streamed) |

> **Constraint**: online mode requires `num_workers=0` in the DataLoader because Stockfish is a subprocess and is not fork-safe. This is enforced automatically.

### Using the Tutor

```python
import torch
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.model import GATEAUChessModel
from tutor import CaseTutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
metadata = builder.get_metadata()

model = GATEAUChessModel(metadata, hidden_channels=128, num_layers=4, temporal_mode="global_gru")
model.load_state_dict(torch.load("output/gateau_distilled.pt", map_location=device))
model.eval()

tutor = CaseTutor(model, device)

# Optional: load a calibration sidecar for temperature-scaled probabilities
from chessgnn.calibration import TemperatureScaler
scaler = TemperatureScaler.load("output/gateau_distilled.pt.calib.json")
tutor.set_calibration(scaler)

# Feed moves as they are played (FEN string after each move)
tutor.update_state(fen_after_move_1)
tutor.update_state(fen_after_move_2)

# Get the recommended next move
best_move, win_prob, ranking, uncertainty = tutor.recommend_move(current_fen)
print(f"Recommended: {best_move}  Win probability: {win_prob:.1%}  Uncertainty: {uncertainty:.2f}")
```

### Analytics from a Lichess game

You can analyse a public Lichess game directly without downloading a PGN first:

```bash
python show_analytics.py --model output/gateau_distilled.pt --lichess-game https://lichess.org/j1dkb5dw
```

If you need authenticated access (for example private games available to your
account), export a token in `LICHESS_API_TOKEN` first and `show_analytics.py`
will attach it automatically when calling the Lichess API.

### Evaluation Harness

Evaluate a trained model across three metrics using `chessgnn/eval.py`:

```bash
python -m chessgnn.eval \
  --model output/gateau_distilled.pt \
  --positions output/distillation_labels.jsonl \
  --puzzles input/lichess_db_puzzle.csv \
  --out output/eval_results.json
```

| Metric | Method | Description |
|--------|--------|-------------|
| **Top-1 / Top-k engine agreement** | `evaluate_engine_agreement()` | Fraction of positions where model's best move matches Stockfish's top-k |
| **Puzzle accuracy** | `evaluate_puzzles()` | First-move accuracy on Lichess puzzle CSV |
| **Value correlation** | `evaluate_value_correlation()` | Pearson R and Spearman ρ vs Stockfish win probability |

For multi-model comparisons use `compare_models(models_dict, ...)` which returns `{name: {metric: value}}`.

### UCI Engine

`uci_engine.py` exposes the GNN as a standard UCI engine, compatible with any chess GUI (Arena, Cutechess-CLI, etc.):

```bash
# Searchless mode (argmax Q-head)
python uci_engine.py

# With a calibrated win-probability sidecar
python uci_engine.py \
  --model output/gateau_distilled.pt \
  --calib output/gateau_distilled.pt.calib.json
```

The `go` command triggers a single forward pass and immediately emits `info score cp <cp> pv <move>` followed by `bestmove`. Win probability is converted to centipawns via $\mathrm{cp} = 111.714 \cdot \tanh(1.562 \cdot (p - 0.5))$.

Supported UCI commands: `uci`, `isready`, `ucinewgame`, `position` (startpos/fen + moves), `go`, `stop`, `ponderhit`, `quit`.

### Calibration

Fit a temperature scalar $T$ on a held-out set of engine-evaluated positions so that predicted win probabilities match empirical win rates:

```bash
python calibrate.py \
  --model output/gateau_distilled.pt \
  --positions output/distillation_labels.jsonl
```

Outputs a `.calib.json` sidecar (same stem as the checkpoint) and two reliability-diagram PNGs (before/after calibration). The `TemperatureScaler` minimises soft NLL via L-BFGS-B.

```python
from chessgnn.calibration import TemperatureScaler, reliability_diagram

scaler = TemperatureScaler()
scaler.fit(logits, targets)          # optimise T on calibration set
print(scaler.ece(probs, targets))    # Expected Calibration Error
scaler.save("model.calib.json")

fig = reliability_diagram(probs, targets)
fig.savefig("reliability.png")
```

### Running Tests

```bash
pytest tests/
```

172 tests across all modules. All pass on the current codebase.


---

## Project Structure

```
chessgnn/
├── graph_builder.py           # FEN → HeteroData (nodes, edges, weights)
├── model.py                   # GATEAUChessModel, STHGATLikeModel, WeightedHGTConv, KVCache
├── dataset.py                 # PGN → full-game sequence yielder
├── distillation_pipeline.py   # PGN sampler + Stockfish MultiPV evaluator + JSONL I/O
├── distillation_dataset.py    # DistillationDataset, soft_policy_target()
├── online_distillation.py     # OnlineDistillationDataset — streaming Stockfish labels
├── eval.py                    # Evaluator: puzzle accuracy, engine agreement, value correlation
├── calibration.py             # TemperatureScaler, reliability_diagram
├── game_processor.py          # Game-level Stockfish evaluation for visualization
├── position_to_graph.py       # Auxiliary ground-truth computation
└── visualizer.py              # Win-probability video generation

agent/
├── core.py               # LLM coaching agent
├── llm.py                # LLM provider abstraction (Groq)
├── schema.py             # Pydantic schemas
└── tools.py              # Agent tools (move explanation, etc.)

train.py                  # Baseline pretraining (PGN game outcomes)
distill_train.py          # Distillation training (Stockfish labels, dual loss)
run_experiment.py         # Scaling sweep runner — reads JSON config, logs to CSV
tutor.py                  # CaseTutor — stateful inference with uncertainty
uci_engine.py             # UCI wrapper for engine gauntlets and GUIs
calibrate.py              # CLI: fit TemperatureScaler and save .calib.json sidecar

tests/
├── test_graph_builder.py
├── test_inference.py
├── test_model.py
├── test_ray_alignment.py
├── test_temporal_ablation.py
├── test_distillation_pipeline.py
├── test_distillation_dataset.py
├── test_eval_harness.py
└── test_calibration.py

configs/
├── tiny_test.json       # h32/l2, 15 epochs, offline
├── tiny_online.json     # h32/l2, 1 epoch, online streaming
├── medium.json          # h64/l3, 20 epochs, offline
├── small_long.json      # h64/l3, 50 epochs, offline (convergence check)
├── sweep_d500.json      # architecture sweep over 500 positions
└── sweep_d5k.json       # architecture sweep over 5 000 positions

input/                    # PGN datasets and Lichess puzzle CSV
output/                   # Checkpoints, JSONL labels, .calib.json sidecars, reliability PNGs
stockfish/src/stockfish   # Bundled Stockfish binary (Linux)
```

---

## Roadmap

| Task | Description | Status |
|------|-------------|--------|
| TASK001 | Graph builder: global node, move edges, ablation flags | ✅ Done |
| TASK002 | Edge-aware GATEAU-style layer + Q-head | ✅ Done |
| TASK003 | Temporal ablation experiments | ✅ Done |
| TASK004 | Engine distillation dataset pipeline | ✅ Done |
| TASK005 | Multi-task distillation training | ✅ Done |
| TASK006 | Evaluation harness (puzzles, engine agreement, value correlation) | ✅ Done |
| TASK007 | UCI wrapper for engine play | ✅ Done |
| TASK008 | Calibration (temperature scaling + reliability diagrams) | ✅ Done |
| TASK009 | Scaling experiments | 🔄 In progress |

**Current results** (2 564 positions, depth 8, CPU + CUDA):
- `gateau_tiny_online` (h32/l2, 1 epoch): **27% puzzle accuracy**, 22.5% top-1 engine agreement, Pearson R=0.33
- `gateau_medium` (h64/l3, 3 epochs so far): **32%+ val top-1** and climbing — loss still decreasing steadily

**Success targets**: ≥ 40% top-1 move agreement with Stockfish on tactical puzzles (searchless), provisional Elo ≥ 1600 vs. reduced-strength Stockfish, calibrated win probabilities (reliability diagram R² > 0.95).

---

## Known Limitations

- `train.py` and `distill_train.py` hardcode Linux paths — adjust `PGN_FILE` and `STOCKFISH_PATH` for other platforms.
- Stockfish binary is a Linux ELF; Windows users need a native `.exe` or WSL.
- `OnlineDistillationDataset` requires `num_workers=0` in the DataLoader (Stockfish subprocess is not fork-safe).
- `DistillationDataset` loads the full JSONL into memory; for datasets larger than ~2M positions, consider switching to memory-mapped or SQLite storage.
- Positions with zero pieces (extreme edge cases) may break `Batch` construction in `STHGATLikeModel.forward()`.
- Positions with zero legal moves (checkmate/stalemate mid-sequence) produce empty `policy_target` tensors; both training and validation loops skip these automatically.

---

## License

See individual file headers. Stockfish is bundled under the GPL v3 (see `stockfish/Copying.txt`).
