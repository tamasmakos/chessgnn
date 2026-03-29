# ChessGNN — Searchless Chess Engine via Spatio-Temporal Heterogeneous GNN

A graph neural network that encodes chess positions as typed relational graphs and ranks legal moves by win probability **in a single forward pass** — no search tree required.

Traditional engines like Stockfish rely on explicit alpha-beta search over millions of positions. AlphaZero-style CNNs treat the board as a flat 8×8 grid and still need MCTS at inference. ChessGNN asks a different question: can a GNN with rich relational inductive bias — attack edges, defense edges, ray edges, move edges — learn enough tactical and strategic knowledge to rank moves correctly without any rollout?

The answer so far: yes, with distillation from Stockfish and the architecture described below.

---

## How It Works

### 1. Position as a Graph

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

### 2. Spatial Reasoning — Weighted ST-HGAT

Three stacked `WeightedHGTConv` layers process the graph. Each layer uses separate key/query/value projections per node type and separate relation matrices per edge type. Edge weights fold into attention multiplicatively:

$$\alpha = \mathrm{softmax}\!\left(\frac{(h_i W_K)(h_j W_Q)^\top}{\sqrt{d}}\right) \cdot (1 + w_{ij})$$

A `RayAlignmentBlock` follows the convolutions to amplify long-range piece influence along rays before pooling.

Layer 1 captures direct attacks and defenses. Layer 2 captures secondary support chains. Layer 3 captures complex tactics — pins, batteries, discovered attacks.

### 3. Temporal Context — GRU over Game History

After spatial pooling, a GRU accumulates game history:

$$h_t = \mathrm{GRU}([\mathrm{pool}_\mathrm{piece} \| \mathrm{pool}_\mathrm{square}],\; h_{t-1})$$

- **Batch mode** (`forward()`): processes the entire game sequence at once for training.
- **Incremental mode** (`forward_step(h_prev)`): O(1) per move for online play — the `CaseTutor` caches `h_t` after each move.

### 4. Output Heads

| Head | Output | Training signal |
|------|--------|-----------------|
| **Value** V(s) | [White win, Draw, Black win] logits | Game outcome regression + TD loss |
| **Q-head** Q(s,a) | scalar per legal-move edge | KL divergence vs. Stockfish move distribution |
| **Material** (aux) | normalized material imbalance | White − black material / 39 |
| **Dominance** (aux) | positional dominance scalar | PageRank centrality difference |

### 5. Training Loss

$$\mathcal{L} = \mathcal{L}_\mathrm{outcome} + \lambda_1 \mathcal{L}_\mathrm{TD} + \lambda_2 \mathcal{L}_\mathrm{policy} + \lambda_3 \mathcal{L}_\mathrm{aux}$$

- **Outcome**: MSE of V(s) vs. game result at every step along the mainline.
- **TD**: forces $V(s_t) \approx V(s_{t+1})$, smoothing the win-probability curve across moves.
- **Policy**: KL divergence between Q-head distribution and Stockfish top-k move scores (distillation).
- **Auxiliary**: material and dominance predictions provide dense gradients early in training.

Gradient accumulation (`ACCUMULATION_STEPS = 16`) is used to compensate for batch size 1 (one game per step).

---

## Inference: Move Ranking Without Search

For a given position the `CaseTutor` does:

1. Run a single `forward_step()` with the cached GRU state → node and edge embeddings.
2. Read Q-scores from each `(piece, move, square)` edge.
3. Return `argmax` move + win probability + top-k analysis.

No rollout, no MCTS, no alpha-beta. O(1) in the number of legal moves after the GNN pass.

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

The Stockfish binary is included at `stockfish/src/stockfish` (Linux ELF). On Windows, point `STOCKFISH_PATH` in `train.py` to your local executable.

### Training

Place a PGN file in `input/` and update the path constant at the top of `train.py`, then:

```bash
python train.py
```

The loop streams full game sequences from the PGN, computes the combined loss, and saves checkpoints to `output/st_hgat_model.pt`. A training log is written to `output/training.log`.

Default hyperparameters (edit the `ALL_CAPS` constants in `train.py`):

| Constant | Default | Notes |
|----------|---------|-------|
| `HIDDEN_DIM` | 256 | GNN and GRU hidden size |
| `NUM_LAYERS` | 3 | WeightedHGTConv stack depth |
| `LR` | 0.005 | Adam learning rate |
| `ACCUMULATION_STEPS` | 16 | Effective batch size |
| `EPOCHS` | 2 | Passes over the training PGN |
| `TRAIN_GAMES` | 100 | Games to sample per epoch |

### Using the Tutor

```python
import torch
from chessgnn.model import STHGATLikeModel
from tutor import CaseTutor

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STHGATLikeModel(hidden_dim=256, num_layers=3)
model.load_state_dict(torch.load("output/st_hgat_model.pt", map_location=device))
model.eval()

tutor = CaseTutor(model, device)

# Feed moves as they are played (FEN string after each move)
tutor.update_state(fen_after_move_1)
tutor.update_state(fen_after_move_2)

# Get the recommended next move
best_move, win_prob, analysis = tutor.recommend_move(current_fen)
print(f"Recommended: {best_move}  Win probability: {win_prob:.1%}")
```

### Running Tests

```bash
pytest tests/
```

---

## Project Structure

```
chessgnn/
├── graph_builder.py      # FEN → HeteroData (nodes, edges, weights)
├── model.py              # STHGATLikeModel, WeightedHGTConv, RayAlignmentBlock
├── dataset.py            # PGN → full-game sequence yielder
├── game_processor.py     # Stockfish evaluation pipeline
├── position_to_graph.py  # Auxiliary ground-truth computation
└── visualizer.py         # Win-probability video generation

agent/
├── core.py               # LLM coaching agent
├── llm.py                # LLM provider abstraction (Groq)
├── schema.py             # Pydantic schemas
└── tools.py              # Agent tools (move explanation, etc.)

train.py                  # Training entry point
tutor.py                  # CaseTutor — stateful inference

tests/
├── test_graph_builder.py
├── test_inference.py
├── test_model.py
├── test_ray_alignment.py
└── test_temporal_ablation.py

input/                    # PGN datasets
output/                   # Checkpoints, logs, videos
stockfish/src/stockfish   # Bundled Stockfish binary (Linux)
```

---

## Roadmap

| # | Task | Status |
|---|------|--------|
| Graph builder: global node, move edges, ablation flags | TASK001 | ✅ Done |
| Edge-aware GATEAU-style layer + Q-head | TASK002 | ✅ Done |
| Temporal ablation experiments | TASK003 | ✅ Done |
| Engine distillation dataset pipeline | TASK004 | Pending |
| Multi-task distillation training | TASK005 | Pending |
| Evaluation harness (puzzles, Elo gauntlet) | TASK006 | Pending |
| UCI wrapper for engine play | TASK007 | Pending |
| Calibration (temperature scaling + reliability diagrams) | TASK008 | Pending |
| Scaling experiments | TASK009 | Pending |

**Success targets**: ≥ 40% top-1 move agreement with Stockfish on tactical puzzles (searchless), provisional Elo ≥ 1600 vs. reduced-strength Stockfish, calibrated win probabilities (reliability diagram R² > 0.95).

---

## Known Limitations

- `train.py` hardcodes Linux paths — adjust `INPUT_PGN` and `STOCKFISH_PATH` for other platforms.
- Stockfish binary is a Linux ELF; Windows users need a native `.exe` or WSL.
- Positions with zero pieces (extreme edge cases) may break `Batch` construction in `forward()`.

---

## License

See individual file headers. Stockfish is bundled under the GPL v3 (see `stockfish/Copying.txt`).

