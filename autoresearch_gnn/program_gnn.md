# autoresearch_gnn

Self-improving research loop for the ChessGNN — the GNN equivalent of [autoresearch](../autoresearch/program.md).

## Setup

1. **Agree on a run tag** — propose a tag based on today's date (e.g. `apr9`). The branch `autoresearch_gnn/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch_gnn/<tag>` from current master.
3. **Read the in-scope files**:
   - `autoresearch_gnn/prepare_gnn.py` — fixed constants, data loading, graph building, evaluation harness. Do not modify.
   - `autoresearch_gnn/train_gnn.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop.
   - `chessgnn/graph_builder.py` — the graph construction code. Read for context; do not modify.
   - `chessgnn/model.py` — the production model for reference. Do not modify.
4. **Verify data files exist**:
   - `output/game_labels_elo1600_probe1024.jsonl` — training labels (1024 positions)
   - `output/distillation_labels_val1k.jsonl` — fixed val set (1000 positions)
   - If missing, tell the human to run:
     `python chessgnn/distillation_pipeline.py` or restore from checkpoint.
5. **Run the smoke test**: `cd /workspaces/chessgnn && python autoresearch_gnn/prepare_gnn.py`
   This should print `Val positions loaded: N` and `Train positions loaded: M` without error.
6. **Initialize results.tsv**: Create `autoresearch_gnn/results.tsv` with just the header row.
7. **Confirm and go**.

---

## Experimentation

Each experiment runs for a **fixed wall-clock budget of 10 minutes** (`TIME_BUDGET = 600` seconds of actual training). Launch with:

```
cd /workspaces/chessgnn
python autoresearch_gnn/train_gnn.py > autoresearch_gnn/run.log 2>&1
```

**What you CAN modify** (`train_gnn.py` only — the sole source of truth for model behaviour):
- Model architecture (layers, widths, heads, attention mechanism, new modules)
- Optimizer type, learning rate schedule, batch size, weight decay
- Training loop (gradient accumulation, mixed precision, curriculum)
- Hyperparameter constants at the top of the file (ALL_CAPS variables)
- The model class itself — rewrite it completely if you have a better idea

**What you CANNOT modify**:
- `prepare_gnn.py` — the evaluation harness is fixed.
- `chessgnn/` package files — they are production code.
- The output format printed by `train_gnn.py`: the `---` summary block keys must stay the same so results can be parsed consistently.

**The goal**: maximise `top1_agreement` (fraction of positions where the model's top move matches Stockfish's best move, evaluated on 1000 fixed positions). Higher is better.

**VRAM** is a soft constraint. Moderate increases are fine for meaningful gains; don't blow up memory dramatically for tiny improvements.

**Simplicity criterion**: Same as autoresearch — a small improvement with ugly complexity is worth less than the raw number suggests. Removing code and getting equal or better performance is a win.

**The first run** should always be the unmodified baseline.

---

## Output format

When training completes, the script prints:

```
---
top1_agreement:   0.349000
training_seconds: 600.3
peak_vram_mb:     412.0
num_steps:        890
num_params_M:     3.72
```

Extract the key metric:

```
grep "^top1_agreement:" autoresearch_gnn/run.log
```

---

## Logging results

Log to `autoresearch_gnn/results.tsv` (tab-separated, NOT commas):

```
commit	top1_agreement	memory_gb	status	description
```

1. 7-char git commit hash
2. `top1_agreement` achieved (use `0.000000` for crashes)
3. Peak memory in GB, rounded to 1 decimal (divide `peak_vram_mb` by 1024; use `0.0` for crashes)
4. Status: `keep`, `discard`, or `crash`
5. Short description of what the experiment tried

Example:

```
commit	top1_agreement	memory_gb	status	description
a1b2c3d	0.349000	0.4	keep	baseline GATEAUChessModel h192 l5
b2c3d4e	0.362000	0.4	keep	increase LR to 0.01
c3d4e5f	0.341000	0.4	discard	switch value head to sigmoid
d4e5f6g	0.000000	0.0	crash	add node dropout (NaN loss)
```

**Keep or discard?** Keep the change if `top1_agreement` improves. Discard if it doesn't improve. When in doubt (very small delta), prefer the simpler code.

---

## The experiment loop

LOOP FOREVER:

1. Inspect the current git state (branch, last commit).
2. Pick an experiment idea (architecture, optimizer, hyperparams, training schedule).
3. Edit `train_gnn.py`.
4. `git commit -am "brief description"`
5. Run: `python autoresearch_gnn/train_gnn.py > autoresearch_gnn/run.log 2>&1`
6. Read results: `grep "^top1_agreement:\|^peak_vram_mb:" autoresearch_gnn/run.log`
7. Decide keep/discard. If discard, `git revert HEAD --no-edit` (or `git checkout HEAD~1 -- autoresearch_gnn/train_gnn.py`).
8. Append row to `autoresearch_gnn/results.tsv`.
9. Repeat.

---

## Baseline expectations

Based on current project state (see `memory-bank/progress.md`):
- `GATEAUChessModel h192 l5` (the default baseline in `train_gnn.py`) achieves ~34–36% top-1 Stockfish agreement on 1k positions after training on the probe1024 dataset.
- The 1024-position training set is small; overfitting is expected. Regularisation and data augmentation are worth trying.
- Primary bottleneck is data quantity, not model size — but architecture improvements can still help.

## Ideas to try

- Learning rate schedule (warmup + cosine decay)
- Larger batch, gradient accumulation
- Label smoothing on the policy target
- Mixed softmax temperature for soft policy targets
- Additional GNN layers or wider hidden dims
- Global node feature engineering
- Different aggregation (sum vs mean)
- Dropout / DropEdge regularisation
- Pretrain on value objective only, then fine-tune Q-head
