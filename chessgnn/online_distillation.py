"""Online distillation dataset: labels positions in a background thread.

The producer thread walks the Lichess PGN, calls Stockfish on each sampled
FEN, and pushes ready-to-use dicts into a bounded queue.  The consumer
(DataLoader) pulls from the queue and yields graph samples — so Stockfish
I/O and GNN forward/backward overlap in time.

Typical usage (inside run_experiment.py or a custom training script)::

    from chessgnn.online_distillation import OnlineDistillationDataset
    from chessgnn.distillation_dataset import distillation_collate

    ds = OnlineDistillationDataset(
        pgn_path="input/lichess_db_standard_rated_2013-01.pgn",
        stockfish_path="stockfish/src/stockfish",
        total_positions=5000,
        depth=8,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=distillation_collate)
    for batch in loader:
        ...  # train as normal
"""

import logging
import math
import queue
import threading
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from .distillation_dataset import soft_policy_target
from .distillation_pipeline import evaluate_positions, sample_positions_from_pgn
from .graph_builder import ChessGraphBuilder

logger = logging.getLogger(__name__)


class OnlineDistillationDataset(IterableDataset):
    """IterableDataset that produces labelled graph samples on-the-fly.

    A single daemon thread runs:
        PGN positions → Stockfish evaluation → queue

    The DataLoader worker calls ``__iter__`` which drains that queue,
    builds graphs, and yields sample dicts with the same schema as
    :class:`chessgnn.distillation_dataset.DistillationDataset`.

    Notes
    -----
    - ``num_workers=0`` in DataLoader is required (single-process), because
      the producer thread spawns its own subprocess (Stockfish via ``popen``).
    - ``__len__`` returns *total_positions* so tqdm progress bars work.
    """

    def __init__(
        self,
        pgn_path: str,
        stockfish_path: str,
        total_positions: int,
        depth: int = 8,
        multipv_k: int = 5,
        buffer_size: int = 128,
        temperature: float = 1.0,
        graph_builder: ChessGraphBuilder | None = None,
        min_move: int = 10,
        max_move: int = 100,
        num_sf_workers: int = 4,
    ) -> None:
        self.pgn_path = pgn_path
        self.stockfish_path = stockfish_path
        self.total_positions = total_positions
        self.depth = depth
        self.multipv_k = multipv_k
        self.buffer_size = buffer_size
        self.temperature = temperature
        self.builder = graph_builder or ChessGraphBuilder(
            use_global_node=True, use_move_edges=True
        )
        self.min_move = min_move
        self.max_move = max_move
        self.num_sf_workers = num_sf_workers

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        # Phase 1: sample FENs up-front (fast – PGN parsing only, no Stockfish)
        fens = list(sample_positions_from_pgn(
            self.pgn_path,
            self.total_positions,
            min_move=self.min_move,
            max_move=self.max_move,
        ))

        # Phase 2: split FENs evenly among N workers
        n_workers = min(self.num_sf_workers, len(fens))
        shard_size = math.ceil(len(fens) / n_workers)
        shards = [
            fens[i * shard_size:(i + 1) * shard_size]
            for i in range(n_workers)
        ]
        shards = [s for s in shards if s]

        out_q: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        _SENTINEL = object()

        def _worker(shard: list[str]) -> None:
            # Each worker owns its own Stockfish engine and graph builder.
            builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
            try:
                for label in evaluate_positions(
                    iter(shard),
                    self.stockfish_path,
                    depth=self.depth,
                    multipv_k=self.multipv_k,
                ):
                    fen = label["fen"]
                    try:
                        graph = builder.fen_to_graph(fen)
                    except Exception as exc:
                        logger.debug("Graph build failed for %s: %s", fen, exc)
                        continue

                    num_legal: int = graph["piece", "move", "square"].edge_index.shape[1]
                    if num_legal == 0:
                        continue

                    value_target = torch.tensor(
                        [2.0 * label["eval_wp"] - 1.0], dtype=torch.float32
                    )
                    policy_target = soft_policy_target(
                        label["top_k_moves"],
                        fen,
                        num_legal,
                        temperature=self.temperature,
                    )
                    out_q.put({
                        "graph": graph,
                        "value_target": value_target,
                        "policy_target": policy_target,
                        "num_legal_moves": num_legal,
                    })
            except Exception as exc:
                logger.error("OnlineDistillationDataset worker error: %s", exc)
            finally:
                out_q.put(_SENTINEL)

        threads = [
            threading.Thread(target=_worker, args=(s,), daemon=True)
            for s in shards
        ]
        for t in threads:
            t.start()
        logger.info(
            "OnlineDistillationDataset: %d workers started (depth=%d, multipv=%d, total=%d)",
            len(shards), self.depth, self.multipv_k, len(fens),
        )

        sentinels_seen = 0
        seen = 0
        while sentinels_seen < len(shards):
            item = out_q.get()
            if item is _SENTINEL:
                sentinels_seen += 1
                continue
            yield item
            seen += 1

        for t in threads:
            t.join(timeout=2.0)
        logger.info("OnlineDistillationDataset: %d positions yielded", seen)

    def __len__(self) -> int:
        return self.total_positions
