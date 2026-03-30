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

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        buf: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        _SENTINEL = object()

        def _producer() -> None:
            try:
                fens = sample_positions_from_pgn(
                    self.pgn_path,
                    self.total_positions,
                    min_move=self.min_move,
                    max_move=self.max_move,
                )
                for label in evaluate_positions(
                    fens,
                    self.stockfish_path,
                    depth=self.depth,
                    multipv_k=self.multipv_k,
                ):
                    buf.put(label)
            except Exception as exc:
                logger.error("OnlineDistillationDataset producer error: %s", exc)
            finally:
                buf.put(_SENTINEL)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        logger.info(
            "OnlineDistillationDataset: producer started (depth=%d, multipv=%d, total=%d)",
            self.depth,
            self.multipv_k,
            self.total_positions,
        )

        seen = 0
        while True:
            item = buf.get()
            if item is _SENTINEL:
                break

            label: dict = item
            fen = label["fen"]
            try:
                graph = self.builder.fen_to_graph(fen)
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

            yield {
                "graph": graph,
                "value_target": value_target,
                "policy_target": policy_target,
                "num_legal_moves": num_legal,
            }

            seen += 1
            if seen >= self.total_positions:
                break

        thread.join(timeout=2.0)
        logger.info("OnlineDistillationDataset: %d positions yielded", seen)

    def __len__(self) -> int:
        return self.total_positions
