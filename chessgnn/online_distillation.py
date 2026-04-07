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

import json
import logging
import math
import queue
import threading
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from .distillation_dataset import (
    hard_policy_target,
    infer_played_move_uci,
    soft_policy_target,
)
from .distillation_pipeline import (
    evaluate_positions,
    evaluate_positions_engine,
    load_jsonl,
    sample_games_from_pgn,
    sample_positions_from_pgn,
)
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


# ---------------------------------------------------------------------------
# Game-level sequential dataset
# ---------------------------------------------------------------------------

_RESULT_VALUE: dict[str, float] = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


class GameSequenceDataset(IterableDataset):
    """IterableDataset that yields full-game sequences for sequential training.

    Each item is a dict ready for :meth:`GATEAUChessModel.forward_sequence_with_q`::

        {
            "graphs":                list[HeteroData],  # T graphs (ELO injected)
            "value_targets_sf":      Tensor[T],         # Stockfish eval (side-to-move WP → [-1,1])
            "value_targets_outcome": Tensor[T],         # game result (+1/-1/0) repeated T times
            "policy_targets":        list[Tensor[M_t]], # per-position soft policy
            "human_policy_targets":  list[Tensor[M_t]], # played human move target
            "elo_norm":              float,             # avg(white_elo, black_elo) / 3000
            "result_value":          float,             # +1.0 / -1.0 / 0.0
        }

    ``num_workers=0`` must be set in the DataLoader (Stockfish subprocesses).
    """

    def __init__(
        self,
        pgn_path: str,
        stockfish_path: str,
        total_games: int,
        depth: int = 8,
        multipv_k: int = 5,
        buffer_size: int = 32,
        temperature: float = 1.0,
        min_move: int = 5,
        max_move: int = 120,
        num_sf_workers: int = 4,
    ) -> None:
        self.pgn_path = pgn_path
        self.stockfish_path = stockfish_path
        self.total_games = total_games
        self.depth = depth
        self.multipv_k = multipv_k
        self.buffer_size = buffer_size
        self.temperature = temperature
        self.min_move = min_move
        self.max_move = max_move
        self.num_sf_workers = num_sf_workers

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        # Phase 1: read all qualifying game dicts up-front (no Stockfish yet)
        games = list(sample_games_from_pgn(
            self.pgn_path,
            self.total_games,
            min_move=self.min_move,
            max_move=self.max_move,
        ))

        n_workers = min(self.num_sf_workers, len(games))
        if n_workers == 0:
            return

        shard_size = math.ceil(len(games) / n_workers)
        shards = [games[i * shard_size:(i + 1) * shard_size] for i in range(n_workers)]
        shards = [s for s in shards if s]

        out_q: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        _SENTINEL = object()

        def _worker(game_shard: list[dict]) -> None:
            import chess.engine as _ce
            w_builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)
            try:
                engine = _ce.SimpleEngine.popen_uci(self.stockfish_path)
                try:
                    for game_dict in game_shard:
                        _process_game(game_dict, engine, w_builder, out_q)
                finally:
                    engine.quit()
            except Exception as exc:
                logger.error("GameSequenceDataset worker error: %s", exc)
            finally:
                out_q.put(_SENTINEL)

        def _process_game(
            game_dict: dict,
            engine,
            w_builder: ChessGraphBuilder,
            q: queue.Queue,
        ) -> None:
            fens: list[str] = game_dict["fens"]
            white_elo: int = game_dict["white_elo"]
            black_elo: int = game_dict["black_elo"]
            result: str = game_dict["result"]
            result_value: float = _RESULT_VALUE[result]
            elo_norm = min(
                (white_elo + black_elo) / 2.0 / 3000.0, 1.0
            )

            graphs: list = []
            sf_targets: list[float] = []
            policy_targets: list[torch.Tensor] = []
            human_policy_targets: list[torch.Tensor] = []
            labels = list(evaluate_positions_engine(
                iter(fens),
                engine,
                depth=self.depth,
                multipv_k=self.multipv_k,
            ))
            for idx, label in enumerate(labels):
                fen = label["fen"]
                try:
                    g = w_builder.fen_to_graph(fen, white_elo=white_elo, black_elo=black_elo)
                except Exception:
                    logger.debug("Graph build failed for fen in game (skipping position)")
                    continue

                num_legal: int = g["piece", "move", "square"].edge_index.shape[1]
                if num_legal == 0:
                    continue

                pol = soft_policy_target(
                    label["top_k_moves"],
                    fen,
                    num_legal,
                    temperature=self.temperature,
                )
                next_fen = labels[idx + 1]["fen"] if idx + 1 < len(labels) else None
                human_pol = hard_policy_target(
                    infer_played_move_uci(fen, next_fen) if next_fen is not None else None,
                    fen,
                    num_legal,
                )
                graphs.append(g)
                sf_targets.append(2.0 * label["eval_wp"] - 1.0)
                policy_targets.append(pol)
                human_policy_targets.append(human_pol)

            if len(graphs) < 2:
                return

            q.put({
                "graphs": graphs,
                "value_targets_sf": torch.tensor(sf_targets, dtype=torch.float32),
                "value_targets_outcome": torch.full(
                    (len(graphs),), result_value, dtype=torch.float32
                ),
                "policy_targets": policy_targets,
                "human_policy_targets": human_policy_targets,
                "elo_norm": elo_norm,
                "result_value": result_value,
            })

        threads = [
            threading.Thread(target=_worker, args=(s,), daemon=True)
            for s in shards
        ]
        for t in threads:
            t.start()
        logger.info(
            "GameSequenceDataset: %d workers, %d games (depth=%d)",
            len(shards), len(games), self.depth,
        )

        sentinels_seen = 0
        yielded = 0
        while sentinels_seen < len(shards):
            item = out_q.get()
            if item is _SENTINEL:
                sentinels_seen += 1
                continue
            yield item
            yielded += 1

        for t in threads:
            t.join(timeout=2.0)
        logger.info("GameSequenceDataset: %d games yielded", yielded)

    def __len__(self) -> int:
        return self.total_games


# ---------------------------------------------------------------------------
# Offline game-sequence dataset (reads pre-generated JSONL)
# ---------------------------------------------------------------------------


class GameSequenceOfflineDataset(IterableDataset):
    """IterableDataset backed by a pre-generated game-labels JSONL file.

    Expected JSONL format (one game per line, produced by
    :func:`chessgnn.distillation_pipeline.save_game_labels_jsonl`)::

        {
            "fens":       [str, ...],
            "sf_labels":  [{"eval_wp": float, "top_k_moves": [...]}, ...],
            "white_elo":  int,
            "black_elo":  int,
            "result":     str,   # "1-0" | "0-1" | "1/2-1/2"
        }

    Compared to :class:`GameSequenceDataset`, this class removes Stockfish
    from the training critical path entirely.  Graph building (``fen_to_graph``)
    still happens here so ``HeteroData`` objects do not need to be serialised.

    DataLoader ``num_workers`` can be > 0 because there is no Stockfish
    subprocess; set it equal to the number of CPU cores for fastest loading.
    """

    _RESULT_VALUE: dict[str, float] = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

    def __init__(
        self,
        jsonl_path: str,
        temperature: float = 1.0,
        elo_min: int = 0,
        elo_max: int = 9999,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.temperature = temperature
        self.elo_min = elo_min
        self.elo_max = elo_max
        # Count only records inside the active ELO range.
        with open(jsonl_path) as f:
            self._len = 0
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                white_elo = int(rec.get("white_elo", 1500))
                black_elo = int(rec.get("black_elo", 1500))
                avg_elo = (white_elo + black_elo) / 2.0
                if self.elo_min <= avg_elo <= self.elo_max:
                    self._len += 1

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        builder = ChessGraphBuilder(use_global_node=True, use_move_edges=True)

        for line_idx, rec in enumerate(load_jsonl(self.jsonl_path)):
            # When using multiple DataLoader workers, each worker handles only
            # its own slice of lines (round-robin by line index).
            if worker_info is not None and line_idx % worker_info.num_workers != worker_info.id:
                continue

            fens: list[str] = rec["fens"]
            sf_labels: list[dict] = rec["sf_labels"]
            white_elo: int = rec.get("white_elo", 1500)
            black_elo: int = rec.get("black_elo", 1500)
            avg_elo = (white_elo + black_elo) / 2.0
            if avg_elo < self.elo_min or avg_elo > self.elo_max:
                continue
            result: str = rec.get("result", "1/2-1/2")
            result_value: float = self._RESULT_VALUE.get(result, 0.0)
            elo_norm = min(avg_elo / 3000.0, 1.0)

            graphs: list = []
            sf_targets: list[float] = []
            policy_targets: list[torch.Tensor] = []
            human_policy_targets: list[torch.Tensor] = []

            for idx, (fen, label) in enumerate(zip(fens, sf_labels)):
                try:
                    g = builder.fen_to_graph(fen, white_elo=white_elo, black_elo=black_elo)
                except Exception:
                    continue
                num_legal: int = g["piece", "move", "square"].edge_index.shape[1]
                if num_legal == 0:
                    continue
                pol = soft_policy_target(
                    label["top_k_moves"],
                    fen,
                    num_legal,
                    temperature=self.temperature,
                )
                next_fen = fens[idx + 1] if idx + 1 < len(fens) else None
                human_pol = hard_policy_target(
                    infer_played_move_uci(fen, next_fen) if next_fen is not None else None,
                    fen,
                    num_legal,
                )
                graphs.append(g)
                sf_targets.append(2.0 * label["eval_wp"] - 1.0)
                policy_targets.append(pol)
                human_policy_targets.append(human_pol)

            if len(graphs) < 2:
                continue

            yield {
                "graphs": graphs,
                "value_targets_sf": torch.tensor(sf_targets, dtype=torch.float32),
                "value_targets_outcome": torch.full(
                    (len(graphs),), result_value, dtype=torch.float32
                ),
                "policy_targets": policy_targets,
                "human_policy_targets": human_policy_targets,
                "elo_norm": elo_norm,
                "result_value": result_value,
            }

    def __len__(self) -> int:
        return self._len
