from __future__ import annotations

import math

import chess
import networkx as nx


_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

_NON_KING_MATERIAL_MAX = 39.0
_MAX_MOBILITY = 64.0
_CENTER = frozenset({chess.D4, chess.D5, chess.E4, chess.E5})
_EXT_CENTER = frozenset({
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
})


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _material_total(board: chess.Board, color: chess.Color) -> float:
    total = 0.0
    for piece in board.piece_map().values():
        if piece.color == color:
            total += _PIECE_VALUES[piece.piece_type]
    return total


def _mobility(board: chess.Board, color: chess.Color) -> int:
    temp = board.copy(stack=False)
    temp.turn = color
    return temp.legal_moves.count()


def _attack_profile(graph: nx.DiGraph, color: chess.Color) -> tuple[float, float, float]:
    nodes = [node for node, data in graph.nodes(data=True) if data["color"] == color]
    if not nodes:
        return 0.0, 0.0, 0.0

    defense_edges = 0
    attack_edges = 0
    outgoing = 0
    for src, dst, data in graph.edges(data=True):
        if src not in nodes:
            continue
        outgoing += 1
        if data["kind"] == "defense":
            defense_edges += 1
        elif data["kind"] == "attack":
            attack_edges += 1

    n = len(nodes)
    density_denom = max(n * max(n - 1, 1), 1)
    attack_denom = max(n * len(graph.nodes), 1)
    mean_out_degree = outgoing / n
    return (
        _clamp01(defense_edges / density_denom),
        _clamp01(attack_edges / attack_denom),
        _clamp01(mean_out_degree / 8.0),
    )


def _contested_squares(board: chess.Board) -> float:
    contested = 0
    for square in chess.SQUARES:
        if board.attackers(chess.WHITE, square) and board.attackers(chess.BLACK, square):
            contested += 1
    return _clamp01(contested / 64.0)


def _center_control(board: chess.Board, color: chess.Color, squares: frozenset[chess.Square]) -> float:
    control = 0
    for square in squares:
        if board.attackers(color, square):
            control += 1
    return _clamp01(control / max(len(squares), 1))


def _pawn_islands(board: chess.Board, color: chess.Color) -> float:
    files_with_pawns = sorted(
        chess.square_file(square)
        for square, piece in board.piece_map().items()
        if piece.color == color and piece.piece_type == chess.PAWN
    )
    if not files_with_pawns:
        return 0.0

    islands = 1
    for prev_file, next_file in zip(files_with_pawns, files_with_pawns[1:]):
        if next_file != prev_file and next_file != prev_file + 1:
            islands += 1
    return _clamp01(islands / 8.0)


def _king_zone_pressure(board: chess.Board, color: chess.Color) -> float:
    king_square = board.king(color)
    if king_square is None:
        return 1.0

    enemy = not color
    pressure = 0
    zone = chess.BB_KING_ATTACKS[king_square] | chess.BB_SQUARES[king_square]
    for square in chess.SquareSet(zone):
        pressure += len(board.attackers(enemy, square))
    return _clamp01(pressure / 24.0)


def build_interaction_graph(board: chess.Board) -> nx.DiGraph:
    graph = nx.DiGraph()

    for square, piece in board.piece_map().items():
        graph.add_node(
            square,
            color=piece.color,
            piece_type=piece.piece_type,
            piece_symbol=piece.symbol(),
            square_name=chess.square_name(square),
            file=chess.square_file(square),
            rank=chess.square_rank(square),
            value=_PIECE_VALUES[piece.piece_type],
        )

    for src_square, src_piece in board.piece_map().items():
        for dst_square in board.attacks(src_square):
            dst_piece = board.piece_at(dst_square)
            if dst_piece is None:
                continue

            kind = "defense" if dst_piece.color == src_piece.color else "attack"
            graph.add_edge(
                src_square,
                dst_square,
                kind=kind,
                weight=_PIECE_VALUES[src_piece.piece_type] / 9.0,
            )

    return graph


def position_fingerprint(
    board: chess.Board,
    graph: nx.DiGraph | None = None,
) -> list[float]:
    interaction_graph = graph if graph is not None else build_interaction_graph(board)

    white_material = _material_total(board, chess.WHITE) / _NON_KING_MATERIAL_MAX
    black_material = _material_total(board, chess.BLACK) / _NON_KING_MATERIAL_MAX
    material_balance = (_material_total(board, chess.WHITE) - _material_total(board, chess.BLACK))
    material_balance = (material_balance + _NON_KING_MATERIAL_MAX) / (2.0 * _NON_KING_MATERIAL_MAX)

    white_mobility = _mobility(board, chess.WHITE) / _MAX_MOBILITY
    black_mobility = _mobility(board, chess.BLACK) / _MAX_MOBILITY

    white_defense, white_attack, white_activity = _attack_profile(interaction_graph, chess.WHITE)
    black_defense, black_attack, black_activity = _attack_profile(interaction_graph, chess.BLACK)

    fingerprint = [
        _clamp01(white_material),
        _clamp01(black_material),
        _clamp01(material_balance),
        _clamp01(white_mobility),
        _clamp01(black_mobility),
        white_defense,
        black_defense,
        white_attack,
        black_attack,
        white_activity,
        black_activity,
        _contested_squares(board),
        _center_control(board, chess.WHITE, _CENTER),
        _center_control(board, chess.BLACK, _CENTER),
        _center_control(board, chess.WHITE, _EXT_CENTER),
        _center_control(board, chess.BLACK, _EXT_CENTER),
        _clamp01(board.legal_moves.count() / _MAX_MOBILITY),
        1.0 if board.turn == chess.WHITE else 0.0,
        _clamp01(chess.popcount(board.castling_rights & chess.BB_RANK_1) / 2.0),
        _clamp01(chess.popcount(board.castling_rights & chess.BB_RANK_8) / 2.0),
        _pawn_islands(board, chess.WHITE),
        _pawn_islands(board, chess.BLACK),
        _king_zone_pressure(board, chess.WHITE),
        _king_zone_pressure(board, chess.BLACK),
    ]
    return [round(value, 6) for value in fingerprint]


def fingerprint_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Fingerprint vectors must have the same dimension")
    if not left:
        return 1.0

    mean_abs_delta = sum(abs(a - b) for a, b in zip(left, right)) / len(left)
    return round(_clamp01(1.0 - mean_abs_delta), 6)


def find_similar_fingerprints(
    query: list[float],
    candidates: list[list[float]],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    ranked = [
        (idx, fingerprint_similarity(query, candidate))
        for idx, candidate in enumerate(candidates)
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]