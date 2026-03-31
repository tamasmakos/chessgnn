
import torch
from torch_geometric.data import HeteroData
import chess
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional

# Constants
SQUARES = chess.SQUARES
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0 # King value handled separately or 0 for trade logic
}

class ChessGraphBuilder:
    def __init__(self, use_global_node: bool = False, use_move_edges: bool = False):
        self.node_type_map = {'piece': 0, 'square': 1}
        self.use_global_node = use_global_node
        self.use_move_edges = use_move_edges
        
    def fen_to_graph(
        self,
        fen: str,
        history_emb: Optional[torch.Tensor] = None,
        white_elo: Optional[int] = None,
        black_elo: Optional[int] = None,
    ) -> HeteroData:
        """
        Converts a FEN string into a HeteroData object.

        Nodes (always present):
            - piece:  [type_onehot(6), color, value, file, rank]  (10-dim)
            - square: [file/7, rank/7, is_occupied]               (3-dim)
        Nodes (optional):
            - global: [side_to_move, castle_kside_w, castle_qside_w, castle_kside_b,
                       castle_qside_b, halfmove_clock/100, material_w/39, material_b/39,
                       game_phase]                                 (9-dim, if use_global_node)

        Edges (always present):
            - (piece, on, square):              location
            - (square, occupied_by, piece):     reverse location
            - (square, adjacent, square):       Chebyshev-1 grid adjacency
            - (piece, interacts, piece):        attack/defense, attr=[weight, type]
            - (piece, ray, piece):              long-range alignment, attr=[dist/7, blocking]
        Edges (optional):
            - (global, global_to_piece, piece):  if use_global_node
            - (piece, piece_to_global, global):  if use_global_node
            - (global, global_to_square, square): if use_global_node
            - (square, square_to_global, global): if use_global_node
            - (piece, move, square):             legal moves,
                                                 attr=[is_capture, captured_val/9,
                                                       is_promotion, is_castling, distance/7]
                                                 if use_move_edges
        """
        board = chess.Board(fen)
        data = HeteroData()
        data.fen = fen

        # 1. Nodes Construction
        pieces = []
        piece_indices = {} # map square -> piece_idx
        
        squares = []
        square_indices = {sq: i for i, sq in enumerate(SQUARES)}
        
        # Squares Nodes
        # Feature: [file, rank, is_occupied]
        for sq in SQUARES:
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            piece = board.piece_at(sq)
            is_occupied = 1.0 if piece else 0.0
            squares.append([file/7.0, rank/7.0, is_occupied])
        
        data['square'].x = torch.tensor(squares, dtype=torch.float)

        # Pieces Nodes
        # Feature: [type(onehot 6), color(1), value(1), file(1), rank(1)]
        current_piece_idx = 0
        for sq in SQUARES:
            piece = board.piece_at(sq)
            if piece:
                # One-hot type
                type_vec = [0]*6
                type_vec[PIECE_TYPES.index(piece.piece_type)] = 1
                
                # Color (White=1, Black=-1)
                color = 1.0 if piece.color == chess.WHITE else -1.0
                
                # Value
                val = PIECE_VALUES[piece.piece_type] / 10.0
                
                # Pos
                file, rank = chess.square_file(sq), chess.square_rank(sq)
                
                feat = type_vec + [color, val, file/7.0, rank/7.0]
                pieces.append(feat)
                piece_indices[sq] = current_piece_idx
                current_piece_idx += 1

        if pieces:
            data['piece'].x = torch.tensor(pieces, dtype=torch.float)
        else:
            data['piece'].x = torch.empty((0, 10), dtype=torch.float)

        # 2. Edges Construction
        
        # Piece-Square (Location)
        # Edge type: ('piece', 'on', 'square')
        edge_index_on = [[], []]
        for sq, p_idx in piece_indices.items():
            edge_index_on[0].append(p_idx)
            edge_index_on[1].append(square_indices[sq])
        
        data['piece', 'on', 'square'].edge_index = torch.tensor(edge_index_on, dtype=torch.long)

        # Reverse Edge: Square -> Piece (Occupied By)
        # Allows squares to inform pieces about their location properties (e.g. center control)
        data['square', 'occupied_by', 'piece'].edge_index = torch.tensor([edge_index_on[1], edge_index_on[0]], dtype=torch.long)

        
        # Square-Square (Adjacency)
        # Edge type: ('square', 'adjacent', 'square')
        # King moves (Chebyshev distance = 1)
        edge_index_adj = [[], []]
        for sq1 in SQUARES:
            for sq2 in SQUARES:
                if sq1 == sq2: continue
                f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
                f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
                if max(abs(f1-f2), abs(r1-r2)) == 1:
                    edge_index_adj[0].append(square_indices[sq1])
                    edge_index_adj[1].append(square_indices[sq2])

        data['square', 'adjacent', 'square'].edge_index = torch.tensor(edge_index_adj, dtype=torch.long)
        
        # Piece-Piece (Interaction: Attack/Defense)
        # Edge type: ('piece', 'interacts', 'piece')
        # Weight computed based on value difference
        edge_index_int = [[], []]
        edge_attr_int = [] 

        # We iterate over all pieces and check attacks
        for sq_src, p_idx_src in piece_indices.items():
            piece_src = board.piece_at(sq_src)
            attacks = board.attacks(sq_src)
            
            for sq_dst in attacks:
                if sq_dst in piece_indices:
                    p_idx_dst = piece_indices[sq_dst]
                    piece_dst = board.piece_at(sq_dst)
                    
                    # Attack or Defense?
                    if piece_src.color != piece_dst.color:
                        # Attack
                        # Weight = sigmoid(Val(Target) - Val(Attacker))
                        # Favors capturing high value with low value
                        diff = PIECE_VALUES[piece_dst.piece_type] - PIECE_VALUES[piece_src.piece_type]
                        weight = self.sigmoid(diff)
                        edge_type = 1.0 # Attack
                    else:
                        # Defense
                        # Weight = sigmoid(Val(Defender)) ?? Or just constant?
                        # Plan says: function of value difference.
                        # Using 1.0 as standard strong connection
                        weight = 1.0 # Strong support
                        edge_type = -1.0 # Defense
                    
                    edge_index_int[0].append(p_idx_src)
                    edge_index_int[1].append(p_idx_dst)
                    edge_attr_int.append([weight, edge_type])
        
        if edge_index_int[0]:
            data['piece', 'interacts', 'piece'].edge_index = torch.tensor(edge_index_int, dtype=torch.long)
            data['piece', 'interacts', 'piece'].edge_attr = torch.tensor(edge_attr_int, dtype=torch.float)
        else:
             data['piece', 'interacts', 'piece'].edge_index = torch.empty((2, 0), dtype=torch.long)
             data['piece', 'interacts', 'piece'].edge_attr = torch.empty((0, 2), dtype=torch.float)

        # Piece-Piece (Ray Edges) - Simplified for now
        # We need to detect pins/skewers. Any piece on the same rank/file/diagonal.
        # This is O(N^2) but N <= 32.
        
        edge_index_ray = [[], []]
        edge_attr_ray = [] # [distance, blocking_count]
        
        pieces_locs = list(piece_indices.items())
        for i in range(len(pieces_locs)):
            for j in range(len(pieces_locs)):
                if i == j: continue
                
                sq1, p1_idx = pieces_locs[i]
                sq2, p2_idx = pieces_locs[j]
                
                # Check alignment
                if self.is_aligned(sq1, sq2):
                    dist = chess.square_distance(sq1, sq2)
                    blocking = self.count_blocking(board, sq1, sq2)
                    
                    edge_index_ray[0].append(p1_idx)
                    edge_index_ray[1].append(p2_idx)
                    edge_attr_ray.append([float(dist)/7.0, float(blocking)])

        if edge_index_ray[0]:
            data['piece', 'ray', 'piece'].edge_index = torch.tensor(edge_index_ray, dtype=torch.long)
            data['piece', 'ray', 'piece'].edge_attr = torch.tensor(edge_attr_ray, dtype=torch.float)
        else:
            data['piece', 'ray', 'piece'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['piece', 'ray', 'piece'].edge_attr = torch.empty((0, 2), dtype=torch.float)

        # --- Optional: Global Node ---
        if self.use_global_node:
            side_to_move = 1.0 if board.turn == chess.WHITE else -1.0
            castle_kside_w = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
            castle_qside_w = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
            castle_kside_b = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
            castle_qside_b = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
            halfmove = min(board.halfmove_clock / 100.0, 1.0)

            material_white = sum(
                PIECE_VALUES[board.piece_at(sq).piece_type]
                for sq in SQUARES if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
            ) / 39.0
            material_black = sum(
                PIECE_VALUES[board.piece_at(sq).piece_type]
                for sq in SQUARES if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
            ) / 39.0

            non_pawn_material = sum(
                PIECE_VALUES[board.piece_at(sq).piece_type]
                for sq in SQUARES
                if board.piece_at(sq) and board.piece_at(sq).piece_type not in (chess.PAWN, chess.KING)
            )
            game_phase = min(non_pawn_material / 62.0, 1.0)

            elo_white = min(max(white_elo or 1500, 0), 3000) / 3000.0
            elo_black = min(max(black_elo or 1500, 0), 3000) / 3000.0

            global_feat = [
                side_to_move, castle_kside_w, castle_qside_w,
                castle_kside_b, castle_qside_b, halfmove,
                material_white, material_black, game_phase,
                elo_white, elo_black,
            ]
            data['global'].x = torch.tensor([global_feat], dtype=torch.float)  # [1, 11]

            # global -> piece  and  piece -> global
            num_pieces = len(piece_indices)
            if num_pieces > 0:
                g2p_src = [0] * num_pieces
                g2p_dst = list(range(num_pieces))
                data['global', 'global_to_piece', 'piece'].edge_index = torch.tensor(
                    [g2p_src, g2p_dst], dtype=torch.long)
                data['piece', 'piece_to_global', 'global'].edge_index = torch.tensor(
                    [g2p_dst, g2p_src], dtype=torch.long)
            else:
                data['global', 'global_to_piece', 'piece'].edge_index = torch.empty((2, 0), dtype=torch.long)
                data['piece', 'piece_to_global', 'global'].edge_index = torch.empty((2, 0), dtype=torch.long)

            # global -> square  and  square -> global
            num_squares = len(SQUARES)
            g2s_src = [0] * num_squares
            g2s_dst = list(range(num_squares))
            data['global', 'global_to_square', 'square'].edge_index = torch.tensor(
                [g2s_src, g2s_dst], dtype=torch.long)
            data['square', 'square_to_global', 'global'].edge_index = torch.tensor(
                [g2s_dst, g2s_src], dtype=torch.long)

        # --- Optional: Move Edges ---
        if self.use_move_edges:
            # Edge type: ('piece', 'move', 'square')
            # One directed edge per legal move: source piece -> destination square
            # Features: [is_capture, captured_value/9, is_promotion, is_castling, distance/7]
            move_src: list[int] = []
            move_dst: list[int] = []
            move_attr: list[list[float]] = []

            for move in board.legal_moves:
                from_sq = move.from_square
                to_sq = move.to_square
                if from_sq not in piece_indices:
                    continue

                p_src_idx = piece_indices[from_sq]
                sq_dst_idx = square_indices[to_sq]

                is_capture = 1.0 if board.is_capture(move) else 0.0
                if board.is_en_passant(move):
                    captured_val = PIECE_VALUES[chess.PAWN] / 9.0
                elif board.is_capture(move):
                    captured_piece = board.piece_at(to_sq)
                    captured_val = PIECE_VALUES[captured_piece.piece_type] / 9.0 if captured_piece else 0.0
                else:
                    captured_val = 0.0

                is_promotion = 1.0 if move.promotion is not None else 0.0
                is_castling = 1.0 if board.is_castling(move) else 0.0
                distance = chess.square_distance(from_sq, to_sq) / 7.0

                move_src.append(p_src_idx)
                move_dst.append(sq_dst_idx)
                move_attr.append([is_capture, captured_val, is_promotion, is_castling, distance])

            if move_src:
                data['piece', 'move', 'square'].edge_index = torch.tensor(
                    [move_src, move_dst], dtype=torch.long)
                data['piece', 'move', 'square'].edge_attr = torch.tensor(move_attr, dtype=torch.float)
            else:
                data['piece', 'move', 'square'].edge_index = torch.empty((2, 0), dtype=torch.long)
                data['piece', 'move', 'square'].edge_attr = torch.empty((0, 5), dtype=torch.float)

        return data

    def get_metadata(self) -> tuple:
        """Returns the (node_types, edge_types) metadata tuple for the current flag configuration."""
        node_types = ['piece', 'square']
        edge_types = [
            ('piece', 'on', 'square'),
            ('square', 'occupied_by', 'piece'),
            ('square', 'adjacent', 'square'),
            ('piece', 'interacts', 'piece'),
            ('piece', 'ray', 'piece'),
        ]
        if self.use_global_node:
            node_types.append('global')
            edge_types.extend([
                ('global', 'global_to_piece', 'piece'),
                ('piece', 'piece_to_global', 'global'),
                ('global', 'global_to_square', 'square'),
                ('square', 'square_to_global', 'global'),
            ])
        if self.use_move_edges:
            edge_types.append(('piece', 'move', 'square'))
        return node_types, edge_types

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def is_aligned(self, sq1, sq2):
        f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
        f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
        return (f1 == f2) or (r1 == r2) or (abs(f1-f2) == abs(r1-r2))

    def count_blocking(self, board, sq1, sq2):
        # Ray cast
        # We use chess.Ray logic or manual steps
        # Manual step iteration
        f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
        f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
        
        df = 0 if f1 == f2 else (1 if f2 > f1 else -1)
        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        
        curr_f, curr_r = f1 + df, r1 + dr
        blocking = 0
        
        while (curr_f != f2) or (curr_r != r2):
            sq = chess.square(curr_f, curr_r)
            if board.piece_at(sq):
                blocking += 1
            curr_f += df
            curr_r += dr
            
        return blocking
