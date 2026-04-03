import chess

def calc_piece_scope(board: chess.Board, color: chess.Color) -> float:
    max_scope = {chess.KNIGHT: 8, chess.BISHOP: 13, chess.ROOK: 14, chess.QUEEN: 27}
    total_scope = 0
    total_max = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type in max_scope:
            total_max += max_scope[p.piece_type]
            total_scope += len(board.attacks(sq))
    return total_scope / total_max if total_max > 0 else 0.0

def calc_pawn_structure(board: chess.Board, color: chess.Color) -> dict:
    pawns = board.pieces(chess.PAWN, color)
    files = [chess.square_file(sq) for sq in pawns]
    file_counts = {f: files.count(f) for f in range(8)}
    doubled = sum(c - 1 for c in file_counts.values() if c > 1)
    isolated = 0
    for f in set(files):
        if file_counts.get(f - 1, 0) == 0 and file_counts.get(f + 1, 0) == 0:
            isolated += file_counts[f]
    return {"doubled": doubled, "isolated": isolated}

def calc_central_dominance(board: chess.Board, color: chess.Color) -> int:
    inner = [chess.D4, chess.E4, chess.D5, chess.E5]
    outer = [chess.C3, chess.D3, chess.E3, chess.F3,
             chess.C4, chess.F4, chess.C5, chess.F5,
             chess.C6, chess.D6, chess.E6, chess.F6]
    score = 0
    for sq in inner:
        if board.is_attacked_by(color, sq): score += 3
    for sq in outer:
        if board.is_attacked_by(color, sq): score += 1
    return score

def find_outposts(board: chess.Board, color: chess.Color) -> int:
    outposts = 0
    pieces = board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color)
    for sq in pieces:
        rank = chess.square_rank(sq)
        if color == chess.WHITE and rank not in (4, 5): continue
        if color == chess.BLACK and rank not in (2, 3): continue
        
        defenders = board.attackers(color, sq)
        if not any(board.piece_at(d) and board.piece_at(d).piece_type == chess.PAWN for d in defenders):
            continue
            
        outposts += 1
    return outposts

def analyze_theoretical(fens: list[str]) -> dict:
    w_scopes, b_scopes = [], []
    w_center, b_center = [], []
    w_outposts, b_outposts = 0, 0
    
    for fen in fens:
        board = chess.Board(fen)
        w_scopes.append(calc_piece_scope(board, chess.WHITE))
        b_scopes.append(calc_piece_scope(board, chess.BLACK))
        w_center.append(calc_central_dominance(board, chess.WHITE))
        b_center.append(calc_central_dominance(board, chess.BLACK))
        w_outposts = max(w_outposts, find_outposts(board, chess.WHITE))
        b_outposts = max(b_outposts, find_outposts(board, chess.BLACK))
        
    final_board = chess.Board(fens[-1])
    w_struct_end = calc_pawn_structure(final_board, chess.WHITE)
    b_struct_end = calc_pawn_structure(final_board, chess.BLACK)
    
    return {
        "w_avg_scope": sum(w_scopes)/len(w_scopes),
        "b_avg_scope": sum(b_scopes)/len(b_scopes),
        "w_avg_center": sum(w_center)/len(w_center),
        "b_avg_center": sum(b_center)/len(b_center),
        "w_outposts_max": w_outposts,
        "b_outposts_max": b_outposts,
        "w_struct": w_struct_end,
        "b_struct": b_struct_end
    }
