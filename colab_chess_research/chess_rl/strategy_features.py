"""Interpretable corpus diagnostics. Heuristics are never used as best-move labels."""

import chess

CENTRE = (chess.D4, chess.E4, chess.D5, chess.E5)


def _ahead(square, other, color):
    return (chess.square_rank(other) - chess.square_rank(square)) * (1 if color else -1) > 0


def _side(board, color):
    pawns = list(board.pieces(chess.PAWN, color))
    enemy = list(board.pieces(chess.PAWN, not color))
    files = [chess.square_file(s) for s in pawns]
    enemy_files = [chess.square_file(s) for s in enemy]
    passed = [
        s
        for s in pawns
        if not any(
            abs(chess.square_file(s) - chess.square_file(e)) <= 1 and _ahead(s, e, color)
            for e in enemy
        )
    ]
    king = board.king(color)
    ring = list(board.attacks(king))
    direction = 8 if color else -8
    backward = [
        s
        for s in pawns
        if 0 <= s + direction < 64
        and any(
            board.piece_type_at(e) == chess.PAWN for e in board.attackers(not color, s + direction)
        )
        and not any(
            abs(chess.square_file(s) - chess.square_file(p)) == 1 and not _ahead(s, p, color)
            for p in pawns
        )
    ]
    rooks = list(board.pieces(chess.ROOK, color))
    bishops = list(board.pieces(chess.BISHOP, color))
    knights = list(board.pieces(chess.KNIGHT, color))
    pieces = list(chess.SquareSet(board.occupied_co[color]))
    moves = list(board.legal_moves) if board.turn == color else None

    def pawn_defended(square):
        return bool(board.attackers(color, square) & board.pieces(chess.PAWN, color))

    outposts = [
        s
        for s in knights
        if pawn_defended(s)
        and not any(
            abs(chess.square_file(s) - chess.square_file(e)) == 1 and _ahead(s, e, color)
            for e in enemy
        )
    ]
    return dict(
        castling_rights=board.has_castling_rights(color),
        pawn_shield=sum(
            1 for s in pawns if chess.square_distance(king, s) <= 1 and _ahead(king, s, color)
        ),
        open_files_near_king=sum(
            1
            for f in range(max(0, chess.square_file(king) - 1), min(8, chess.square_file(king) + 2))
            if f not in files and f not in enemy_files
        ),
        king_attackers=len(set().union(*(set(board.attackers(not color, s)) for s in ring))),
        isolated_pawns=sum(not any(abs(f - g) == 1 for g in files) for f in files),
        doubled_pawns=len(files) - len(set(files)),
        passed_pawns=len(passed),
        backward_pawns_heuristic=len(backward),
        pawn_chains=sum(pawn_defended(s) for s in pawns),
        queenside_majority=sum(f < 4 for f in files) - sum(f < 4 for f in enemy_files),
        kingside_majority=sum(f >= 4 for f in files) - sum(f >= 4 for f in enemy_files),
        legal_mobility=len(moves) if moves is not None else None,
        outposts_heuristic=len(outposts),
        bishop_pair=len(bishops) >= 2,
        trapped_pieces_heuristic=sum(
            not any(m.from_square == s for m in moves)
            for s in pieces
            if board.piece_type_at(s) in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        )
        if moves is not None
        else None,
        bad_bishops_heuristic=sum(
            sum(
                (chess.square_rank(s) + chess.square_file(s)) % 2
                == (chess.square_rank(b) + chess.square_file(b)) % 2
                for s in pawns
            )
            >= 4
            for b in bishops
        ),
        active_knights_heuristic=sum(
            len(board.attacks(s) & ~chess.SquareSet(board.occupied_co[color])) >= 5 for s in knights
        ),
        central_pawns=sum(s in CENTRE for s in pawns),
        attacked_central_squares=sum(board.is_attacked_by(color, s) for s in CENTRE),
        central_piece_pressure=sum(
            len(board.attackers(color, s) - board.pieces(chess.PAWN, color)) for s in CENTRE
        ),
        rook_open_files=sum(chess.square_file(s) not in files + enemy_files for s in rooks),
        rook_semi_open_files=sum(
            chess.square_file(s) not in files and chess.square_file(s) in enemy_files for s in rooks
        ),
        rooks_on_seventh=sum(chess.square_rank(s) == (6 if color else 1) for s in rooks),
        rooks_behind_passers=sum(
            any(
                chess.square_file(r) == chess.square_file(p) and _ahead(r, p, color) for p in passed
            )
            for r in rooks
        ),
        king_center_distance=min(chess.square_distance(king, s) for s in CENTRE),
        promotion_distance=min(
            (7 - chess.square_rank(s) if color else chess.square_rank(s) for s in passed),
            default=None,
        ),
        hanging_pieces_heuristic=sum(
            board.is_attacked_by(not color, s) and not board.is_attacked_by(color, s)
            for s in pieces
            if board.piece_type_at(s) != chess.KING
        ),
        pinned_pieces=sum(board.is_pinned(color, s) for s in pieces),
    )


def extract_features(board, history_known=False):
    if not board.is_valid():
        raise ValueError("Invalid feature FEN")
    legal = list(board.legal_moves)
    non_pawns = [
        p.piece_type
        for p in board.piece_map().values()
        if p.piece_type not in (chess.PAWN, chess.KING)
    ]
    features = {
        "white": _side(board, chess.WHITE),
        "black": _side(board, chess.BLACK),
        "captures": sum(board.is_capture(m) for m in legal),
        "checks": sum(board.gives_check(m) for m in legal),
        "stalemate": board.is_stalemate(),
        "insufficient_material": board.is_insufficient_material(),
        "fifty_move_rule": board.is_fifty_moves(),
        "threefold_repetition": board.is_repetition(3) if history_known else None,
        "history_known": history_known,
        "castled_white": None,
        "castled_black": None,
        "low_material": len(non_pawns) <= 4,
        "rook_endgame": bool(non_pawns) and set(non_pawns) == {chess.ROOK},
        "minor_piece_endgame": bool(non_pawns) and set(non_pawns) <= {chess.BISHOP, chess.KNIGHT},
        "queen_endgame": bool(non_pawns) and set(non_pawns) == {chess.QUEEN},
        "fortress_like_heuristic": len(non_pawns) <= 4 and board.halfmove_clock >= 40,
    }
    if history_known:
        replay = board.root()
        if replay.fen() == chess.STARTING_FEN:
            features.update(castled_white=False, castled_black=False)
        for move in board.move_stack:
            if replay.is_castling(move):
                features["castled_white" if replay.turn else "castled_black"] = True
            replay.push(move)
    return features
