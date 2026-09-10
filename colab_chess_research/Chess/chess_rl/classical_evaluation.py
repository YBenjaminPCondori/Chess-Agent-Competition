import math
import chess
from .piece_tables import PIECE_VALUES, PIECE_SQUARE_TABLES


def evaluate_cp(board):
    white_score = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        for piece_type, value in PIECE_VALUES.items():
            for square in board.pieces(piece_type, color):
                index = square if color else chess.square_mirror(square)
                white_score += sign * (value + PIECE_SQUARE_TABLES[piece_type][index])
                if piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    white_score += sign * 2 * len(board.attacks(square))
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            white_score += sign * 35
        pawns = list(board.pieces(chess.PAWN, color))
        files = [chess.square_file(s) for s in pawns]
        white_score -= sign * sum(12 * max(0, files.count(f) - 1) for f in range(8))
        for square in pawns:
            file, rank = chess.square_file(square), chess.square_rank(square)
            if not any(abs(file - f) == 1 for f in files):
                white_score -= sign * 10
            enemies_ahead = [
                s
                for s in board.pieces(chess.PAWN, not color)
                if abs(chess.square_file(s) - file) <= 1
                and ((chess.square_rank(s) > rank) if color else (chess.square_rank(s) < rank))
            ]
            if not enemies_ahead:
                white_score += sign * (20 + 8 * (rank if color else 7 - rank))
    return white_score if board.turn else -white_score


def evaluate(board):
    return math.tanh(evaluate_cp(board) / 600)


def tactical_score(board, move):
    score = 8000 + PIECE_VALUES[move.promotion] if move.promotion else 0
    if board.is_capture(move):
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        score += (
            10000
            + 10 * PIECE_VALUES.get(victim, 0)
            - PIECE_VALUES[board.piece_type_at(move.from_square)]
        )
    return score


def fallback_move(board, moves=None):
    moves = list(board.legal_moves) if moves is None else moves
    return (
        min(moves, key=lambda m: (-tactical_score(board, m), m.uci()))
        if moves
        else chess.Move.null()
    )
