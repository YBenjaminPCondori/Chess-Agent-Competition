"""action_v1: channel-major 73 x 8 x 8, absolute White-oriented squares."""

import chess
import torch

VERSION = "action_v1"
ACTION_SIZE = 4672
DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
KNIGHTS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def encode_move(board: chess.Board, move: chess.Move) -> int:
    if move not in board.legal_moves:
        raise ValueError(f"Illegal target move: {move.uci()}")
    df = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    if move.promotion in UNDERPROMOTIONS:
        plane = 64 + UNDERPROMOTIONS.index(move.promotion) * 3 + df + 1
    elif (df, dr) in KNIGHTS:
        plane = 56 + KNIGHTS.index((df, dr))
    else:
        distance = max(abs(df), abs(dr))
        direction = (df // distance, dr // distance)
        plane = DIRECTIONS.index(direction) * 7 + distance - 1
    return plane * 64 + move.from_square


def decode_move(board: chess.Board, index: int) -> chess.Move:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < ACTION_SIZE:
        raise ValueError("Action index out of range")
    plane, source = divmod(index, 64)
    promotion = None
    if plane < 56:
        direction, offset = divmod(plane, 7)
        df, dr = DIRECTIONS[direction]
        df, dr = df * (offset + 1), dr * (offset + 1)
    elif plane < 64:
        df, dr = KNIGHTS[plane - 56]
    else:
        piece, offset = divmod(plane - 64, 3)
        promotion = UNDERPROMOTIONS[piece]
        df, dr = offset - 1, 1 if board.turn else -1
    file = chess.square_file(source) + df
    rank = chess.square_rank(source) + dr
    if not 0 <= file < 8 or not 0 <= rank < 8:
        raise ValueError("Off-board action")
    if promotion is None and board.piece_type_at(source) == chess.PAWN and rank in (0, 7):
        promotion = chess.QUEEN
    move = chess.Move(source, chess.square(file, rank), promotion=promotion)
    if move not in board.legal_moves:
        raise ValueError("Action is illegal in this position")
    return move


def legal_indices(board):
    return [encode_move(board, move) for move in board.legal_moves]


def legal_mask(board):
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    mask[legal_indices(board)] = True
    return mask
