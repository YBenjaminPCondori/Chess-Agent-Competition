"""board_v1: no hidden referee history or side-relative rotations."""

import chess
import torch

VERSION = "board_v1"
SHAPE = (21, 8, 8)


def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(SHAPE, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        channel = piece.piece_type - 1 + (0 if piece.color else 6)
        planes[channel, chess.square_rank(square), chess.square_file(square)] = 1
    planes[12].fill_(float(board.turn))
    for channel, color, kingside in (
        (13, True, True),
        (14, True, False),
        (15, False, True),
        (16, False, False),
    ):
        right = (
            board.has_kingside_castling_rights(color)
            if kingside
            else board.has_queenside_castling_rights(color)
        )
        planes[channel].fill_(float(right))
    if board.has_legal_en_passant():
        planes[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    planes[18].fill_(min(board.halfmove_clock, 100) / 100)
    planes[19].fill_(min(board.ply(), 600) / 600)
    return planes
