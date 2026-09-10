import math
from collections.abc import Mapping
import chess
from .piece_tables import PIECE_VALUES, PIECE_SQUARE_TABLES

DEFAULT_WEIGHTS = {
    "material": 1.0,
    "piece_square": 1.0,
    "activity": 2.0,
    "bishop_pair": 35.0,
    "doubled_pawns": -12.0,
    "isolated_pawns": -10.0,
    "passed_pawns": 20.0,
    "passed_pawn_advance": 8.0,
}


def resolve_weights(overrides=None):
    weights = DEFAULT_WEIGHTS.copy()
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("evaluation_weights must be a mapping")
        unknown = set(overrides) - set(weights)
        if unknown:
            raise ValueError(f"Unknown evaluation weights: {sorted(unknown)}")
        for key, value in overrides.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Evaluation weight {key} must be a finite number")
            weights[key] = float(value)
    return weights


def position_features(board):
    """White-minus-Black features; negative weights penalize pawn weaknesses."""
    features = dict.fromkeys(DEFAULT_WEIGHTS, 0)
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        for piece_type, value in PIECE_VALUES.items():
            for square in board.pieces(piece_type, color):
                index = square if color else chess.square_mirror(square)
                features["material"] += sign * value
                features["piece_square"] += sign * PIECE_SQUARE_TABLES[piece_type][index]
                if piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    features["activity"] += sign * len(board.attacks(square))
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            features["bishop_pair"] += sign
        pawns = list(board.pieces(chess.PAWN, color))
        files = [chess.square_file(s) for s in pawns]
        features["doubled_pawns"] += sign * sum(max(0, files.count(f) - 1) for f in range(8))
        for square in pawns:
            file, rank = chess.square_file(square), chess.square_rank(square)
            if not any(abs(file - f) == 1 for f in files):
                features["isolated_pawns"] += sign
            enemies_ahead = [
                s
                for s in board.pieces(chess.PAWN, not color)
                if abs(chess.square_file(s) - file) <= 1
                and ((chess.square_rank(s) > rank) if color else (chess.square_rank(s) < rank))
            ]
            if not enemies_ahead:
                features["passed_pawns"] += sign
                features["passed_pawn_advance"] += sign * (rank if color else 7 - rank)
    return features


def weighted_cp(board, weights):
    """Hot-path evaluation with weights already checked when the search engine initializes."""
    white_score = sum(weights[name] * value for name, value in position_features(board).items())
    return white_score if board.turn else -white_score


def evaluate_cp(board, weights=None):
    return weighted_cp(board, resolve_weights(weights))


def evaluate(board, weights=None):
    return math.tanh(evaluate_cp(board, weights) / 600)


def evaluation_breakdown(board, weights=None):
    weights = resolve_weights(weights)
    sign = 1 if board.turn else -1
    return [
        dict(
            feature=name,
            value=sign * value,
            weight=weights[name],
            contribution_cp=sign * value * weights[name],
        )
        for name, value in position_features(board).items()
    ]


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
