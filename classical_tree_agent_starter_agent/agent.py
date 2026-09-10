"""
Classical tree-search starter agent for the AI Chessathon interface.

Competition entry point:
    get_move(fen: str, time_left_ms: int) -> str

This file intentionally uses only python-chess and the Python standard library.
It is written for clarity and reliability, not maximum engine strength.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess


MATE_SCORE = 100_000
INF = 1_000_000

EXACT = 0
LOWER = 1
UPPER = 2

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


# Piece-square tables are from White's perspective. Black uses mirrored squares.
PAWN_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
    0, 0, 5, 10, 10, 5, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 5, 5, 0, 0, 0,
]

QUEEN_PST = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_MIDGAME_PST = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_MIDGAME_PST,
}


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move_uci: str | None


class SearchTimeout(Exception):
    pass


class Engine:
    def __init__(self) -> None:
        self.deadline_ns = 0
        self.nodes = 0
        self.tt: dict[object, TTEntry] = {}
        self.max_tt_entries = 100_000

    def choose_move(self, board: chess.Board, time_left_ms: int) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()

        fallback = self.quick_fallback(board, legal_moves)
        if time_left_ms <= 80:
            return fallback

        budget_ms = self.allocate_time_ms(board, time_left_ms)
        self.deadline_ns = time.perf_counter_ns() + int(budget_ms * 1_000_000)
        self.nodes = 0

        best_move = fallback
        previous_best: chess.Move | None = None

        try:
            for depth in range(1, 64):
                depth_best, _score = self.search_root(board, depth, previous_best)
                best_move = depth_best
                previous_best = depth_best
                if self.out_of_time():
                    break
        except SearchTimeout:
            pass
        except Exception:
            return fallback

        return best_move

    def allocate_time_ms(self, board: chess.Board, time_left_ms: int) -> int:
        if time_left_ms < 250:
            return 20
        if time_left_ms < 1_000:
            return max(25, time_left_ms // 12)
        if time_left_ms < 5_000:
            return min(250, max(60, time_left_ms // 18))

        phase_factor = 1.25 if board.fullmove_number < 45 else 0.75
        base = int(time_left_ms / 35 * phase_factor)
        return min(1_500, max(120, base))

    def out_of_time(self) -> bool:
        return time.perf_counter_ns() >= self.deadline_ns

    def check_time(self) -> None:
        self.nodes += 1
        if self.nodes & 1023 == 0 and self.out_of_time():
            raise SearchTimeout

    def search_root(
        self,
        board: chess.Board,
        depth: int,
        previous_best: chess.Move | None,
    ) -> tuple[chess.Move, int]:
        alpha = -INF
        beta = INF
        best_score = -INF
        moves = self.ordered_moves(board, previous_best)
        best_move = moves[0]

        for move in moves:
            self.check_time()
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_move, best_score

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.check_time()

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, 0)

        original_alpha = alpha
        original_beta = beta
        key = self.position_key(board)
        entry = self.tt.get(key)
        tt_best = None

        if entry is not None:
            if entry.best_move_uci is not None:
                try:
                    move = chess.Move.from_uci(entry.best_move_uci)
                    if move in board.legal_moves:
                        tt_best = move
                except ValueError:
                    tt_best = None

            if entry.depth >= depth:
                if entry.flag == EXACT:
                    return entry.score
                if entry.flag == LOWER:
                    alpha = max(alpha, entry.score)
                elif entry.flag == UPPER:
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

        best_score = -INF
        best_move = None

        for move in self.ordered_moves(board, tt_best):
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        flag = EXACT
        if best_score <= original_alpha:
            flag = UPPER
        elif best_score >= original_beta:
            flag = LOWER
        self.store_tt(key, TTEntry(depth, best_score, flag, best_move.uci() if best_move else None))
        return best_score

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int, q_depth: int) -> int:
        self.check_time()

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        if q_depth >= 5:
            return alpha

        noisy_moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                noisy_moves.append(move)

        for move in self.sort_moves(board, noisy_moves, None):
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1, q_depth + 1)
            finally:
                board.pop()

            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    def ordered_moves(self, board: chess.Board, preferred: chess.Move | None) -> list[chess.Move]:
        return self.sort_moves(board, list(board.legal_moves), preferred)

    def sort_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        preferred: chess.Move | None,
    ) -> list[chess.Move]:
        def score_move(move: chess.Move) -> int:
            if preferred is not None and move == preferred:
                return 1_000_000

            score = 0
            if move.promotion:
                score += PIECE_VALUES.get(move.promotion, 0) + 8_000

            if board.is_capture(move):
                attacker = board.piece_at(move.from_square)
                captured = board.piece_at(move.to_square)
                if captured is None and board.is_en_passant(move):
                    captured_value = PIECE_VALUES[chess.PAWN]
                else:
                    captured_value = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
                attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
                score += 10_000 + 10 * captured_value - attacker_value

            if board.gives_check(move):
                score += 500

            return score

        moves.sort(key=score_move, reverse=True)
        return moves

    def quick_fallback(self, board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        return self.sort_moves(board, legal_moves[:], None)[0]

    def position_key(self, board: chess.Board) -> object:
        key_fn = getattr(board, "transposition_key", None)
        if key_fn is None:
            key_fn = getattr(board, "_transposition_key", None)
        if key_fn is not None:
            return key_fn()
        return board.fen()

    def store_tt(self, key: object, entry: TTEntry) -> None:
        if len(self.tt) >= self.max_tt_entries:
            self.tt.clear()
        self.tt[key] = entry


def evaluate(board: chess.Board) -> int:
    """Return a score from the side-to-move perspective."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    white_score = 0

    for piece_type in PIECE_VALUES:
        table = PST[piece_type]
        value = PIECE_VALUES[piece_type]

        for square in board.pieces(piece_type, chess.WHITE):
            white_score += value + table[square]
        for square in board.pieces(piece_type, chess.BLACK):
            white_score -= value + table[chess.square_mirror(square)]

    white_score += mobility_score(board)
    white_score += pawn_structure_score(board)
    white_score += bishop_pair_score(board)

    return white_score if board.turn == chess.WHITE else -white_score


def mobility_score(board: chess.Board) -> int:
    turn = board.turn

    board.turn = chess.WHITE
    try:
        white_mobility = board.legal_moves.count()
    finally:
        board.turn = turn

    board.turn = chess.BLACK
    try:
        black_mobility = board.legal_moves.count()
    finally:
        board.turn = turn

    return 2 * (white_mobility - black_mobility)


def pawn_structure_score(board: chess.Board) -> int:
    score = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        pawns = list(board.pieces(chess.PAWN, color))
        files = [chess.square_file(sq) for sq in pawns]

        for file_index in range(8):
            count = files.count(file_index)
            if count > 1:
                score -= sign * 12 * (count - 1)

        for square in pawns:
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            adjacent_files = [f for f in (file_index - 1, file_index + 1) if 0 <= f <= 7]

            if all(f not in files for f in adjacent_files):
                score -= sign * 10

            enemy_pawns = board.pieces(chess.PAWN, not color)
            passed = True
            for enemy_square in enemy_pawns:
                enemy_file = chess.square_file(enemy_square)
                enemy_rank = chess.square_rank(enemy_square)
                if abs(enemy_file - file_index) <= 1:
                    if color == chess.WHITE and enemy_rank > rank_index:
                        passed = False
                        break
                    if color == chess.BLACK and enemy_rank < rank_index:
                        passed = False
                        break
            if passed:
                advance = rank_index if color == chess.WHITE else 7 - rank_index
                score += sign * (20 + 8 * advance)

    return score


def bishop_pair_score(board: chess.Board) -> int:
    score = 0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 35
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 35
    return score


ENGINE = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    try:
        board = chess.Board(fen)
        move = ENGINE.choose_move(board, time_left_ms)
        if move in board.legal_moves:
            return move.uci()
    except Exception:
        pass

    # Last-resort safety path for valid non-terminal positions.
    board = chess.Board(fen)
    return next(iter(board.legal_moves)).uci()
