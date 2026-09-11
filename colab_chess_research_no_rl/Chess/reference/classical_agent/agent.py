"""Classical tree-search agent for AI Chessathon.

The platform imports this file and calls:

    get_move(fen: str, time_left_ms: int) -> str

This baseline intentionally uses only python-chess and the standard library. It is
structured to be readable and safe to experiment with: legal fallback first,
iterative deepening under a soft deadline, negamax with alpha-beta pruning,
capture-only quiescence, move ordering, and a bounded transposition table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess


MATE_SCORE = 100_000
INF = 1_000_000

EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

CHECK_INTERVAL = 64
MAX_Q_DEPTH = 4
MAX_TT_ENTRIES = 100_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


# Piece-square tables are indexed from White's perspective. Black pieces use
# mirrored square indices.
PAWN_PST = (
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
)

KNIGHT_PST = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)

BISHOP_PST = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)

ROOK_PST = (
    0, 0, 5, 10, 10, 5, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 5, 5, 0, 0, 0,
)

QUEEN_PST = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)

KING_MIDGAME_PST = (
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
)

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_MIDGAME_PST,
}


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move_uci: str | None


class SearchTimeout(Exception):
    """Raised internally when the soft search deadline is reached."""


class Engine:
    def __init__(self) -> None:
        self.deadline_ns = 0
        self.nodes = 0
        self.tt: dict[object, TTEntry] = {}

    def choose_move(self, board: chess.Board, time_left_ms: int) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        if len(legal_moves) == 1:
            return legal_moves[0]

        fallback = self.fallback_move(board, legal_moves)
        if time_left_ms <= 100:
            return fallback

        budget_ms = self.allocate_time_ms(board, time_left_ms)
        self.deadline_ns = time.perf_counter_ns() + budget_ms * 1_000_000
        self.nodes = 0

        best_move = fallback
        previous_best: chess.Move | None = None

        try:
            for depth in range(1, 64):
                depth_best, _depth_score = self.search_root(board, depth, previous_best)
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
        """Return a conservative soft budget for this move in milliseconds."""
        if time_left_ms < 250:
            return 20
        if time_left_ms < 1_000:
            return max(20, time_left_ms // 16)
        if time_left_ms < 5_000:
            return min(180, max(45, time_left_ms // 25))

        expected_moves_left = max(18, 45 - board.fullmove_number)
        budget = time_left_ms // expected_moves_left
        budget = min(850, max(90, budget))

        # The harness has a watchdog grace, but the competition clock is wall time.
        safety_margin = max(20, min(120, time_left_ms // 80))
        return max(15, budget - safety_margin)

    def out_of_time(self) -> bool:
        return time.perf_counter_ns() >= self.deadline_ns

    def check_time(self) -> None:
        self.nodes += 1
        if self.nodes % CHECK_INTERVAL == 0 and self.out_of_time():
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

        terminal_score = terminal_value(board, ply)
        if terminal_score is not None:
            return terminal_score
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, 0)

        original_alpha = alpha
        original_beta = beta
        key = self.position_key(board)
        entry = self.tt.get(key)
        tt_best: chess.Move | None = None

        if entry is not None:
            tt_best = legal_uci_to_move(board, entry.best_move_uci)
            if entry.depth >= depth:
                if entry.flag == EXACT:
                    return entry.score
                if entry.flag == LOWER_BOUND:
                    alpha = max(alpha, entry.score)
                elif entry.flag == UPPER_BOUND:
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

        best_score = -INF
        best_move: chess.Move | None = None

        for move in self.ordered_moves(board, tt_best):
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        flag = EXACT
        if best_score <= original_alpha:
            flag = UPPER_BOUND
        elif best_score >= original_beta:
            flag = LOWER_BOUND

        self.store_tt(key, TTEntry(depth, best_score, flag, best_move.uci() if best_move else None))
        return best_score

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int, q_depth: int) -> int:
        self.check_time()

        terminal_score = terminal_value(board, ply)
        if terminal_score is not None:
            return terminal_score

        if q_depth >= MAX_Q_DEPTH:
            return evaluate(board)

        if board.is_check():
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
            best = -INF
            for move in self.sort_moves(board, moves, None):
                board.push(move)
                try:
                    score = -self.quiescence(board, -beta, -alpha, ply + 1, q_depth + 1)
                finally:
                    board.pop()
                best = max(best, score)
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            return best

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        noisy_moves = [
            move
            for move in board.legal_moves
            if board.is_capture(move) or move.promotion is not None
        ]
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
        moves.sort(key=lambda move: move_order_score(board, move, preferred), reverse=True)
        return moves

    def fallback_move(self, board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        return max(legal_moves, key=lambda move: fallback_score(board, move))

    def position_key(self, board: chess.Board) -> object:
        public_key = getattr(board, "transposition_key", None)
        if public_key is not None:
            return public_key()
        private_key = getattr(board, "_transposition_key", None)
        if private_key is not None:
            return private_key()
        return board.board_fen(), board.turn, board.castling_rights, board.ep_square

    def store_tt(self, key: object, entry: TTEntry) -> None:
        if len(self.tt) >= MAX_TT_ENTRIES:
            self.tt.clear()
        self.tt[key] = entry


def legal_uci_to_move(board: chess.Board, move_uci: str | None) -> chess.Move | None:
    if move_uci is None:
        return None
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def terminal_value(board: chess.Board, ply: int) -> int | None:
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if board.is_fifty_moves():
        return 0
    return None


def evaluate(board: chess.Board) -> int:
    """Return a score from the side-to-move perspective."""
    terminal_score = terminal_value(board, 0)
    if terminal_score is not None:
        return terminal_score

    white_score = 0
    for piece_type, value in PIECE_VALUES.items():
        table = PIECE_SQUARE_TABLES[piece_type]
        for square in board.pieces(piece_type, chess.WHITE):
            white_score += value + table[square]
        for square in board.pieces(piece_type, chess.BLACK):
            white_score -= value + table[chess.square_mirror(square)]

    white_score += activity_score(board)
    white_score += pawn_structure_score(board)
    white_score += bishop_pair_score(board)

    return white_score if board.turn == chess.WHITE else -white_score


def activity_score(board: chess.Board) -> int:
    score = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for square in board.pieces(piece_type, color):
                score += sign * len(board.attacks(square))
    return 2 * score


def pawn_structure_score(board: chess.Board) -> int:
    score = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        pawns = list(board.pieces(chess.PAWN, color))
        pawn_files = [chess.square_file(square) for square in pawns]

        for file_index in range(8):
            doubled_count = pawn_files.count(file_index)
            if doubled_count > 1:
                score -= sign * 12 * (doubled_count - 1)

        for square in pawns:
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            adjacent_files = {file_index - 1, file_index + 1} & set(range(8))

            if all(adjacent_file not in pawn_files for adjacent_file in adjacent_files):
                score -= sign * 10

            if is_passed_pawn(board, square, color):
                advance = rank_index if color == chess.WHITE else 7 - rank_index
                score += sign * (20 + 8 * advance)

    return score


def is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    for enemy_square in board.pieces(chess.PAWN, not color):
        enemy_file = chess.square_file(enemy_square)
        enemy_rank = chess.square_rank(enemy_square)
        if abs(enemy_file - file_index) > 1:
            continue
        if color == chess.WHITE and enemy_rank > rank_index:
            return False
        if color == chess.BLACK and enemy_rank < rank_index:
            return False
    return True


def bishop_pair_score(board: chess.Board) -> int:
    score = 0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 35
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 35
    return score


def move_order_score(
    board: chess.Board,
    move: chess.Move,
    preferred: chess.Move | None,
) -> int:
    if preferred is not None and move == preferred:
        return 1_000_000

    score = fallback_score(board, move)
    if board.gives_check(move):
        score += 500
    return score


def fallback_score(board: chess.Board, move: chess.Move) -> int:
    score = 0
    if move.promotion is not None:
        score += 8_000 + PIECE_VALUES.get(move.promotion, 0)

    if board.is_capture(move):
        attacker = board.piece_at(move.from_square)
        captured = board.piece_at(move.to_square)
        if captured is None and board.is_en_passant(move):
            captured_value = PIECE_VALUES[chess.PAWN]
        else:
            captured_value = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
        attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        score += 10_000 + 10 * captured_value - attacker_value

    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
        score += 20

    return score


ENGINE = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move for the side to move in ``fen``."""
    try:
        board = chess.Board(fen)
        move = ENGINE.choose_move(board, time_left_ms)
        if move in board.legal_moves:
            return move.uci()
    except Exception:
        pass

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    return legal_moves[0].uci() if legal_moves else chess.Move.null().uci()
