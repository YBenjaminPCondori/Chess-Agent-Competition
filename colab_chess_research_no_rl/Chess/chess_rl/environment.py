"""Tournament transition rules; referee history is not a policy observation."""

from dataclasses import dataclass
import math
import chess

FAILURES = frozenset({"crash", "illegal", "flag", "init", "both_failed"})


@dataclass(frozen=True)
class Finish:
    winner: bool | None
    reason: str
    void: bool = False

    @property
    def result(self):
        return (
            "*"
            if self.void
            else ("1/2-1/2" if self.winner is None else "1-0" if self.winner else "0-1")
        )


def terminal(board: chess.Board, ply_cap: int = 600) -> Finish | None:
    outcome = board.outcome()
    if outcome is not None:
        return Finish(outcome.winner, outcome.termination.name.lower())
    if board.is_repetition(3):
        return Finish(None, "threefold_repetition")
    if board.is_fifty_moves():
        return Finish(None, "fifty_moves")
    if board.ply() >= ply_cap:
        return Finish(None, "ply_cap")
    return None


def result_value(result: str, side_to_move: bool) -> float:
    if result == "1/2-1/2":
        return 0.0
    if result not in {"1-0", "0-1"}:
        raise ValueError("An unfinished game has no outcome target")
    return 1.0 if (result == "1-0") == side_to_move else -1.0


class ChessEnvironment:
    def __init__(self, fen=chess.STARTING_FEN, base_ms=120000, increment_ms=500, ply_cap=600):
        self.board = chess.Board(fen)
        if not self.board.is_valid():
            raise ValueError("Invalid standard-chess FEN")
        self.initial_fen = self.board.fen()
        self.clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
        self.increment_ms = increment_ms
        self.ply_cap = ply_cap
        self.finish = terminal(self.board, self.ply_cap)
        self.move_times = []

    def observe(self):
        return self.board.fen(), int(self.clock[self.board.turn])

    def fail(self, reason, mover=None, both=False):
        mover = self.board.turn if mover is None else mover
        self.finish = Finish(None, "both_failed", True) if both else Finish(not mover, reason)
        return self.finish

    def step(self, uci: str, spent_ms: float):
        if self.finish:
            raise RuntimeError("Cannot move after termination")
        if not math.isfinite(spent_ms) or spent_ms < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        mover = self.board.turn
        self.clock[mover] -= spent_ms
        self.move_times.append(spent_ms)
        if self.clock[mover] < 0:
            winner = None if self.board.has_insufficient_material(not mover) else not mover
            self.finish = Finish(winner, "flag")
            return self.finish
        try:
            move = chess.Move.from_uci(uci)
        except (ValueError, TypeError, AttributeError):
            return self.fail("illegal")
        if move not in self.board.legal_moves:
            return self.fail("illegal")
        self.board.push(move)
        self.clock[mover] += self.increment_ms
        self.finish = terminal(self.board, self.ply_cap)
        return self.finish
