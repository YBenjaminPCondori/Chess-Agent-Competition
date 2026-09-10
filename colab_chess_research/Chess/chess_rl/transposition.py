from dataclasses import dataclass

EXACT, LOWER, UPPER = "exact", "lower", "upper"
MATE_SCORE = 10000.0


def position_key(board):
    return (
        board.board_fen(),
        board.turn,
        board.castling_rights,
        board.ep_square if board.has_legal_en_passant() else None,
    )


def cache_key(board, history, version):
    # Full ordered search history is conservative: no graph-history score contamination.
    return (position_key(board), board.halfmove_clock, board.ply(), tuple(history), version)


def to_table(score, ply):
    return score + ply if score > 9000 else score - ply if score < -9000 else score


def from_table(score, ply):
    return score - ply if score > 9000 else score + ply if score < -9000 else score


@dataclass(frozen=True)
class Entry:
    depth: int
    score: float
    flag: str
    move: str


class TranspositionTable:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.entries = {}

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, entry):
        if self.capacity <= 0:
            return
        if key not in self.entries and len(self.entries) >= self.capacity:
            self.entries.pop(next(iter(self.entries)))
        self.entries[key] = entry

    def clear(self):
        self.entries.clear()
