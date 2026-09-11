"""Deadline-aware negamax; teacher mode returns only completed full-window root scores."""

from dataclasses import dataclass, field
import math
import time
import chess
from .classical_evaluation import weighted_cp, fallback_move, tactical_score, resolve_weights
from .environment import terminal
from .time_management import Deadline, SearchTimeout, allocate_ms
from .transposition import (
    TranspositionTable,
    Entry,
    cache_key,
    EXACT,
    LOWER,
    UPPER,
    MATE_SCORE,
    to_table,
    from_table,
)


@dataclass
class SearchResult:
    move: chess.Move
    depth: int = 0
    score: float | None = None
    root_scores: dict = field(default_factory=dict)
    nodes: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None


class SearchEngine:
    def __init__(self, config=None, neural=None, version="classical"):
        self.config = config or {}
        self.evaluation_weights = resolve_weights(self.config.get("evaluation_weights"))
        self.neural = neural
        self.version = version
        self.tt = TranspositionTable(self.config.get("transposition_table_size", 100000))
        self.nodes = 0
        self.history = []
        self.deadline = Deadline(math.inf)
        self.cumulative = dict(
            moves=0,
            nodes=0,
            search_errors=0,
            fallback_moves=0,
            depths={},
            inference_calls=0,
            inference_ms=0.0,
        )

    def check(self):
        self.nodes += 1
        self.deadline.check()

    def terminal_score(self, board, ply):
        outcome = terminal(board)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        return MATE_SCORE - ply if outcome.winner == board.turn else -MATE_SCORE + ply

    def order(self, board, preferred=None):
        self.check()
        moves = list(board.legal_moves)
        ranks = {}
        if self.neural and self.neural.policy_ordering:
            output = self.neural.predict(board, self.deadline)
            if output:
                ranks = self.neural.ranks(board, moves, output[0])
        return sorted(
            moves,
            key=lambda m: (m != preferred, -tactical_score(board, m), -ranks.get(m, 0.0), m.uci()),
        )

    def leaf(self, board):
        self.check()
        classical = math.tanh(weighted_cp(board, self.evaluation_weights) / 600)
        if self.neural and self.neural.value_mix:
            output = self.neural.predict(board, self.deadline)
            if output:
                mix = self.neural.value_mix
                return (1 - mix) * classical + mix * output[1]
        return classical

    def child(self, board, move, depth, alpha, beta, ply, quiescence=False):
        board.push(move)
        self.history.append(move.uci())
        try:
            if quiescence:
                return -self.quiescence(board, -beta, -alpha, ply + 1, depth)
            return -self.negamax(board, depth, -beta, -alpha, ply + 1)
        finally:
            self.history.pop()
            board.pop()

    def quiescence(self, board, alpha, beta, ply, remaining):
        self.check()
        outcome = self.terminal_score(board, ply)
        if outcome is not None:
            return outcome
        in_check = board.is_check()
        if not in_check:
            stand_pat = self.leaf(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            if remaining <= 0:
                return stand_pat
        for move in self.order(board):
            if not in_check and not (board.is_capture(move) or move.promotion):
                continue
            score = self.child(board, move, remaining - 1, alpha, beta, ply, True)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        self.check()
        outcome = self.terminal_score(board, ply)
        if outcome is not None:
            return outcome
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, self.config.get("quiescence_depth", 4))
        original_alpha, original_beta = alpha, beta
        key = cache_key(board, self.history, self.version)
        entry = self.tt.get(key)
        preferred = None
        if entry:
            move = chess.Move.from_uci(entry.move)
            preferred = move if move in board.legal_moves else None
            # Same nominal depth keeps the teacher's root labels comparable.
            if preferred is not None and entry.depth == depth:
                score = from_table(entry.score, ply)
                if entry.flag == EXACT:
                    return score
                if entry.flag == LOWER:
                    alpha = max(alpha, score)
                if entry.flag == UPPER:
                    beta = min(beta, score)
                if alpha >= beta:
                    return score
        best_score, best_move = -math.inf, None
        for move in self.order(board, preferred):
            score = self.child(board, move, depth - 1, alpha, beta, ply)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        flag = (
            UPPER
            if best_score <= original_alpha
            else LOWER
            if best_score >= original_beta
            else EXACT
        )
        if best_move:
            self.tt.put(key, Entry(depth, to_table(best_score, ply), flag, best_move.uci()))
        return best_score

    def choose(self, board, time_left_ms, teacher=False, max_depth=None, started_at=None):
        started = time.monotonic() if started_at is None else started_at
        moves = list(board.legal_moves)
        result = SearchResult(fallback_move(board, moves))
        self.nodes = 0
        self.tt.clear()
        self.history = [board.root().fen()] + [move.uci() for move in board.move_stack]
        if not moves or time_left_ms <= 100:
            return self._finish(result, started)
        budget = allocate_ms(
            time_left_ms,
            self.config.get("time_budget_fraction", 0.05),
            self.config.get("max_budget_ms", 1000.0),
        )
        self.deadline = Deadline(started + budget / 1000)
        max_depth = max_depth or self.config.get("max_depth", 6)
        try:
            for depth in range(1, max_depth + 1):
                alpha, scores = -math.inf, {}
                best_root, best_root_score = None, -math.inf
                for move in self.order(board, result.move):
                    lower = -math.inf if teacher else alpha
                    score = self.child(board, move, depth - 1, lower, math.inf, 0)
                    scores[move.uci()] = score
                    if score > best_root_score:
                        best_root, best_root_score = move.uci(), score
                    alpha = max(alpha, score)
                self.deadline.check()
                # Only teacher scores are all exact; an equal fail-low bound is not a tie.
                best = min(scores, key=lambda uci: (-scores[uci], uci)) if teacher else best_root
                result.move, result.depth, result.score = (
                    chess.Move.from_uci(best),
                    depth,
                    scores[best],
                )
                result.root_scores = scores if teacher else {}
                if result.score > 9000 and not teacher:
                    break
        except SearchTimeout:
            pass
        except Exception as error:
            result.error = f"{type(error).__name__}: {error}"
        return self._finish(result, started)

    def _finish(self, result, started):
        result.nodes = self.nodes
        result.elapsed_ms = (time.monotonic() - started) * 1000
        self.cumulative["moves"] += 1
        self.cumulative["nodes"] += result.nodes
        self.cumulative["search_errors"] += int(result.error is not None)
        self.cumulative["fallback_moves"] += int(result.depth == 0)
        depths = self.cumulative["depths"]
        depths[str(result.depth)] = depths.get(str(result.depth), 0) + 1
        if self.neural:
            self.cumulative.update(self.neural.stats)
            self.cumulative["model_error_detail"] = self.neural.last_error
        return result


class ResearchAgent:
    def __init__(self, model=None, config=None, device="cpu", version="classical"):
        from .inference import NeuralEvaluator

        config = config or {}
        neural = (
            NeuralEvaluator(
                model,
                device,
                config.get("policy_ordering", True),
                config.get("value_eval_mix", 0.5),
            )
            if model is not None
            else None
        )
        self.engine = SearchEngine(config, neural, version)

    def get_move(self, fen: str, time_left_ms: int) -> str:
        started = time.monotonic()
        board = chess.Board(fen)
        return self.engine.choose(board, time_left_ms, started_at=started).move.uci()
