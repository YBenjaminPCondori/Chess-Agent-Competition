from pathlib import Path
import chess
from chess_rl.environment import terminal, ChessEnvironment
from chess_rl.evaluation import harness_modules, summarize, promotion_allowed
from chess_rl.self_play import search_distribution

ROOT = Path(__file__).resolve().parents[1]


class ScriptAgent:
    name = "script"
    stderr_log = ""

    def __init__(self, moves):
        self.moves = iter(moves)

    def start(self, budget):
        return None

    def stop(self):
        return None

    def suspend(self):
        return None

    def resume(self):
        return None

    def move(self, fen, remaining):
        return next(self.moves)


def test_referee_terminal_precedence():
    referee, _ = harness_modules(ROOT)
    fens = [
        "7k/6Q1/6K1/8/8/8/8/8 b - - 100 301",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 100 301",
        "7k/8/8/8/8/8/8/KR6 w - - 100 301",
        "7k/8/8/8/8/8/8/KR6 w - - 0 301",
    ]
    for fen in fens:
        finish = terminal(chess.Board(fen))
        outcome = referee.play_match(ScriptAgent([]), ScriptAgent([]), 120000, 500, start_fen=fen)
        assert outcome.termination == finish.reason


def test_referee_played_cap():
    referee, _ = harness_modules(ROOT)
    fen = "7k/8/8/8/8/8/8/KR6 w - - 0 300"
    outcome = referee.play_match(
        ScriptAgent(["b1b2"]), ScriptAgent(["h8h7"]), 120000, 500, start_fen=fen
    )
    env = ChessEnvironment(fen)
    env.step("b1b2", 0)
    env.step("h8h7", 0)
    assert outcome.termination == env.finish.reason == "ply_cap"


def test_teacher_probabilities():
    moves, probabilities = search_distribution({"e2e4": 0.3, "d2d4": 0.1}, 0.25)
    assert moves == ["d2d4", "e2e4"]
    assert abs(probabilities.sum() - 1) < 1e-10
    assert probabilities[1] > probabilities[0]
    _, probabilities = search_distribution({"e2e4": 9995, "d2d4": 9997}, 1)
    assert probabilities.tolist() == [1, 0]


def test_paired_statistics_and_promotion():
    rows = [
        dict(
            family_id=f"pair-{i // 2}",
            score=1.0,
            candidate_failure=None,
            move_times_ms=[1, 2],
            telemetry={},
        )
        for i in range(20)
    ]
    result = summarize(rows, samples=100)
    assert result["score"] == 1 and result["interval95"] == [1, 1]
    assert promotion_allowed(result)
    rows[0]["candidate_failure"] = "illegal"
    assert not promotion_allowed(summarize(rows, samples=100))
