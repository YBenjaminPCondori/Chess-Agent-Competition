"""Classical features, search integration and bounded match-based tuning; no submission checks."""

from copy import deepcopy
import csv
import math
from pathlib import Path
import shutil
import chess
import pytest
from chess_rl.classical_evaluation import (
    DEFAULT_WEIGHTS,
    evaluate,
    evaluate_cp,
    evaluation_breakdown,
    position_features,
    resolve_weights,
)
from chess_rl.dataset import write_jsonl
from chess_rl import evaluation_tuning as tuning
from chess_rl.non_rl import candidate_spec, classical_candidate, load_no_rl_config
from chess_rl.reproducibility import read_json
from chess_rl.search import SearchEngine

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "fen, expected",
    [
        ("rnbqkb1r/ppppp1pp/5p1n/8/8/N4P2/PPPPP1PP/R1BQKBNR w KQkq - 0 3", 0),
        ("rnbq1bnr/p2ppkpp/5p2/1pp5/4P3/5P1P/PPPP2P1/RNBQKBNR b Q - 2 7", -22),
        ("rnb2bnr/p3p2p/q2pkpp1/1B6/3PPP2/2p1KNPP/PPP5/RNBQ3R w - - 3 15", 238),
        ("rn2qbn1/p3p3/b2pkp2/4P3/P2P1Ppr/R1P1K1P1/2PNNR2/2B2Q2 b - - 1 28", 96),
        ("r3qb2/p3p3/n2pP3/2k1Pp2/P3QP1P/2P1K1p1/1BP3R1/3R2N1 w - - 0 46", 672),
        ("8/4p3/2q1Pn2/p3Pp1P/k2K1b2/2P1B3/1QP5/r5NR b - - 1 71", -97),
        ("8/r3p1bP/4P2n/p3P3/8/k3K2N/2P2Q2/8 w - - 4 89", 330),
        ("3B4/4p3/4Pb2/p7/8/8/2k1K3/8 w - - 2 116", -181),
    ],
)
def test_default_scores_preserve_original_evaluator(fen, expected):
    # Frozen scores from the previous evaluator on seeded legal-playout positions.
    board = chess.Board(fen)
    assert evaluate_cp(board) == expected
    assert evaluate_cp(board.mirror()) == expected
    assert sum(row["contribution_cp"] for row in evaluation_breakdown(board)) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown": 1},
        {"activity": float("nan")},
        {"activity": float("inf")},
        {"material": True},
        {"bishop_pair": "35"},
        [1, 2],
    ],
)
def test_invalid_coefficients_are_rejected(overrides):
    with pytest.raises(ValueError):
        resolve_weights(overrides)
    with pytest.raises(ValueError):
        SearchEngine({"evaluation_weights": overrides})


def test_features_and_weights_reach_search_leaves_and_continuations():
    board = chess.Board("6k1/8/8/8/8/2P5/2P5/6K1 w - - 0 1")
    features = position_features(board)
    assert features["doubled_pawns"] == 1
    assert features["isolated_pawns"] == 2
    assert features["passed_pawns"] == 2
    assert features["passed_pawn_advance"] == 3
    weights = resolve_weights({"passed_pawns": 80})
    assert weights["material"] == DEFAULT_WEIGHTS["material"]
    assert evaluate_cp(board, weights) - evaluate_cp(board) == 120
    assert evaluate(board, {"passed_pawns": 80}) == evaluate(board, weights)
    baseline = SearchEngine({"quiescence_depth": 0})
    candidate = SearchEngine({"quiescence_depth": 0, "evaluation_weights": weights})
    assert candidate.leaf(board) == evaluate(board, weights)
    assert candidate.leaf(board) > baseline.leaf(board)
    original = board.fen()
    assert candidate.negamax(board, 1, -math.inf, math.inf, 0) > baseline.negamax(
        board, 1, -math.inf, math.inf, 0
    )
    assert board.fen() == original
    assert candidate.history == []
    before = evaluate_cp(board, weights)
    board.turn = not board.turn
    assert evaluate_cp(board, weights) == -before


def test_random_search_is_seeded_bounded_and_distinct():
    ranges = {"bishop_pair": [10, 60], "isolated_pawns": [-20, 0]}
    proposals = tuning.propose_weights(DEFAULT_WEIGHTS, ranges, 8, 42)
    assert proposals == tuning.propose_weights(DEFAULT_WEIGHTS, ranges, 8, 42)
    assert proposals != tuning.propose_weights(DEFAULT_WEIGHTS, ranges, 8, 43)
    assert len({tuple(p.items()) for p in proposals}) == 8
    for row in proposals:
        assert row["material"] == 1
        assert 10 <= row["bishop_pair"] <= 60
        assert -20 <= row["isolated_pawns"] <= 0
    for bad in ({}, {"bogus": [0, 1]}, {"activity": [4, 1]}, {"activity": [0, math.inf]}):
        with pytest.raises(ValueError):
            tuning.propose_weights(DEFAULT_WEIGHTS, bad, 2, 42)
    with pytest.raises(ValueError, match="distinct"):
        tuning.propose_weights(DEFAULT_WEIGHTS, {"activity": [2, 2]}, 1, 42)


def opening_rows():
    return [
        dict(
            fen=f"6k1/8/8/8/8/8/{pawns}/6K1 b - - 0 300",
            opening_id=f"fixture-{i}",
            family_id=f"fixture-{i}",
        )
        for i, pawns in enumerate(("6PP", "5P1P", "4P2P"))
    ]


def test_tuning_partitions_exclude_confirmation_and_heldout():
    rows = opening_rows()
    screening, confirmation = tuning.split_tuning_openings(rows[:2], rows[2:], 2, 2)
    assert screening == rows[:1] and confirmation == rows[1:2]
    with pytest.raises(ValueError, match="even"):
        tuning.split_tuning_openings(rows[:2], rows[2:], 3, 2)
    with pytest.raises(ValueError, match="too small"):
        tuning.split_tuning_openings(rows[:1], rows[2:], 2, 2)
    with pytest.raises(ValueError, match="disjoint"):
        tuning.split_tuning_openings(rows[:2], rows[:1], 2, 2)
    rows[1]["family_id"] = rows[0]["family_id"]
    with pytest.raises(ValueError, match="disjoint"):
        tuning.split_tuning_openings(rows[:2], rows[2:], 2, 2)


def positive_summary():
    # Synthetic selection-logic fixture, never a measured chess-strength result.
    return dict(
        score=0.8,
        wins=8,
        draws=0,
        losses=2,
        games=10,
        interval95=[0.6, 1.0],
        candidate_failures=0,
        void_games=0,
        search_errors=0,
        model_errors=0,
    )


def test_confirmation_requires_positive_interval_score_and_clean_games():
    assert tuning.confirmed_improvement(positive_summary(), 0.55)
    for update in (
        {"interval95": [None, None]},
        {"interval95": [0.5, 1]},
        {"score": 0.51},
        {"score": None},
        {"candidate_failures": 1},
        {"void_games": 1},
        {"search_errors": 1},
        {"model_errors": 1},
        {"failed_terminations": 1},
        {"opponent_search_errors": 1},
    ):
        assert not tuning.confirmed_improvement({**positive_summary(), **update}, 0.55)


@pytest.fixture
def tuning_project(tmp_path):
    for name in ("chess_rl", "configs", "templates", "reference"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = load_no_rl_config(tmp_path)
    cfg["non_rl"]["tuning"].update(trials=1, games_per_trial=2, confirmation_games=2)
    cfg["evaluation"].update(development_games=2, heldout_games=2, bootstrap_samples=10)
    rows = opening_rows()
    write_jsonl(tmp_path / "datasets/openings/development.jsonl", rows[:2])
    write_jsonl(tmp_path / "datasets/openings/heldout.jsonl", rows[2:])
    return tmp_path, cfg


def test_actual_bounded_tuning_matches_resume_and_keep_uncertain_baseline(
    tuning_project, monkeypatch
):
    import chess_rl.evaluation as evaluation

    root, cfg = tuning_project
    preserved = evaluation.harness_modules(ROOT)
    monkeypatch.setattr(evaluation, "harness_modules", lambda _: preserved)
    with pytest.raises(RuntimeError, match="notebook 03"):
        candidate_spec(root, cfg)
    result = tuning.tune_classical_weights(root, cfg)
    assert result["accepted"] is False
    assert result["selected_weights"] == DEFAULT_WEIGHTS
    assert result["trials"][0]["summary"]["draws"] == 2
    assert result["trials"][0]["summary"]["search_errors"] == 0
    assert result["confirmation"] is None
    assert len(list((root / "results" / cfg["run_id"] / "matches").glob("*/game-*.pgn"))) == 2
    monkeypatch.setattr(tuning, "_match", lambda *args: pytest.fail("Resume started another match"))
    assert tuning.tune_classical_weights(root, cfg) == result
    assert candidate_spec(root, cfg)["config"]["search"]["evaluation_weights"] == DEFAULT_WEIGHTS
    changed = deepcopy(cfg)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="inputs changed"):
        tuning.selected_config(root, changed)
    assert not (root / "checkpoints").exists()


def test_confirmed_weights_are_saved_and_propagate_to_runtime(tuning_project, monkeypatch):
    root, cfg = tuning_project
    original = deepcopy(cfg)
    calls = []

    def fixture_match(root, cfg, candidate, baseline, suite, name):
        calls.append((name, [row["opening_id"] for row in suite]))
        assert (
            read_json(candidate / "config.json")["search"]["evaluation_weights"] != DEFAULT_WEIGHTS
        )
        assert (
            read_json(baseline / "config.json")["search"]["evaluation_weights"] == DEFAULT_WEIGHTS
        )
        return positive_summary()

    monkeypatch.setattr(tuning, "_match", fixture_match)
    result = tuning.tune_classical_weights(root, cfg)
    assert result["accepted"]
    assert calls == [("weight-trial-001", ["fixture-0"]), ("weight-confirmation", ["fixture-1"])]
    directory = root / "results" / cfg["run_id"] / "no_rl/weight_tuning"
    assert read_json(directory / "selected_weights.json") == result["selected_weights"]
    with (directory / "trials.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["score"] == "0.8"
    spec = candidate_spec(root, cfg)
    runtime = classical_candidate(root, spec["config"])
    assert (
        read_json(runtime / "config.json")["search"]["evaluation_weights"]
        == result["selected_weights"]
    )
    assert cfg == original
    assert not (root / "checkpoints").exists()
