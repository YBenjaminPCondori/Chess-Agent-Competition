"""Bounded integration tests use temporary fixture data, not research training artifacts."""

from copy import deepcopy
from pathlib import Path
import chess
import pytest
from chess_rl.config import load_config
from chess_rl.dataset import make_record, write_jsonl
from chess_rl.training import fit_supervised
from chess_rl.checkpoints import save_checkpoint, load_checkpoint
from chess_rl.model import ChessPolicyValueNetSmall
from chess_rl.reproducibility import sha256
from chess_rl.self_play import import_baseline, collect_game
from chess_rl.league import League
from chess_rl.evaluation import harness_modules
from chess_rl.search import SearchEngine

ROOT = Path(__file__).resolve().parents[1]


def test_one_epoch_and_resume(tmp_path):
    cfg = load_config(ROOT)
    cfg["run_id"] = "unit-fixture"
    cfg["device"] = "cpu"
    cfg["model"].update(channels=8, residual_blocks=1, value_head_hidden=16)
    cfg["training"].update(max_epochs=1, num_workers=0, batch_size_cpu=2)
    paths = {}
    for split in ("train", "val", "test"):
        board = chess.Board()
        records = [
            make_record(
                dict(fen=board.fen(), game_id=f"{split}-{i}"),
                dict(best_move_uci=move.uci(), value_target=0.0, value_target_kind="fixture"),
            )
            for i, move in enumerate(list(board.legal_moves)[:4])
        ]
        path = tmp_path / (split + ".jsonl")
        write_jsonl(path, records)
        paths[split] = path.name
    manifest = dict(paths=paths, hashes={k: sha256(tmp_path / v) for k, v in paths.items()})
    result = fit_supervised(tmp_path, cfg, manifest)
    payload = load_checkpoint(result)
    assert payload["epoch"] == 1 and payload["update"] == 2
    assert fit_supervised(tmp_path, cfg, manifest) == result
    changed = deepcopy(cfg)
    changed["training"]["learning_rate"] = 0.01
    with pytest.raises(ValueError):
        fit_supervised(tmp_path, changed, manifest)


def test_original_classical_import():
    get_move = import_baseline(ROOT / "reference/classical_agent/agent.py", 42)
    move = get_move(chess.STARTING_FEN, 1)
    assert chess.Move.from_uci(move) in chess.Board().legal_moves


def test_collector_finishes_fixture_game(tmp_path, monkeypatch):
    harness_modules(ROOT)
    import harness.rules

    monkeypatch.setattr(
        harness.rules, "OPENINGS", [("near-cap fixture", "7k/8/8/8/8/8/8/KR6 w - - 0 300")]
    )
    # Keep helper resolution on the preserved source; no full tournament games are run here.
    import chess_rl.self_play as self_play

    monkeypatch.setattr(self_play, "harness_modules", lambda root: harness_modules(ROOT))
    cfg = load_config(ROOT)
    cfg["device"] = "cpu"
    cfg["search"].update(quiescence_depth=0, max_depth=1, max_budget_ms=3000)
    cfg["self_play"]["teacher_max_depth"] = 1
    model = ChessPolicyValueNetSmall(1, 8, 16)
    checkpoint = save_checkpoint(tmp_path / "fixtures", model, epoch=0, update=0)
    league = League(tmp_path, cfg, checkpoint)
    relative = str(checkpoint.relative_to(tmp_path))
    monkeypatch.setattr(
        league, "opponent", lambda seed: (("neural", relative, 1.0), [("neural", relative, 1.0)])
    )
    game = collect_game(tmp_path, cfg, league, 1, 0, set())
    assert game["termination"] in {"ply_cap", "checkmate", "stalemate"}
    assert game["errors"] == {}
    assert len(game["moves"]) <= 2
    assert all(row["game_result"] == game["result"] for row in game["records"])


@pytest.mark.parametrize("moves", [[], ["e2e4", "d7d5"], ["d2d4", "g8f6", "c2c4"]])
def test_pruned_search_chooses_a_teacher_best_move(moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    cfg = dict(max_depth=2, quiescence_depth=0, max_budget_ms=10000)
    teacher = SearchEngine(cfg).choose(board, 1000000, teacher=True)
    normal = SearchEngine(cfg).choose(board, 1000000)
    assert teacher.depth == normal.depth == 2
    assert teacher.root_scores[normal.move.uci()] == max(teacher.root_scores.values())


def test_jsonl_dataset_preparation_and_resume(tmp_path):
    import random
    from chess_rl.dataset import prepare_dataset, prepare_openings

    cfg = load_config(ROOT)
    cfg["run_id"] = "dataset-fixture"
    rng = random.Random(77)
    records = []
    for game_index in range(30):
        board = chess.Board()
        for _ in range(10):
            board.push(rng.choice(list(board.legal_moves)))
        if board.outcome() is None:
            move = next(iter(board.legal_moves))
            records.append(
                make_record(
                    dict(fen=board.fen(), game_id=f"fixture-{game_index}"),
                    dict(best_move_uci=move.uci(), value_target=0.0, value_target_kind="fixture"),
                )
            )
    path = tmp_path / "import.jsonl"
    write_jsonl(path, records)
    cfg["dataset"].update(jsonl_paths=[str(path)], target_positions=len(records))
    prepare_openings(tmp_path)
    manifest = prepare_dataset(tmp_path, cfg)
    assert all(count > 0 for count in manifest["counts"].values())
    assert prepare_dataset(tmp_path, cfg) == manifest


def test_real_harness_process_adapter(tmp_path, monkeypatch):
    import shutil
    import chess_rl.evaluation as evaluation
    from chess_rl.evaluation import build_research_agent, run_matchup

    cfg = load_config(ROOT)
    cfg["run_id"] = "process-fixture"
    cfg["search"].update(max_depth=1, quiescence_depth=0)
    shutil.copytree(
        ROOT / "chess_rl", tmp_path / "chess_rl", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(
        ROOT / "templates", tmp_path / "templates", ignore=shutil.ignore_patterns("__pycache__")
    )
    checkpoint = save_checkpoint(
        tmp_path / "fixture", ChessPolicyValueNetSmall(1, 8, 16), epoch=0, update=0
    )
    candidate = build_research_agent(tmp_path, checkpoint, cfg)
    actual_referee, actual_sandbox = harness_modules(ROOT)
    monkeypatch.setattr(evaluation, "harness_modules", lambda _: (actual_referee, actual_sandbox))
    openings = [
        dict(
            fen="7k/8/8/8/8/8/8/KR6 w - - 0 300", opening_id="fixture-cap", family_id="fixture-cap"
        )
    ]
    summary = run_matchup(
        tmp_path,
        candidate,
        ROOT / "reference/starter/baselines/greedy",
        openings,
        cfg,
        "near-cap",
        games=2,
    )
    assert summary["games"] == 2 and summary["candidate_failures"] == 0
    assert summary["model_errors"] == 0 and summary["scored_games"] == 2
