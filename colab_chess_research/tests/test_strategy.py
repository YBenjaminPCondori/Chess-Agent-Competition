"""Bounded research fixtures. No final submission checks or real training runs."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import chess
import chess.engine
import pytest
import torch
from chess_rl.action_encoding import encode_move
from chess_rl.checkpoints import load_checkpoint, save_checkpoint, checkpoint_path
from chess_rl.config import load_config
from chess_rl.dataset import PositionDataset
from chess_rl.model import ChessPolicyValueNetSmall
from chess_rl.reproducibility import sha256
from chess_rl.research_workflow import select_winner
from chess_rl.reservations import prepare_reservations
from chess_rl.strategy_dataset import (
    SCHEMA,
    board_with_history,
    build_strategy_datasets,
    label_with_engine,
    load_source,
    load_strategy_manifest,
    normalize_row,
    split_rows,
    to_training_record,
    validate_schema,
    write_csv,
)
from chess_rl.strategy_evaluation import evaluate_suite, load_suites
from chess_rl.strategy_features import extract_features
from chess_rl.strategy_finetuning import train_candidate, load_finetuning_config
from chess_rl.strategy_taxonomy import TAXONOMY, validate_theme

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = dict(min_accuracy=0.5, max_value_mae=0.5, min_legal_rate=1.0)


def row(**changes):
    return normalize_row(
        dict(
            fen=chess.STARTING_FEN,
            source="synthetic_test",
            source_id="fixture",
            theme="tactics",
            subtheme="checks",
            label_kind="synthetic_test",
            created_at="2026-01-01T00:00:00+00:00",
            **changes,
        )
    )


def varied_rows(count=60):
    rng = random.Random(80)
    rows = []
    for index in range(count):
        board = chess.Board()
        for _ in range(12):
            if not board.legal_moves:
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.outcome() is not None:
            continue
        rows.append(
            normalize_row(
                dict(
                    fen=board.fen(),
                    source="synthetic_test",
                    source_id=f"game-{index}",
                    theme="tactics",
                    subtheme="checks",
                    best_move=next(iter(board.legal_moves)).uci(),
                    value_target=0.0,
                    label_kind="synthetic_test",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            )
        )
    return rows


def project(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    return tmp_path


def test_taxonomy():
    assert len(TAXONOMY) == 11
    for theme, subthemes in TAXONOMY.items():
        assert len(subthemes) == len(set(subthemes))
        for subtheme in subthemes:
            assert validate_theme(theme, subtheme) == (theme, subtheme)
    with pytest.raises(ValueError):
        validate_theme("openings", "forks")
    with pytest.raises(ValueError):
        validate_theme("unknown")


def test_schema_and_legal_action_mapping():
    record = row(best_move="e2e4")
    assert set(SCHEMA) <= record.keys()
    converted = to_training_record(record)
    assert converted["policy_target_distribution"] == [
        (encode_move(chess.Board(), chess.Move.from_uci("e2e4")), 1.0)
    ]
    assert PositionDataset([converted])[0]["policy_valid"]
    with pytest.raises(ValueError, match="Missing"):
        validate_schema({"fen": chess.STARTING_FEN})
    assert row()["best_move"] is None and row()["policy_target"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"fen": "8/8/8/8/8/8/8/8 w - - 0 1"},
        {"fen": "not a FEN"},
        {"best_move": "e2e5"},
        {"legal_moves": ["e2e4"]},
        {"policy_target": {"e2e5": 1.0}},
        {"policy_target": {"e2e4": -0.1, "d2d4": 1.1}},
        {"policy_target": {"e2e4": 0.5}},
        {"policy_target": {"e2e4": float("nan")}},
        {"best_move": "e2e4", "policy_target": {"d2d4": 1.0}},
        {"value_target": float("nan")},
        {"value_target": 1.01},
        {"engine_depth": -1},
        {"line": ["e2e4", "e7e4"]},
        {"history_moves": ["e2e4"], "history_start_fen": chess.STARTING_FEN},
    ],
)
def test_bad_records_fail(changes):
    raw = dict(row())
    raw.update(changes)
    with pytest.raises((ValueError, TypeError)):
        normalize_row(raw)


def test_promotion_and_pinned_en_passant():
    good = dict(row(), fen="7k/P7/8/8/8/8/8/7K w - - 0 1", legal_moves=None, best_move="a7a8n")
    record = normalize_row(good)
    assert PositionDataset([to_training_record(record)])[0]["policy_valid"]
    with pytest.raises(ValueError):
        normalize_row(
            dict(
                row(), fen="k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", legal_moves=None, best_move="e5d6"
            )
        )


def test_csv_roundtrip_and_unlabelled_line(tmp_path):
    record = row(line=["e2e4", "e7e5", "g1f3"])
    path = tmp_path / "source.csv"
    write_csv(path, [record])
    imported = list(load_source(tmp_path, dict(path=path.name)))[0]
    assert imported == dict(
        record, source_sha256=sha256(path), source_line_id=f"{sha256(path)}:row-0"
    )
    assert record["policy_target"] is None


def test_pgn_ignores_comments_and_preserves_draw_history(tmp_path):
    path = tmp_path / "source.pgn"
    path.write_text(
        '[Result "*"]\n\n1. Nf3 {PRIVATE EXPLANATION OMIT} Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 *\n'
    )
    records = list(
        load_source(tmp_path, dict(path=path.name, theme="draw_awareness", subtheme="repetition"))
    )
    assert "PRIVATE EXPLANATION" not in json.dumps(records)
    assert all(r["best_move"] is None for r in records)
    board = board_with_history(records[-1])
    assert board.is_repetition(3)
    features = extract_features(board, True)
    assert features["threefold_repetition"] is True
    assert extract_features(chess.Board(board.fen()))["threefold_repetition"] is None


def test_split_reproducible_independent_and_duplicate_exclusion():
    records = varied_rows()
    a = split_rows(records, 42)
    b = split_rows(reversed(records), 42)
    assert all(a.values())
    assert {s: {r["source_id"] for r in rows} for s, rows in a.items()} == {
        s: {r["source_id"] for r in rows} for s, rows in b.items()
    }
    assert {r["source_id"] for r in a["train"]} != {
        r["source_id"] for r in split_rows(records, 99)["train"]
    }
    duplicate = dict(records[0], source_id="another-source", split="test")
    original = dict(records[0], split="train")
    assert sum(map(len, split_rows([original, duplicate]).values())) == 0
    with pytest.raises(ValueError, match="conflicting"):
        split_rows([original, dict(original, split="test")])
    benchmark = dict(records[0], theme="annotated_benchmark_games", subtheme="rubinstein")
    assert len(split_rows([benchmark])["test"]) == 1


def test_engine_labels_side_to_move_and_pv():
    class FixtureEngine:
        id = {"name": "synthetic_test_engine"}

        def analyse(self, board, limit):
            return dict(
                pv=[chess.Move.from_uci("e2e4")],
                score=chess.engine.PovScore(chess.engine.Cp(120), board.turn),
                depth=5,
            )

    labelled = label_with_engine(row(), FixtureEngine(), dict(nodes=10))
    assert labelled["best_move"] == "e2e4" and labelled["engine_depth"] == 5
    assert labelled["value_target"] > 0 and labelled["label_kind"] == "engine"
    mate = normalize_row(dict(row(), fen="7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", legal_moves=None))
    terminal = label_with_engine(mate, FixtureEngine(), dict(nodes=10))
    assert terminal["value_target"] == -1 and terminal["best_move"] is None
    assert terminal["label_kind"] == "rule_outcome"


def test_builder_suites_resume_and_tamper(tmp_path):
    cfg = load_config(project(tmp_path), "strategy.yaml")
    cfg["run_id"] = "fixture"
    write_csv(tmp_path / "input.csv", varied_rows())
    cfg["strategy_dataset"]["sources"] = [dict(path="input.csv")]
    cfg["evaluation"].update(development_games=2, confirmation_games=2, heldout_games=2)
    prepare_reservations(tmp_path, cfg)
    manifest = build_strategy_datasets(tmp_path, cfg)
    assert build_strategy_datasets(tmp_path, cfg) == manifest
    assert all(manifest["counts"].values())
    suites, _ = load_suites(tmp_path, cfg["strategy_evaluation"], "strategy_v1")
    assert len(suites) == 11 and suites["tactics"]
    changed = deepcopy(cfg["strategy_evaluation"])
    changed["split"] = "train"
    with pytest.raises(ValueError):
        load_suites(tmp_path, changed, "fixture")
    (tmp_path / manifest["paths"]["train"]).write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        load_strategy_manifest(tmp_path, "data/strategy/strategy_v1/manifest.json")


def test_evaluation_labels_missing_and_failures():
    labelled = row(best_move="e2e4", value_target=0.5)

    def predict(fen):
        return dict(move="e2e4", ranked=["d2d4", "e2e4"], value=0.25)

    summary, _ = evaluate_suite([labelled], predict, THRESHOLDS)
    assert summary["accuracy"] == 1 and summary["top_k_agreement"] == 1
    assert summary["value_mae"] == 0.25 and summary["legal_move_rate"] == 1
    assert summary["status"] == "pass"
    assert evaluate_suite([], predict, THRESHOLDS)[0]["status"] == "not_assessable"
    assert evaluate_suite([row()], predict, THRESHOLDS)[0]["status"] == "not_assessable"

    def failure(fen):
        raise RuntimeError("synthetic test failure")

    failed, _ = evaluate_suite([labelled], failure, THRESHOLDS)
    assert failed["policy_labelled"] == 1 and failed["value_labelled"] == 1
    assert failed["status"] == "fail" and failed["counts"]["fail"] == 1 and failed["errors"] == 1


def test_finetuning_config(tmp_path):
    project(tmp_path)
    cfg = load_finetuning_config(tmp_path)
    assert cfg["strategy_finetuning"]["heads"] == dict(epochs=2, learning_rate=0.0001)
    import yaml

    path = tmp_path / "configs/strategy.yaml"
    settings = yaml.safe_load(path.read_text())
    settings["strategy_finetuning"]["heads"]["epochs"] = -1
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        load_finetuning_config(tmp_path)


@pytest.mark.parametrize("interrupt_after", [1, 3])
def test_phased_transfer_freezes_buffers_and_exact_epoch_resume(tmp_path, interrupt_after):
    cfg = load_config(ROOT, "strategy.yaml")
    cfg.update(run_id="fixture", device="cpu", deterministic=True)
    cfg["training"].update(num_workers=0)
    cfg["strategy_finetuning"]["training"].update(batch_size_cpu=4)
    cfg["strategy_finetuning"]["full"]["epochs"] = 2
    cfg["strategy_finetuning"]["steps_per_epoch"] = 1
    model = ChessPolicyValueNetSmall(1, 8, 16, dropout=0.1)
    initial_state = deepcopy(model.state_dict())
    base = save_checkpoint(tmp_path / "base", model, epoch=1)
    rows = varied_rows(8)
    replay = [to_training_record(r) for r in rows[4:6]]
    signature = dict(config=cfg, data_hashes={"fixture": "explicit synthetic regression rows"})
    phases = []

    def inspect(path):
        payload = load_checkpoint(path)
        phases.append(payload["phase"])
        if payload["phase"] == "heads":
            for name, tensor in initial_state.items():
                if name.startswith(("stem.", "trunk.")):
                    torch.testing.assert_close(
                        payload["model_state_dict"][name], tensor, rtol=0, atol=0
                    )
        assert payload["sampler"]["strategy_per_batch"] == 3
        assert payload["sampler"]["replay_per_batch"] == 1

    complete = tmp_path / "continuous"
    train_candidate(tmp_path, cfg, base, complete, rows[:4], replay, rows[6:], signature, inspect)
    assert phases == ["heads", "heads", "full", "full"]
    expected = load_checkpoint(checkpoint_path(complete))
    assert not torch.equal(
        expected["model_state_dict"]["stem.0.weight"], initial_state["stem.0.weight"]
    )

    def interrupt(path):
        inspect(path)
        if load_checkpoint(path)["global_epoch"] == interrupt_after:
            raise InterruptedError("Synthetic epoch-boundary interruption")

    resumed = tmp_path / "resumed"
    with pytest.raises(InterruptedError):
        train_candidate(
            tmp_path, cfg, base, resumed, rows[:4], replay, rows[6:], signature, interrupt
        )
    selected = train_candidate(tmp_path, cfg, base, resumed, rows[:4], replay, rows[6:], signature)
    actual = load_checkpoint(checkpoint_path(resumed))
    for name, tensor in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][name], tensor, rtol=0, atol=0)
    assert actual["sampler"] == expected["sampler"]
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    assert (
        actual["optimizer_state_dict"]["param_groups"]
        == expected["optimizer_state_dict"]["param_groups"]
    )
    for index, state in expected["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                actual["optimizer_state_dict"]["state"][index][key], value, rtol=0, atol=0
            )
    torch.testing.assert_close(actual["rng"]["torch"], expected["rng"]["torch"], rtol=0, atol=0)
    assert load_checkpoint(selected)["phase"] == "full"
    assert (
        train_candidate(tmp_path, cfg, base, resumed, rows[:4], replay, rows[6:], signature)
        == selected
    )


def test_selection_rejects_failures_and_can_choose_classical():
    candidates = [dict(id="broad"), dict(id="classical")]
    good = dict(score=0.6, games=2, candidate_failures=0, model_errors=0, void_games=0)
    general = dict(
        results=dict(
            broad=dict(opponent=dict(good, score=1, model_errors=1)), classical=dict(opponent=good)
        )
    )
    strategy = dict(
        summaries=[
            dict(candidate=c["id"], status="unlabelled", errors=0, accuracy=None)
            for c in candidates
        ]
    )
    winner, ranking = select_winner(
        candidates, general, strategy, dict(min_general_score=0, require_strategy_pass=False)
    )
    assert winner["id"] == "classical" and not ranking[0]["eligible"]
    with pytest.raises(ValueError):
        select_winner(
            candidates, general, strategy, dict(min_general_score=0, require_strategy_pass=True)
        )


def test_export_contract_is_static_and_checks_stay_final():
    for name in ("classical_agent.py", "submission_agent.py", "research_agent.py"):
        tree = ast.parse((ROOT / "templates" / name).read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_move"
        )
        assert [a.arg for a in function.args.args] == ["fen", "time_left_ms"]
        assert [ast.unparse(a.annotation) for a in function.args.args] == ["str", "int"]
        assert ast.unparse(function.returns) == "str"
    # No runtime import of exported agents or packaging occurs in this test.
    import nbformat

    for path in (ROOT / "notebooks").rglob("*.ipynb"):
        if "archive" in path.parts:
            continue
        if path.name in {"06b_export_submission.ipynb", "05_export_no_rl.ipynb"}:
            continue
        code = "\n".join(
            c.source for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code"
        )
        assert "chess_rl.export" not in code and "non_rl_export" not in code
        assert "final_checks(" not in code and "check_candidate_runtime" not in code
