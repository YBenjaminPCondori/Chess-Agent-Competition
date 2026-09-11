"""Acceptance regressions using explicitly synthetic, temporary chess fixtures."""

from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import chess
import pytest
import torch
from chess_rl.checkpoints import load_checkpoint, save_checkpoint
from chess_rl.config import load_config
from chess_rl.dataset import identity, prepare_dataset, read_jsonl, write_jsonl
from chess_rl.reproducibility import atomic_json, read_json, sha256
from chess_rl.reservations import (
    prepare_reservations,
    load_reservations,
    assert_independent,
    audit_base,
)
from chess_rl.strategy_dataset import (
    build_strategy_datasets,
    normalize_row,
    split_rows,
    to_training_record,
    write_csv,
)
from chess_rl.strategy_evaluation import evaluate_suite
from chess_rl.strategy_features import extract_features
from chess_rl.strategy_finetuning import MixedBatchSampler, finetune_strategy, train_candidate
from chess_rl.model import ChessPolicyValueNetSmall
from test_strategy import varied_rows, row, THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]


def research_fixture(tmp_path):
    cfg = load_config(ROOT, "strategy.yaml")
    cfg.update(run_id="synthetic_fixture", device="cpu", deterministic=True)
    cfg["evaluation"].update(development_games=2, confirmation_games=2, heldout_games=2)
    cfg["strategy_dataset"].update(
        dataset_id="independent_corpus", sources=[dict(path="strategy.csv")]
    )
    cfg["research_workflow"]["stages"]["self_play"] = "skipped"
    cfg["training"].update(num_workers=0)
    cfg["strategy_finetuning"]["training"].update(batch_size_cpu=4)
    cfg["strategy_finetuning"].update(
        heads=dict(epochs=1, learning_rate=1e-4),
        full=dict(epochs=1, learning_rate=3e-5),
        steps_per_epoch=1,
    )
    write_csv(tmp_path / "strategy.csv", varied_rows(60))
    broad_file = tmp_path / "broad.jsonl"
    write_jsonl(broad_file, [to_training_record(r) for r in varied_rows(150)])
    cfg["dataset"].update(jsonl_paths=[str(broad_file)], target_positions=150)
    return cfg


def test_reservations_precede_both_builders_and_cached_data_is_reaudited(tmp_path):
    cfg = research_fixture(tmp_path)
    with pytest.raises(ValueError, match="notebook 01"):
        prepare_dataset(tmp_path, cfg)
    with pytest.raises(ValueError, match="notebook 01"):
        build_strategy_datasets(tmp_path, cfg)
    reserved = prepare_reservations(tmp_path, cfg)
    assert prepare_reservations(tmp_path, cfg) == reserved
    assert set(reserved["suites"]) == {"development", "confirmation", "heldout"}
    strategy_before = build_strategy_datasets(tmp_path, cfg)
    assert "independent_corpus" in strategy_before["paths"]["train"]
    broad = prepare_dataset(tmp_path, cfg)
    assert build_strategy_datasets(tmp_path, cfg) == strategy_before
    assert prepare_dataset(tmp_path, cfg) == broad
    for split in broad["paths"]:
        assert_independent(read_jsonl(tmp_path / broad["paths"][split]), reserved)
    train = list(read_jsonl(tmp_path / broad["paths"]["train"]))
    reserved_row = next(read_jsonl(tmp_path / strategy_before["paths"]["validation"]))
    contaminated = to_training_record(reserved_row)
    contaminated["split"] = "train"
    write_jsonl(tmp_path / broad["paths"]["train"], train + [contaminated])
    broad["hashes"]["train"] = sha256(tmp_path / broad["paths"]["train"])
    atomic_json(tmp_path / "datasets/manifests/synthetic_fixture.json", broad)
    model = ChessPolicyValueNetSmall(1, 8, 16)
    base = save_checkpoint(tmp_path / "base", model, dataset_hashes=broad["hashes"])
    with pytest.raises(ValueError, match="overlaps reserved"):
        prepare_dataset(tmp_path, cfg)
    with pytest.raises(ValueError, match="overlaps reserved"):
        audit_base(tmp_path, cfg, base, broad)


def test_source_groups_position_identity_and_immutable_sources(tmp_path):
    cfg = research_fixture(tmp_path)
    reserved = prepare_reservations(tmp_path, cfg)
    group = next(g for g in reserved["reserved_groups"] if g.startswith("game-"))
    with pytest.raises(ValueError, match="overlaps"):
        assert_independent([dict(fen=chess.STARTING_FEN, game_id=group)], reserved)
    fen = next(read_jsonl(tmp_path / reserved["suites"]["confirmation"]))["fen"]
    board = chess.Board(fen)
    board.halfmove_clock += 2
    assert identity(board.fen()) == identity(fen)
    with pytest.raises(ValueError, match="overlaps"):
        assert_independent([dict(fen=board.fen(), game_id="other")], reserved)
    with (tmp_path / "strategy.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="inputs changed"):
        load_reservations(tmp_path, cfg)


def test_group_split_uses_original_game_identifier():
    a, b = varied_rows(2)
    a.update(source_game_id="same-game", split="train")
    b.update(source_game_id="same-game", split="test")
    with pytest.raises(ValueError, match="conflicting"):
        split_rows([a, b])


def test_exact_replay_quota_and_balanced_themes():
    themes = ["common"] * 99 + ["rare"]
    sampler = MixedBatchSampler(themes, 23, 64, 0.25, 42, steps=4)
    counts = Counter()
    batches = list(sampler)
    assert batches == list(sampler)
    for batch in batches:
        assert len(batch) == 64
        assert sum(i >= len(themes) for i in batch) == 16
        counts.update(themes[i] for i in batch if i < len(themes))
    assert counts == dict(common=96, rare=96)
    assert batches != list(MixedBatchSampler(themes, 23, 64, 0.25, 43, steps=4))
    with pytest.raises(ValueError, match="quotas"):
        MixedBatchSampler(themes, 23, 7, 0.25, 42)


def test_combined_and_specialist_start_independently_from_same_base(tmp_path, monkeypatch):
    import chess_rl.strategy_finetuning as module
    from chess_rl.research_workflow import self_play_inputs
    from chess_rl.candidate_registry import audit_checkpoint

    cfg = research_fixture(tmp_path)
    cfg["strategy_finetuning"]["specialists"] = [dict(id="tactics_only", themes=["tactics"])]
    prepare_reservations(tmp_path, cfg)
    build_strategy_datasets(tmp_path, cfg)
    broad = prepare_dataset(tmp_path, cfg)
    base = save_checkpoint(
        tmp_path / "checkpoints/supervised/synthetic_fixture",
        ChessPolicyValueNetSmall(1, 8, 16),
        dataset_hashes=broad["hashes"],
    )
    atomic_json(base.parent / "complete.json", dict(status="completed", fixture=True))
    atomic_json(
        tmp_path / "results/synthetic_fixture/initial_selection.json",
        dict(checkpoint=str(base.relative_to(tmp_path)), sha256=sha256(base)),
    )
    seen = []

    def observed(root, config, initial, *args):
        seen.append(initial)
        assert initial == base
        return train_candidate(root, config, initial, *args)

    monkeypatch.setattr(module, "train_candidate", observed)
    results = finetune_strategy(tmp_path, cfg)
    assert set(results) == {"combined", "tactics_only"} and seen == [base, base]
    a = load_checkpoint(tmp_path / results["combined"]["checkpoint"])
    b = load_checkpoint(tmp_path / results["tactics_only"]["checkpoint"])
    assert a["parent_sha256"] == b["parent_sha256"] == sha256(base)
    for key, value in a["model_state_dict"].items():
        torch.testing.assert_close(b["model_state_dict"][key], value, rtol=0, atol=0)
    audit_checkpoint(tmp_path, cfg, tmp_path / results["combined"]["checkpoint"])
    assert self_play_inputs(tmp_path, cfg)[0] == tmp_path / results["combined"]["checkpoint"]
    assert finetune_strategy(tmp_path, cfg) == results
    cfg["research_workflow"]["self_play_initial_checkpoint"] = str(base.relative_to(tmp_path))
    with pytest.raises(ValueError, match="strategy explicitly skipped"):
        self_play_inputs(tmp_path, cfg)


def test_head_staleness_cannot_skip_full_phase(tmp_path, monkeypatch):
    import chess_rl.strategy_finetuning as module

    cfg = research_fixture(tmp_path)
    cfg["strategy_finetuning"]["heads"]["epochs"] = 2
    cfg["strategy_finetuning"]["full"]["epochs"] = 8
    cfg["strategy_finetuning"]["training"]["early_stopping_patience"] = 1
    monkeypatch.setattr(module, "evaluate_dataset", lambda *args: dict(total_loss=1.0))
    base = save_checkpoint(tmp_path / "base", ChessPolicyValueNetSmall(1, 8, 16))
    rows = varied_rows(8)
    phases = []
    selected = train_candidate(
        tmp_path,
        cfg,
        base,
        tmp_path / "candidate",
        rows[:4],
        [to_training_record(r) for r in rows[4:6]],
        rows[6:],
        dict(data_hashes={}, config=cfg),
        lambda p: phases.append(load_checkpoint(p)["phase"]),
    )
    assert phases == ["heads", "heads", "full", "full"]
    assert load_checkpoint(selected)["phase"] == "full"


def test_acceptable_moves_metrics_and_history_limits():
    labelled = row(acceptable_moves=["e2e4", "d2d4"], value_target=0.5)

    def predict(fen):
        return dict(move="d2d4", ranked=["g1f3", "d2d4", "e2e4"], value=0.25)

    summary, positions = evaluate_suite([labelled], predict, THRESHOLDS)
    assert summary["search_move_agreement"] == 1
    assert summary["policy_top1_agreement"] == 0 and summary["policy_top3_agreement"] == 1
    assert summary["counts"]["pass"] == 1
    assert positions[0]["status"] == "pass"
    dependent = dict(labelled, history_dependent=True)
    summary, _ = evaluate_suite([dependent, row()], predict, THRESHOLDS)
    assert summary["counts"]["not_assessable"] == 2
    assert summary["history_counts"]["not_assessable"] == 1
    assert summary["search_move_agreement"] is None
    failed, _ = evaluate_suite([row()], lambda fen: dict(move="e2e5"), THRESHOLDS)
    assert failed["errors"] == 1 and failed["counts"]["fail"] == 1
    with pytest.raises(ValueError, match="acceptable_moves"):
        normalize_row(dict(labelled, acceptable_moves=["e2e5"]))


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "4k3/pp3ppp/8/3N4/2P1P3/8/PP3PPP/R3K2R w KQ - 0 1",
        "4k3/1R3ppp/P7/8/8/2P5/2P2PPP/4K3 w - - 0 1",
    ],
)
def test_detectors_are_color_symmetric(fen):
    board = chess.Board(fen)
    assert board.is_valid()
    original, mirrored = extract_features(board), extract_features(board.mirror())
    assert original["white"] == mirrored["black"]
    assert original["black"] == mirrored["white"]
    assert original["white"]["legal_mobility"] == len(list(board.legal_moves))
    assert original["black"]["legal_mobility"] is None


def test_detector_definitions_and_fortress_does_not_adjudicate():
    board = chess.Board("4k3/1R3ppp/P7/8/8/2P5/2P2PPP/4K3 w - - 40 1")
    features = extract_features(board)
    assert features["white"]["doubled_pawns"] == 1
    assert features["white"]["isolated_pawns"] == 3
    assert features["white"]["passed_pawns"] == 3
    assert features["white"]["rook_open_files"] == 1
    assert features["white"]["rooks_on_seventh"] == 1
    assert features["fortress_like_heuristic"] and board.outcome() is None
    assert features["threefold_repetition"] is None
    partial = extract_features(board, history_known=True)
    assert partial["castled_white"] is None and partial["castled_black"] is None


def test_strategy_diagnostics_never_break_general_score_ties():
    from chess_rl.research_workflow import select_winner

    candidates = [dict(id="alpha"), dict(id="zulu")]
    match = dict(score=0.6, games=256, candidate_failures=0, model_errors=0, void_games=0)
    general = dict(
        results={c["id"]: dict(greedy=match, minimax=match, classical=match) for c in candidates}
    )
    strategy = dict(
        summaries=[
            dict(candidate=c["id"], status="pass", errors=0, accuracy=i)
            for i, c in enumerate(candidates)
        ]
    )
    winner, _ = select_winner(
        candidates, general, strategy, dict(min_general_score=0, require_strategy_pass=False)
    )
    assert winner["id"] == "alpha"


def test_incomplete_collection_and_failed_game_labels_rejected():
    from chess_rl.self_play import _validate_collection_game

    with pytest.raises(ValueError, match="Incomplete"):
        _validate_collection_game(dict(records=[]), 0)
    with pytest.raises(ValueError, match="Failed self-play"):
        _validate_collection_game(
            dict(game_index=0, result="0-1", termination="crash", records=[{}]), 0
        )
    _validate_collection_game(dict(game_index=0, result="0-1", termination="crash", records=[]), 0)


def test_confirmation_claim_exclusive_creation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from chess_rl.reservations import claim_once

    path = tmp_path / "claim.json"

    def claim(candidate):
        try:
            claim_once(path, dict(candidate=candidate))
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(claim, ["first", "second"])) == 1
    saved = read_json(path)
    claim_once(path, saved)
    assert read_json(path) == saved


def test_protected_sources_and_archived_git_blobs():
    manifest = read_json(ROOT / "docs/source_manifest.json")
    for relative, digest in manifest["files"].items():
        assert sha256(ROOT / relative) == digest
    for relative, digest in read_json(ROOT / "docs/protected_sources.json").items():
        assert sha256(ROOT / relative) == digest
    git = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True
    )
    git_root = Path(git.stdout.strip()) if git.returncode == 0 else None
    protected = [ROOT / "chess_rl/classical_evaluation.py", ROOT / "chess_rl/evaluation_tuning.py"]
    protected += list((ROOT / "notebooks/no_rl").glob("*.ipynb"))
    for path in protected:
        if git_root:
            blob = subprocess.check_output(
                ["git", "show", "HEAD:" + path.relative_to(git_root).as_posix()], cwd=ROOT
            )
            assert sha256(path) == hashlib.sha256(blob).hexdigest()
    archives = list((ROOT / "notebooks/archive").glob("*/provenance.json"))
    assert archives
    for path in archives:
        records = read_json(path)
        assert len(records) == 5
        for name, info in records.items():
            assert sha256(path.parent / name) == info["sha256"]
            if git_root:
                blob = subprocess.check_output(
                    ["git", "show", info["commit"] + ":" + info["git_path"]], cwd=ROOT
                )
                assert info["sha256"] == hashlib.sha256(blob).hexdigest()
