"""Stage boundaries and immutable selection, with stubbed games and optimization."""

from copy import deepcopy
from pathlib import Path
import chess
import pytest
from chess_rl.config import load_config
from chess_rl.dataset import write_jsonl
from chess_rl.reproducibility import read_json, sha256
from chess_rl.reservations import prepare_reservations, evidence

ROOT = Path(__file__).resolve().parents[1]


def summary(score=0.6):
    return dict(
        score=score,
        games=2,
        interval95=[0.5, 0.7],
        candidate_failures=0,
        model_errors=0,
        void_games=0,
    )


def classical_config(root):
    cfg = load_config(ROOT, "strategy.yaml")
    cfg["run_id"] = "fixture"
    cfg["evaluation"].update(development_games=2, confirmation_games=2, heldout_games=2)
    cfg["research_workflow"]["stages"] = dict(
        broad="skipped", strategy="skipped", self_play="skipped"
    )
    prepare_reservations(root, cfg)
    return cfg


def test_generation_training_boundary_and_resume(tmp_path, monkeypatch):
    import chess_rl.self_play as module

    cfg = classical_config(tmp_path)
    cfg["self_play"].update(iterations=1, games_per_iteration=2)
    initial = tmp_path / "fixture-initial.pt"
    initial.write_bytes(b"synthetic test checkpoint; model loading is stubbed")
    train_board = chess.Board()
    train_board.push_uci("e2e4")
    record = dict(fen=train_board.fen(), game_id="synthetic", split="train")
    manifest = dict(paths={}, hashes={}, reservations=evidence(tmp_path, cfg))
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        write_jsonl(path, [record] if split == "train" else [])
        manifest["paths"][split] = path.name
        manifest["hashes"][split] = sha256(path)
    for split in ("development", "heldout"):
        write_jsonl(
            tmp_path / "datasets/openings" / f"{split}.jsonl", [dict(fen=chess.STARTING_FEN)]
        )
    calls = []

    def collect(*args):
        calls.append("collect")
        return dict(records=[record], game_index=args[4], result="1/2-1/2", termination="fixture")

    def train(root, cfg, champion, replay, supervised, iteration, replay_manifest):
        calls.append("train")
        assert len(replay) == 2 and len(supervised) == 1
        path = tmp_path / "fixture-candidate.pt"
        path.write_bytes(b"synthetic test candidate")
        return path

    monkeypatch.setattr(module, "collect_game", collect)
    monkeypatch.setattr(module, "train_iteration", train)
    monkeypatch.setattr(module, "build_research_agent", lambda *args: tmp_path)
    monkeypatch.setattr(module, "run_matchup", lambda *args: summary())
    with pytest.raises(FileNotFoundError):
        module.train_and_promote(tmp_path, cfg, initial, manifest)
    assert not calls
    generated = module.generate_self_play(tmp_path, cfg, initial, manifest)
    assert generated["status"] == "completed"
    assert calls == ["collect", "collect"]
    assert module.generate_self_play(tmp_path, cfg, initial, manifest) == generated
    assert calls == ["collect", "collect"]
    assert not (tmp_path / "results/fixture/self_play/iteration-002").exists()
    game_path = tmp_path / next(iter(generated["files"]))
    original = game_path.read_bytes()
    game_path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        module.train_and_promote(tmp_path, cfg, initial, manifest)
    game_path.write_bytes(original)
    retained = module.train_and_promote(tmp_path, cfg, initial, manifest)
    assert retained == initial
    assert calls == ["collect", "collect", "train"]
    assert (tmp_path / "checkpoints/self_play/fixture/complete.json").exists()
    assert module.train_and_promote(tmp_path, cfg, initial, manifest) == initial
    assert calls.count("train") == 1


def test_registry_independence_and_freeze_before_heldout(tmp_path, monkeypatch):
    import chess_rl.research_workflow as workflow
    import chess_rl.strategy_evaluation as evaluation
    from chess_rl.candidate_registry import load_candidate_registry

    cfg = classical_config(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("fixture = 1\n")
    monkeypatch.setattr(workflow, "source_hashes", lambda root: {"source.py": sha256(source)})
    monkeypatch.setattr(workflow, "_adapter", lambda *args: tmp_path)
    monkeypatch.setattr(evaluation, "CandidatePredictor", lambda *args: lambda fen: None)
    registry = load_candidate_registry(tmp_path, cfg)
    assert [c["id"] for c in registry["candidates"]] == ["classical"]
    assert all(s["status"] == "skipped" for s in registry["stages"].values())
    strategy = evaluation.evaluate_candidates(tmp_path, cfg)
    assert not (tmp_path / "results/fixture/general_evaluation.json").exists()
    assert all(s["status"] == "not_assessable" for s in strategy["summaries"])
    monkeypatch.setattr(workflow, "run_matchup", lambda *args: summary())
    workflow.general_evaluation(tmp_path, cfg)
    directory = tmp_path / "results/fixture"

    def heldout(*args):
        lock = read_json(directory / "winner_lock.json")
        assert lock["selected_id"] == "classical"
        assert lock["evaluation_complete"] is False
        assert lock["checkpoint"] is None
        assert lock["stages"]["self_play"]["status"] == "skipped"
        return summary()

    monkeypatch.setattr(workflow, "run_matchup", heldout)
    path = workflow.freeze_winner(tmp_path, cfg)
    frozen = read_json(path)
    assert frozen["evaluation_complete"] and len(frozen["config_sha256"]) == 64
    assert frozen["confirmation"]["status"] == "not_applicable"
    assert workflow.freeze_winner(tmp_path, cfg) == path
    changed = deepcopy(cfg)
    changed["research_workflow"]["selection"]["min_general_score"] = 0.1
    with pytest.raises(ValueError, match="already frozen"):
        workflow.freeze_winner(tmp_path, changed)
    source.write_text("fixture = 2\n")
    with pytest.raises(ValueError, match="registry inputs changed"):
        workflow.freeze_winner(tmp_path, cfg)


@pytest.mark.parametrize(
    "score,lower,failures,accepted",
    [
        (0.55, 0.501, 0, True),
        (0.549, 0.51, 0, False),
        (0.60, 0.50, 0, False),
        (0.65, 0.60, 1, False),
    ],
)
def test_one_confirmation_challenger_and_thresholds(
    tmp_path, monkeypatch, score, lower, failures, accepted
):
    import chess_rl.research_workflow as workflow

    cfg = classical_config(tmp_path)
    calls = []
    challenger, incumbent = dict(id="combined"), dict(id="classical")
    monkeypatch.setattr(workflow, "_adapter", lambda *args: tmp_path)

    def match(*args):
        calls.append(args)
        return dict(summary(score), interval95=[lower, 0.8], candidate_failures=failures)

    monkeypatch.setattr(workflow, "run_matchup", match)
    result = workflow.confirm_challenger(
        tmp_path, cfg, challenger, incumbent, "development-hash", "registry-hash"
    )
    assert result["accepted"] is accepted
    assert len(calls) == 1
    assert (
        workflow.confirm_challenger(
            tmp_path, cfg, challenger, incumbent, "development-hash", "registry-hash"
        )
        == result
    )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="never try a second"):
        workflow.confirm_challenger(
            tmp_path, cfg, dict(id="next"), incumbent, "development-hash", "registry-hash"
        )
    renamed = deepcopy(cfg)
    renamed["research_workflow"]["reservation_id"] = "same_suite_different_name"
    prepare_reservations(tmp_path, renamed)
    with pytest.raises(ValueError, match="never try a second"):
        workflow.confirm_challenger(
            tmp_path, renamed, dict(id="next"), incumbent, "development-hash", "registry-hash"
        )
