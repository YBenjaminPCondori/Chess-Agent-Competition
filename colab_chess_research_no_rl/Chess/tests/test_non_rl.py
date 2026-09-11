"""Ordinary workflow correctness, not submission export or compliance execution."""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat
import pytest
from chess_rl.dataset import write_jsonl
from chess_rl.non_rl import (
    candidate_spec,
    classical_candidate,
    evaluate_no_rl,
    evaluate_and_freeze_no_rl,
    load_no_rl_config,
    selected_no_rl,
    variant_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_rl_project(tmp_path):
    for name in ("chess_rl", "configs", "templates", "reference"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = load_no_rl_config(tmp_path)
    cfg["non_rl"]["tuning"]["enabled"] = False
    cfg["evaluation"].update(development_games=2, heldout_games=2, bootstrap_samples=10)
    # One remaining ply at the event cap exercises actual process matches without a tournament.
    for name in ("development", "heldout"):
        write_jsonl(
            tmp_path / f"datasets/openings/{name}.jsonl",
            [
                dict(
                    fen="6k1/8/8/8/8/8/6PP/6K1 b - - 0 300",
                    opening_id=f"bounded-{name}",
                    family_id=f"bounded-{name}",
                )
            ],
        )
    return tmp_path, cfg


def test_classical_config_has_no_training_requirement(no_rl_project):
    root, cfg = no_rl_project
    assert "self_play" not in cfg
    assert not (root / "checkpoints").exists()
    spec = candidate_spec(root, cfg)
    assert spec["checkpoint"] is None
    assert spec["config"]["search"]["value_eval_mix"] == 0
    assert spec["config"]["search"]["policy_ordering"] is False
    assert cfg["search"]["value_eval_mix"] == 0


def test_supervised_requires_only_its_own_training(no_rl_project):
    root, cfg = no_rl_project
    supervised = variant_config(cfg, "supervised")
    with pytest.raises(RuntimeError, match="notebook 02"):
        candidate_spec(root, supervised)


def test_supervised_selection_uses_no_league(no_rl_project):
    from chess_rl.checkpoints import save_checkpoint, set_pointer
    from chess_rl.model import ChessPolicyValueNetSmall
    from chess_rl.reproducibility import atomic_json

    root, cfg = no_rl_project
    cfg = variant_config(cfg, "supervised")
    directory = root / "checkpoints/supervised" / cfg["run_id"]
    model = ChessPolicyValueNetSmall(residual_blocks=1, channels=8, value_head_hidden=8)
    checkpoint = save_checkpoint(directory, model)
    set_pointer(directory, "best_validation", checkpoint)
    atomic_json(directory / "complete.json", {"fixture_only": True})
    spec = candidate_spec(root, cfg)
    assert root / spec["checkpoint"] == checkpoint
    assert spec["checkpoint_sha256"]
    assert not (root / "checkpoints/self_play").exists()


def test_classical_research_agent_never_imports_torch(no_rl_project):
    root, cfg = no_rl_project
    candidate = classical_candidate(root, cfg)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,chess; import agent; "
            "b=chess.Board(); m=chess.Move.from_uci(agent.get_move(b.fen(),1)); "
            "assert m in b.legal_moves; assert 'torch' not in sys.modules; "
            "assert agent.ENGINE.neural is None",
        ],
        cwd=candidate,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_classical_evaluate_freeze_and_resume_without_checkpoints(no_rl_project, monkeypatch):
    root, cfg = no_rl_project
    import chess_rl.evaluation as evaluation
    import chess_rl.non_rl as workflow

    preserved_harness = evaluation.harness_modules(ROOT)
    monkeypatch.setattr(evaluation, "harness_modules", lambda _: preserved_harness)
    monkeypatch.setattr(
        workflow,
        "_opponents",
        lambda path: {"greedy": Path(path) / "reference/starter/baselines/greedy"},
    )
    result = evaluate_no_rl(root, cfg)
    assert result["summaries"]["greedy"]["games"] == 2
    selection = evaluate_and_freeze_no_rl(root, cfg)
    assert selection["training_required"] is False
    assert selection["training_status"] == "not_applicable"
    assert selected_no_rl(root, cfg["run_id"]) == selection
    assert evaluate_and_freeze_no_rl(root, cfg) == selection
    assert not (root / "checkpoints").exists()
    changed = variant_config(cfg, "classical")
    changed["search"]["max_depth"] += 1
    with pytest.raises(ValueError, match="changed since development"):
        evaluate_and_freeze_no_rl(root, changed)
    source = root / "chess_rl/classical_evaluation.py"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Source changed"):
        selected_no_rl(root, cfg["run_id"])


def test_no_rl_notebooks_have_independent_stages():
    paths = sorted((ROOT / "notebooks/no_rl").glob("*.ipynb"))
    assert len(paths) == 5
    for path in paths:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        code = [cell.source for cell in book.cells if cell.cell_type == "code"]
        assert "drive.mount" in code[0]
        for cell in code:
            ast.parse(cell)
            assert "chess_rl.self_play" not in cell and "run_league" not in cell
            if not path.name.startswith("05_"):
                assert "non_rl_export" not in cell and "final_no_rl_checks" not in cell
