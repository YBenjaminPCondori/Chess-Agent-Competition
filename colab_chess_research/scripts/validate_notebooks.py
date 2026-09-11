"""Structural and syntax validation only; does not train or export a model."""

import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "01_environment_and_encoding.ipynb",
    "02_supervised_broad_training.ipynb",
    "03a_build_strategy_datasets.ipynb",
    "03b_strategy_finetuning.ipynb",
    "04a_self_play_generation.ipynb",
    "04b_self_play_training_and_promotion.ipynb",
    "05a_evaluate_general_strength.ipynb",
    "05b_evaluate_strategy_suites.ipynb",
    "06a_select_and_freeze_winner.ipynb",
    "06b_export_submission.ipynb",
}
assert {p.name for p in (ROOT / "notebooks").glob("*.ipynb")} == EXPECTED
for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
    if "archive" in path.relative_to(ROOT / "notebooks").parts:
        continue
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if path.parent.name != "no_rl":
        stage = notebook.metadata.research_workflow.stage
        dependencies = notebook.metadata.research_workflow.depends_on
        if stage in ("02", "03a"):
            assert dependencies == ["01"]
        if stage == "05b":
            assert "05a" not in dependencies
            assert all("general_evaluation.json" not in cell.source for cell in code)
        assert 'load_config(PROJECT_ROOT, "strategy.yaml")' in "\n".join(c.source for c in code)
    assert "drive.mount" in code[0].source, path
    for index, cell in enumerate(code):
        ast.parse(cell.source, filename=f"{path.name}:cell{index}")
        final = path.name in {"06b_export_submission.ipynb", "05_export_no_rl.ipynb"}
        if not final:
            assert "chess_rl.export" not in cell.source
            assert "non_rl_export" not in cell.source
            assert "final_checks" not in cell.source
            assert "check_candidate_runtime" not in cell.source
            assert "check_classical_runtime" not in cell.source
        if path.parent.name == "no_rl":
            assert "chess_rl.self_play" not in cell.source
    print(path.name, len(code), "code cells: valid")
