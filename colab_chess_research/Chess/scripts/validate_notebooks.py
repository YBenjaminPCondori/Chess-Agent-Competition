"""Structural and syntax validation only; does not train or export a model."""

import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "drive.mount" in code[0].source, path
    for index, cell in enumerate(code):
        ast.parse(cell.source, filename=f"{path.name}:cell{index}")
        if not path.name.startswith("05_"):
            assert "chess_rl.export" not in cell.source
    print(path.name, len(code), "code cells: valid")
