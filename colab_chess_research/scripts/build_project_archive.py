"""Portable source project archive; this is NOT the competition submission ZIP."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT.parent / "colab_chess_research.zip"
skip_parts = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", ".git"}
skip_roots = {"checkpoints", "exports", "submission_candidate", "logs", "results"}
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if skip_parts.intersection(relative.parts) or relative.parts[0] in skip_roots:
            continue
        if path.is_file() and path.suffix not in {".tmp", ".pt", ".onnx", ".pyc"}:
            archive.write(path, "Chess/" + relative.as_posix())
print(destination)
print(destination.stat().st_size, "bytes")
