"""Portable updated project, including both independent notebook sequences."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent / "colab_chess_research_no_rl.zip"
SOURCE_DIRS = {
    "chess_rl",
    "configs",
    "templates",
    "reference",
    "notebooks",
    "tests",
    "scripts",
    "docs",
}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".ipynb_checkpoints"}


if __name__ == "__main__":
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".md", ".txt", ".toml"}]
    for name in sorted(SOURCE_DIRS):
        files.extend((ROOT / name).rglob("*"))
    # Opening fixtures are versioned inputs; trained models and run outputs are not source.
    files.extend((ROOT / "datasets/openings").glob("*.json*"))
    with zipfile.ZipFile(DESTINATION, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            relative = path.relative_to(ROOT)
            if not path.is_file() or SKIP_PARTS.intersection(relative.parts):
                continue
            if path.suffix in {".pyc", ".pt", ".onnx", ".tmp"}:
                continue
            archive.write(path, "Chess/" + relative.as_posix())
    print(DESTINATION)
    print(DESTINATION.stat().st_size, "bytes")
