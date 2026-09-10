import argparse
import ast
import zipfile
from collections.abc import Iterator
from pathlib import Path

from harness.rules import MAX_UNZIPPED_BYTES

DEFAULT_INCLUDES = ("weights",)
SKIP = {"__pycache__", ".DS_Store"}


def members(root: Path, includes: tuple[str, ...]) -> Iterator[tuple[Path, str]]:
    modules = sorted(root.glob("*.py"))
    for path in modules:
        yield path, path.name
    named = {path.name for path in modules}
    for name in sorted(_imported(root, modules) | set(includes)):
        if name in named:
            continue
        source = root / name
        if source.is_file():
            yield source, name
        elif source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and not SKIP & set(path.parts):
                    yield path, path.relative_to(root).as_posix()


def _imported(root: Path, modules: list[Path]) -> set[str]:
    found: set[str] = set()
    pending = list(modules)
    while pending:
        for name in _names(pending.pop()) - found:
            package = root / name
            if package.is_dir():
                found.add(name)
                pending.extend(sorted(package.rglob("*.py")))
    return found


def _names(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            found.add(node.module.split(".")[0])
    return found


def build(root: Path, destination: Path, includes: tuple[str, ...]) -> list[str]:
    entries = list(members(root, includes))
    written = [name for _, name in entries]
    if "agent.py" not in written:
        raise SystemExit(f"No agent.py in {root}. The platform imports it by name")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in entries:
            archive.write(source, name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a submission zip.")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    parser.add_argument("--include", action="append", default=[])
    arguments = parser.parse_args()

    root = Path.cwd()
    written = build(root, arguments.out, DEFAULT_INCLUDES + tuple(arguments.include))
    unzipped = sum((root / name).stat().st_size for name in written)
    print(f"{arguments.out} ({arguments.out.stat().st_size:,} bytes, {unzipped:,} unzipped)")
    for name in written:
        print(f"  {name}")

    if unzipped > MAX_UNZIPPED_BYTES:
        raise SystemExit(
            f"Over the {MAX_UNZIPPED_BYTES // 1_000_000} MB limit at {unzipped:,} bytes unzipped"
        )
    print("\nPackage built. Run harness.play or harness.arena for game testing before upload.")


if __name__ == "__main__":
    main()
