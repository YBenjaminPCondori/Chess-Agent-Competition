"""Generate provenance and extract original team PST constants without editing the source."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "reference/classical_agent/agent.py"
tree = ast.parse(source.read_text())
names = {
    "PIECE_VALUES",
    "PAWN_PST",
    "KNIGHT_PST",
    "BISHOP_PST",
    "ROOK_PST",
    "QUEEN_PST",
    "KING_MIDGAME_PST",
    "PIECE_SQUARE_TABLES",
}
nodes = [
    node
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
]
tables = ast.Module(body=[ast.Import(names=[ast.alias(name="chess")])] + nodes, type_ignores=[])
(ROOT / "chess_rl/piece_tables.py").write_text(
    '"""Piece-square constants extracted unchanged from the team baseline."""\n'
    + ast.unparse(tables)
    + "\n"
)
files = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT / "reference").rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
}
manifest = dict(
    inspected_at="2026-09-10",
    files=files,
    source_urls=[
        "https://aichessathon.com/docs",
        "https://github.com/advitrocks9/aichessathon-starter",
    ],
    runtime_versions=dict(
        python="3.12",
        torch="2.13.0+cpu",
        numpy="2.5.2",
        chess="1.11.2",
        onnxruntime="1.29.0",
        numba="0.67.0",
    ),
    notebook_references=["Part_C_PPO_Environment.ipynb", "Part_C_PPO_Training.ipynb"],
)
result = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True
)
manifest["workspace_commit"] = result.stdout.strip() if result.returncode == 0 else None
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Recorded", len(files), "reference-file hashes and extracted PST constants.")
