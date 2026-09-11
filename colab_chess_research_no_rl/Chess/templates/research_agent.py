"""Generated research adapter; the final shipping wrapper is produced in notebook 05."""

import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
SETTINGS = json.loads((HERE / "research.json").read_text())
sys.path.insert(0, SETTINGS["project_root"])
from chess_rl.checkpoints import load_model  # noqa: E402
from chess_rl.search import ResearchAgent  # noqa: E402

MODEL = load_model(SETTINGS["checkpoint"], "cpu")[0] if SETTINGS["checkpoint"] else None
AGENT = ResearchAgent(MODEL, SETTINGS["search"], "cpu", SETTINGS["version"])


def get_move(fen: str, time_left_ms: int) -> str:
    move = AGENT.get_move(fen, time_left_ms)
    print("CHESS_METRICS " + json.dumps(AGENT.engine.cumulative), flush=True)
    return move
