"""Team-trained policy/value network with legal classical fallback."""

import json
from pathlib import Path
import torch
from chess_runtime.runtime_loader import reconstruct
from chess_runtime.search import ResearchAgent

device = "cpu"
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
MODEL_ERROR = None
CONFIG = {}
try:
    CONFIG = json.loads((HERE / "config.json").read_text())
    MODEL = reconstruct(HERE, CONFIG)
except Exception as error:
    MODEL = None
    MODEL_ERROR = f"{type(error).__name__}: {error}"
    print("MODEL_LOAD_FAILURE " + MODEL_ERROR, flush=True)
AGENT = ResearchAgent(
    MODEL, CONFIG.get("search", {}), device, CONFIG.get("model_version", "unavailable")
)


def get_move(fen: str, time_left_ms: int) -> str:
    move = AGENT.get_move(fen, time_left_ms)
    metrics = dict(AGENT.engine.cumulative, model_load_failure=MODEL_ERROR)
    print("CHESS_METRICS " + json.dumps(metrics), flush=True)
    return move
