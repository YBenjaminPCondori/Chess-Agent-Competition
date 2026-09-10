"""Classical search only: standard library and python-chess; no learned weights."""

import json
from pathlib import Path
import time
import chess
from chess_runtime.search import SearchEngine

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ENGINE = SearchEngine(CONFIG["search"], version=CONFIG["model_version"])


def get_move(fen: str, time_left_ms: int) -> str:
    started = time.monotonic()
    result = ENGINE.choose(chess.Board(fen), time_left_ms, started_at=started)
    print("CHESS_METRICS " + json.dumps(ENGINE.cumulative), flush=True)
    return result.move.uci()
