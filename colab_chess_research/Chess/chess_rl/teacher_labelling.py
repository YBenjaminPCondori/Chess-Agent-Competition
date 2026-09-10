"""Offline data acquisition only. Never included in the runtime bundle."""

import math
from pathlib import Path
import shutil
import chess.engine
from .reproducibility import sha256
from .search import SearchEngine
from .classical_evaluation import evaluate_cp


class Teacher:
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.classical = None
        if config["teacher"] == "stockfish":
            binary = Path(
                config.get("engine_path") or shutil.which("stockfish") or "/usr/games/stockfish"
            )
            if not binary.is_file():
                raise FileNotFoundError(
                    "Stockfish unavailable. Run the Colab apt installation cell, or explicitly select teacher: classical."
                )
            self.engine = chess.engine.SimpleEngine.popen_uci(str(binary), timeout=90)
            self.engine.configure({"Threads": 1, "Hash": config.get("hash_mb", 128)})
            self.source = dict(name=self.engine.id, binary_sha256=sha256(binary))
        elif config["teacher"] == "classical":
            self.classical = SearchEngine(dict(max_depth=3, quiescence_depth=4, max_budget_ms=2000))
            self.source = dict(name="team_classical", search_version="0.1.0")
        else:
            raise ValueError("Unknown teacher")

    def label(self, board):
        if self.engine:
            info = self.engine.analyse(board, chess.engine.Limit(nodes=self.config["nodes"]))
            if not info.get("pv"):
                raise ValueError("Teacher supplied no principal variation")
            score = info["score"].pov(board.turn)
            cp, mate = score.score(), score.mate()
            value = (1.0 if mate > 0 else -1.0) if mate is not None else math.tanh(cp / 600)
            return dict(
                best_move_uci=info["pv"][0].uci(),
                value_target=value,
                value_target_kind="engine_mate" if mate is not None else "engine_cp",
                source_label="stockfish",
                source_version=self.source,
                raw_engine_cp=cp,
                raw_engine_mate=mate,
                teacher_search_depth=info.get("depth"),
                teacher_node_count=info.get("nodes"),
                node_limit_completed=info.get("nodes", 0) >= self.config["nodes"],
            )
        result = self.classical.choose(board, 120000, teacher=True, max_depth=3)
        board.push(result.move)
        try:
            cp = -evaluate_cp(board)
        finally:
            board.pop()
        return dict(
            best_move_uci=result.move.uci(),
            value_target=math.tanh(cp / 600),
            value_target_kind="classical_cp",
            source_label="team_classical",
            source_version=self.source,
            raw_engine_cp=cp,
            raw_engine_mate=None,
            teacher_search_depth=result.depth,
            teacher_node_count=result.nodes,
        )

    def close(self):
        if self.engine:
            self.engine.quit()
