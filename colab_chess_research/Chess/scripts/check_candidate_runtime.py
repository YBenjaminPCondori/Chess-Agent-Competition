"""Final-notebook-only fresh-process checks for an extracted candidate."""

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    import torch
    import chess
    import socket

    def deny_network(*args, **kwargs):
        raise RuntimeError("Network use is disabled during final candidate checks")

    socket.socket = deny_network
    sys.path.insert(0, str(args.candidate.resolve()))
    import agent

    initialization_ms = (time.monotonic() - started) * 1000
    assert agent.device == "cpu" and torch.get_num_threads() == 1
    assert bool(agent.MODEL_ERROR) == args.fallback, agent.MODEL_ERROR
    if agent.MODEL is not None:
        assert not agent.MODEL.training
        assert all(p.device.type == "cpu" for p in agent.MODEL.parameters())
    fens = [
        chess.STARTING_FEN,
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        "7k/8/8/8/8/8/p7/7K b - - 0 1",
        "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ]
    elapsed = []
    for fen in fens:
        board = chess.Board(fen)
        assert board.is_valid() and board.legal_moves
        call_started = time.monotonic()
        move = chess.Move.from_uci(agent.get_move(fen, 1))
        elapsed.append((time.monotonic() - call_started) * 1000)
        assert move in board.legal_moves
    board = chess.Board()
    move = chess.Move.from_uci(agent.get_move(board.fen(), 1000))
    assert move in board.legal_moves
    assert max(elapsed) < 250, elapsed
    print(
        json.dumps(
            dict(
                init_ms=initialization_ms,
                low_clock_ms=elapsed,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                network_blocked_after_dependencies=True,
                fallback=args.fallback,
                metrics=agent.AGENT.engine.cumulative,
            )
        )
    )


if __name__ == "__main__":
    main()
