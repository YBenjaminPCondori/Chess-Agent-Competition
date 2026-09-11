"""Run only from the final no-RL notebook, after evaluation and selection."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    import chess
    import socket

    def deny_network(*args, **kwargs):
        raise RuntimeError("Network disabled in local final check")

    socket.socket = deny_network
    sys.path.insert(0, str(args.candidate.resolve()))
    import agent

    init_ms = (time.monotonic() - started) * 1000
    assert "torch" not in sys.modules and agent.ENGINE.neural is None
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
        start = time.monotonic()
        move = chess.Move.from_uci(agent.get_move(fen, 1))
        elapsed.append((time.monotonic() - start) * 1000)
        assert move in board.legal_moves
    assert max(elapsed) < 250
    board = chess.Board()
    assert chess.Move.from_uci(agent.get_move(board.fen(), 1000)) in board.legal_moves
    assert agent.ENGINE.cumulative["search_errors"] == 0
    print(
        json.dumps(
            dict(
                init_ms=init_ms,
                low_clock_ms=elapsed,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                imports_torch=False,
                network_blocked_after_dependencies=True,
                metrics=agent.ENGINE.cumulative,
            )
        )
    )


if __name__ == "__main__":
    main()
