import random
import time
import chess
import pytest
import torch
from chess_rl.action_encoding import encode_move, decode_move, legal_mask
from chess_rl.board_encoding import encode_board
from chess_rl.environment import terminal, ChessEnvironment, result_value

FENS = [
    chess.STARTING_FEN,
    "7k/P7/8/8/8/8/8/7K w - - 0 1",
    "7k/8/8/8/8/8/p7/7K b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
]


@pytest.mark.parametrize("fen", FENS)
def test_round_trip(fen):
    board = chess.Board(fen)
    assert board.is_valid()
    pairs = [(move, encode_move(board, move)) for move in board.legal_moves]
    assert len({index for _, index in pairs}) == len(pairs)
    assert int(legal_mask(board).sum()) == len(pairs)
    for move, index in pairs:
        assert decode_move(board, index) == move
        plane, square = divmod(index, 64)
        logits = torch.zeros(73, 8, 8)
        logits[plane, square // 8, square % 8] = 1
        assert logits.flatten()[index] == 1


def test_many_positions():
    rng = random.Random(42)
    board = chess.Board()
    for _ in range(200):
        if board.outcome():
            board.reset()
        test_round_trip(board.fen())
        board.push(rng.choice(list(board.legal_moves)))


def test_all_promotions():
    for fen, prefix in ((FENS[1], "a7a8"), (FENS[2], "a2a1")):
        board = chess.Board(fen)
        for suffix in "qrbn":
            move = chess.Move.from_uci(prefix + suffix)
            assert decode_move(board, encode_move(board, move)) == move


def test_invalid_actions():
    board = chess.Board()
    for index in (-1, 4672, True, 12.5):
        with pytest.raises(ValueError):
            decode_move(board, index)
    with pytest.raises(ValueError):
        encode_move(board, chess.Move.from_uci("e2e5"))
    with pytest.raises(ValueError):
        decode_move(board, 0)  # a1 rook cannot move through its pawn.


def test_planes_and_no_hidden_history():
    board = chess.Board()
    planes = encode_board(board)
    assert tuple(planes.shape) == (21, 8, 8)
    assert planes[0, 1, 0] == 1 and planes[6, 6, 0] == 1
    assert planes[12:].sum() == 5 * 64
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    torch.testing.assert_close(encode_board(board), encode_board(chess.Board(board.fen())))
    assert encode_board(board)[20].sum() == 0
    board.halfmove_clock, board.fullmove_number = 150, 500
    assert encode_board(board)[18].min() == 1 and encode_board(board)[19].max() == 1


def test_castling_and_pinned_en_passant():
    board = chess.Board("k4r2/8/8/8/8/8/8/4K2R w K - 0 1")
    assert encode_board(board)[13, 0, 0] == 1
    assert chess.Move.from_uci("e1g1") not in board.legal_moves
    board = chess.Board(FENS[4])
    assert chess.Move.from_uci("e5d6") not in board.legal_moves
    assert encode_board(board)[17].sum() == 0


def test_draw_order_and_claimable_next_move():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    assert board.can_claim_threefold_repetition()
    assert not board.is_repetition(3)
    assert terminal(board) is None
    board.push_uci("f6g8")
    assert terminal(board).reason == "threefold_repetition"
    mate = chess.Board(FENS[5])
    mate.halfmove_clock, mate.fullmove_number = 100, 301
    assert terminal(mate).reason == "checkmate"
    assert terminal(chess.Board(FENS[6])).reason == "stalemate"
    assert terminal(chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")).reason == "insufficient_material"


def test_counters_clock_and_cap():
    env = ChessEnvironment("7k/8/8/8/8/8/8/KR6 w - - 99 300")
    assert env.board.ply() == 598
    assert env.board.halfmove_clock == 99
    env.step("b1b2", 0)
    assert env.finish.reason == "fifty_moves"
    env = ChessEnvironment("7k/8/8/8/8/8/8/KR6 w - - 0 300")
    env.step("b1b2", 0)
    assert env.finish is None
    env.step("h8h7", 0)
    assert env.finish.reason == "ply_cap"
    env = ChessEnvironment(base_ms=10, increment_ms=5)
    env.step("e2e4", 10)
    assert env.clock[chess.WHITE] == 5
    env = ChessEnvironment(base_ms=10)
    assert env.step("e2e4", 10.01).reason == "flag"
    env = ChessEnvironment("7k/8/8/8/8/8/8/KR6 w - - 0 1", base_ms=1)
    assert env.step("b1b2", 2).winner is None


def test_rewards():
    assert result_value("1-0", True) == 1
    assert result_value("1-0", False) == -1
    assert result_value("0-1", False) == 1
    assert result_value("1/2-1/2", False) == 0
    with pytest.raises(ValueError):
        result_value("*", True)


def test_public_api_low_clock():
    from chess_rl.search import ResearchAgent

    agent = ResearchAgent()
    for fen in FENS[:5]:
        board = chess.Board(fen)
        started = time.monotonic()
        move = agent.get_move(fen, 1)
        assert time.monotonic() - started < 0.25
        assert chess.Move.from_uci(move) in board.legal_moves
