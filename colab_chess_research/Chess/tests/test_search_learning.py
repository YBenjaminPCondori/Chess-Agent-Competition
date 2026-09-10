from copy import deepcopy
import math
import random
import chess
import pytest
import torch
from torch.utils.data import default_collate
from chess_rl.action_encoding import encode_move
from chess_rl.checkpoints import save_checkpoint, load_checkpoint
from chess_rl.dataset import make_record, PositionDataset, split_records, identity, prepare_openings
from chess_rl.model import ChessPolicyValueNetSmall
from chess_rl.search import SearchEngine, ResearchAgent
from chess_rl.time_management import SearchTimeout, Deadline
from chess_rl.training import loss_components, OptimizerLoop
from chess_rl.transposition import cache_key, Entry, EXACT, LOWER, UPPER, to_table, from_table
from chess_rl.reproducibility import seed_all, restore_rng


def rows():
    board = chess.Board()
    result = []
    for index, move in enumerate(list(board.legal_moves)[:4]):
        result.append(
            make_record(
                dict(fen=board.fen(), game_id=str(index)),
                dict(best_move_uci=move.uci(), value_target=0.5, value_target_kind="engine_cp"),
            )
        )
    return result


def settings():
    return dict(
        learning_rate=0.0003,
        weight_decay=0.0001,
        batch_size_cpu=4,
        batch_size_cuda=4,
        policy_label_smoothing=0.05,
        huber_delta=1.0,
        gradient_clip_norm=1.0,
        value_weight=1.0,
        scheduler="cosine",
    )


def test_default_architecture_and_variants():
    model = ChessPolicyValueNetSmall()
    assert sum(p.numel() for p in model.parameters()) == 2335370
    model.eval()
    with torch.inference_mode():
        logits, values = model(torch.zeros(2, 21, 8, 8))
    assert logits.shape == (2, 4672) and values.shape == (2,)
    assert bool((values.abs() <= 1).all())
    model = ChessPolicyValueNetSmall(1, 8, 16, "silu", False, 0.1)
    assert model(torch.zeros(2, 21, 8, 8))[0].shape == (2, 4672)


@pytest.mark.parametrize("smoothing", [0, 0.05, 0.1])
def test_legal_loss_finite_and_gradients(smoothing):
    batch = default_collate([PositionDataset(rows())[i] for i in range(4)])
    logits = torch.randn(4, 4672, requires_grad=True)
    values = torch.zeros(4, requires_grad=True)
    policy, value = loss_components(logits, values, batch, smoothing)
    (policy + value).backward()
    assert torch.isfinite(policy + value)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[~batch["mask"]].abs().sum() == 0


def test_terminal_value_only_and_bad_labels():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    row = make_record(
        dict(fen=board.fen(), game_id="mate"), dict(value_target=-1, best_move_uci=None)
    )
    batch = default_collate([PositionDataset([row])[0]])
    policy, value = loss_components(
        torch.zeros(1, 4672, requires_grad=True), torch.zeros(1), batch, 0.1
    )
    assert policy == 0 and torch.isfinite(value)
    bad = rows()[0]
    bad["policy_target_index"] = encode_move(chess.Board(), chess.Move.from_uci("e2e4"))
    with pytest.raises(ValueError):
        PositionDataset([bad])[0]


def test_timeout_restores_board():
    board = chess.Board()
    engine = SearchEngine(dict(max_depth=3))
    before = board.fen()

    def stop(_board):
        raise SearchTimeout()

    engine.leaf = stop
    result = engine.choose(board, 120000)
    assert result.move in board.legal_moves
    assert board.fen() == before and not board.move_stack


def test_teacher_root_scores_and_mate():
    engine = SearchEngine(dict(max_depth=1, quiescence_depth=0, max_budget_ms=5000))
    board = chess.Board()
    result = engine.choose(board, 120000, teacher=True)
    assert result.depth == 1 and len(result.root_scores) == 20
    assert not board.move_stack
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert engine.terminal_score(mate, 3) == -9997
    assert to_table(-9997, 3) == -10000
    assert from_table(-10000, 7) == -9993


def test_tt_rule_context_and_bounds():
    board = chess.Board()
    assert cache_key(board, ["a"], "v1") != cache_key(board, ["b"], "v1")
    original = cache_key(board, [], "v1")
    board.halfmove_clock = 99
    assert original != cache_key(board, [], "v1")
    board = chess.Board()
    engine = SearchEngine()
    engine.deadline = Deadline(math.inf)
    engine.history = ["root"]
    key = cache_key(board, engine.history, engine.version)
    for flag, alpha, beta in ((EXACT, -1, 1), (LOWER, -0.1, 0.1), (UPPER, 0.5, 0.6)):
        score = 0.3
        engine.tt.put(key, Entry(1, score, flag, "e2e4"))
        assert engine.negamax(board, 1, alpha, beta, 0) == score


def test_model_failure_fallback_and_frozen_bn():
    model = ChessPolicyValueNetSmall(1, 8, 16)
    agent = ResearchAgent(model, dict(max_depth=1, quiescence_depth=0))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    board = chess.Board()
    move = agent.get_move(board.fen(), 1000)
    assert chess.Move.from_uci(move) in board.legal_moves
    assert not model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])

    def fail(x):
        raise RuntimeError("deliberate model failure")

    model.forward = fail
    move = agent.get_move(board.fen(), 1000)
    assert chess.Move.from_uci(move) in board.legal_moves
    assert agent.engine.neural.stats["model_errors"] == 1


def test_data_groups_and_duplicates(tmp_path):
    records = rows()
    for index, row in enumerate(records):
        board = chess.Board()
        for move in list(board.legal_moves)[:1]:
            board.push(move)
        row["game_id"] = f"source-{index}"
        row["fen"] = board.fen()
    splits, dropped = split_records(records)
    assert sum(len(items) for items in splits.values()) == 1 and dropped == 3
    splits, _ = split_records(records, forbidden={identity(records[0]["fen"])})
    assert sum(len(items) for items in splits.values()) == 0
    manifest = prepare_openings(tmp_path, development=3, heldout=3)
    assert prepare_openings(tmp_path, development=3, heldout=3) == manifest
    with pytest.raises(ValueError):
        prepare_openings(tmp_path, seed=99, development=3, heldout=3)


def test_checkpoint_rng_optimizer_resume(tmp_path):
    seed_all(42)
    model = ChessPolicyValueNetSmall(1, 8, 16)
    loop = OptimizerLoop(model, settings(), "cpu", constant_lr=True)
    dataset = PositionDataset(rows())
    batch = default_collate([dataset[i] for i in range(len(dataset))])
    loop.train_group([batch])
    path = save_checkpoint(
        tmp_path,
        model,
        loop.optimizer,
        loop.scheduler,
        loop.scaler,
        config={},
        epoch=1,
        update=loop.update,
    )
    payload = load_checkpoint(path)
    next_random = random.random()
    expected = deepcopy(model.state_dict())
    restored_model = ChessPolicyValueNetSmall(1, 8, 16)
    restored_loop = OptimizerLoop(restored_model, settings(), "cpu", constant_lr=True)
    restored_loop.restore(payload)
    assert random.random() == next_random and restored_loop.update == 1
    for key, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    restore_rng(payload["rng"])
    loop.train_group([batch])
    restore_rng(payload["rng"])
    restored_loop.train_group([batch])
    for first, second in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(first, second)
