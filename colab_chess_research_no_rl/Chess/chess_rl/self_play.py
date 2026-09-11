"""Search policy improvement plus outcome regression; no gradients through alpha-beta."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import sys
import time
import chess
import numpy as np
from torch.utils.data import default_collate
from .checkpoints import load_model, load_checkpoint, save_checkpoint, checkpoint_path
from .dataset import PositionDataset, read_jsonl, identity, make_record
from .environment import ChessEnvironment, FAILURES, result_value
from .search import ResearchAgent
from .reproducibility import atomic_json, read_json, sha256, seed_all, resolve_device, append_csv
from .training import OptimizerLoop
from .league import League
from .evaluation import harness_modules, build_research_agent, run_matchup, promotion_allowed
from .action_encoding import encode_move


def search_distribution(scores, temperature):
    if not scores:
        raise ValueError("A completed root iteration is required")
    moves = sorted(scores)
    values = np.array([scores[m] for m in moves], dtype=np.float64)
    if temperature <= 0 or np.max(values) > 9000:
        probabilities = np.zeros(len(moves))
        probabilities[int(np.argmax(values))] = 1
    else:
        logits = (values - np.max(values)) / temperature
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
    return moves, probabilities


def frozen_model(path, device):
    model, _ = load_model(path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def import_baseline(path, seed):
    spec = importlib.util.spec_from_file_location(f"baseline_{seed}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    if hasattr(module, "RNG"):
        module.RNG.seed(str(seed))
    return module.get_move


def collect_game(root, cfg, league, iteration, game_index, forbidden):
    root = Path(root)
    settings = cfg["self_play"]
    seed = cfg["seed"] + iteration * 100000 + game_index
    rng = random.Random(seed)
    device = resolve_device(cfg["device"])
    (kind, opponent_path, _), mixture = league.opponent(seed)
    learner_white = game_index % 2 == 0
    champion_hash = sha256(league.champion)
    mirror = kind == "neural" and opponent_path == league.state["champion"]
    harness_modules(root)
    from harness.rules import OPENINGS

    opening_name, fen = OPENINGS[rng.randrange(len(OPENINGS))]
    env = ChessEnvironment(fen, cfg["evaluation"]["base_ms"], cfg["evaluation"]["increment_ms"])
    actors = {}
    initialization = {}
    for color in (True, False):
        started = time.monotonic()
        try:
            if color == learner_white or mirror:
                actors[color] = ResearchAgent(
                    frozen_model(league.champion, device), cfg["search"], device, champion_hash
                )
            elif kind == "neural":
                weights = root / opponent_path
                actors[color] = ResearchAgent(
                    frozen_model(weights, device), cfg["search"], device, sha256(weights)
                )
            else:
                actors[color] = import_baseline(root / opponent_path, seed)
            if time.monotonic() - started > 90:
                initialization[color] = "init"
        except Exception as error:
            initialization[color] = f"init: {type(error).__name__}: {error}"
    if initialization:
        if len(initialization) == 2:
            env.fail("init", both=True)
        else:
            env.fail("init", mover=next(iter(initialization)))
    records, moves, episode_ply = [], [], 0
    while env.finish is None:
        color = env.board.turn
        fen, time_left_ms = env.observe()
        actor = actors[color]
        started = time.monotonic()
        row = None
        try:
            if color == learner_white or mirror:
                board = chess.Board(fen)
                result = actor.engine.choose(
                    board,
                    time_left_ms,
                    teacher=True,
                    max_depth=settings["teacher_max_depth"],
                    started_at=started,
                )
                chosen = result.move.uci()
                if result.root_scores:
                    labels, targets = search_distribution(
                        result.root_scores, settings["teacher_target_temperature"]
                    )
                    early = episode_ply < settings["temperature_cutoff_episode_ply"]
                    temperature = (
                        settings["temperature_early"] if early else settings["temperature_late"]
                    )
                    epsilon = (
                        settings["epsilon_exploration_early"]
                        if early
                        else settings["epsilon_exploration_late"]
                    )
                    actions, probabilities = search_distribution(result.root_scores, temperature)
                    probabilities = (1 - epsilon) * probabilities + epsilon / len(probabilities)
                    chosen = rng.choices(actions, weights=probabilities, k=1)[0]
                    distribution = [
                        [encode_move(board, chess.Move.from_uci(move)), float(prob)]
                        for move, prob in zip(labels, targets)
                    ]
                else:
                    distribution = None
                row = make_record(
                    dict(
                        fen=fen,
                        game_id=f"self-{cfg['run_id']}-{iteration}-{game_index}",
                        opening_id=opening_name,
                        ply=board.ply(),
                        split="train",
                    ),
                    dict(
                        best_move_uci=result.move.uci() if result.root_scores else None,
                        played_move_uci=chosen,
                        policy_target_distribution=distribution,
                        source_label="search_self_play",
                        value_target_kind="game_result",
                        model_version=champion_hash,
                        source_version=iteration,
                        teacher_search_depth=result.depth,
                        teacher_node_count=result.nodes,
                    ),
                )
            else:
                chosen = (
                    actor.get_move(fen, time_left_ms)
                    if isinstance(actor, ResearchAgent)
                    else actor(fen, time_left_ms)
                )
            spent = (time.monotonic() - started) * 1000
            if settings["clock_mode"] not in {"competition", "untimed_ablation"}:
                raise ValueError("Unknown collection clock mode")
            env.step(chosen, spent if settings["clock_mode"] == "competition" else 0.0)
            moves.append(dict(uci=chosen, spent_ms=spent, color=color))
            if row and identity(fen) not in forbidden:
                records.append(row)
        except Exception as error:
            env.fail("crash", mover=color)
            initialization["move_error"] = f"{type(error).__name__}: {error}"
        episode_ply += 1
    if env.finish.reason in FAILURES:
        records = []
    else:
        for row in records:
            row["value_target"] = result_value(env.finish.result, chess.Board(row["fen"]).turn)
            row["game_result"] = env.finish.result
            row["termination_reason"] = env.finish.reason
    return dict(
        game_index=game_index,
        seed=seed,
        result=env.finish.result,
        termination=env.finish.reason,
        records=records,
        moves=moves,
        opponent=opponent_path,
        opponent_mixture=mixture,
        device=device,
        clock_mode=settings["clock_mode"],
        errors={str(k): v for k, v in initialization.items()},
    )


def train_iteration(
    root, cfg, champion, replay_records, supervised_records, iteration, replay_manifest
):
    root = Path(root)
    directory = root / "checkpoints/self_play" / cfg["run_id"] / f"iteration-{iteration:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "candidate.json").exists():
        reference = read_json(directory / "candidate.json")
        path = directory / reference["path"]
        if sha256(path) != reference["sha256"]:
            raise ValueError("Candidate checkpoint modified")
        return path
    device = resolve_device(cfg["device"])
    model, _ = load_model(champion, device)
    settings = deepcopy(cfg["training"])
    settings["learning_rate"] = cfg["self_play"]["learning_rate"]
    loop = OptimizerLoop(model, settings, device, constant_lr=True)
    seed_all(cfg["seed"] + iteration, cfg.get("deterministic", False))
    if (directory / "latest.json").exists():
        payload = load_checkpoint(checkpoint_path(directory))
        if payload["replay_manifest"] != replay_manifest or payload["config"] != cfg:
            raise ValueError("Replay/config changed while resuming an iteration")
        loop.restore(payload)
    replay, supervised = PositionDataset(replay_records), PositionDataset(supervised_records)
    if not replay or not supervised:
        raise ValueError("No eligible replay or supervised records; inspect failed-game logs")
    batch_size = settings["batch_size_cuda" if device.startswith("cuda") else "batch_size_cpu"]
    supervised_count = max(1, round(batch_size * cfg["self_play"]["supervised_replay_fraction"]))
    updates = cfg["self_play"]["gradient_updates_per_iteration"]
    path = checkpoint_path(directory) if (directory / "latest.json").exists() else None
    while loop.update < updates:
        groups = []
        for _ in range(settings["gradient_accumulation_steps"]):
            rows = [supervised[random.randrange(len(supervised))] for _ in range(supervised_count)]
            rows += [
                replay[random.randrange(len(replay))] for _ in range(batch_size - supervised_count)
            ]
            groups.append(default_collate(rows))
        metrics = loop.train_group(groups)
        append_csv(
            root / "logs" / cfg["run_id"] / f"self-play-{iteration:03d}.csv",
            dict(update=loop.update, **metrics),
        )
        if (
            loop.update % cfg["self_play"]["checkpoint_every_updates"] == 0
            or loop.update == updates
        ):
            path = save_checkpoint(
                directory,
                model,
                loop.optimizer,
                loop.scheduler,
                loop.scaler,
                config=cfg,
                epoch=0,
                update=loop.update,
                iteration=iteration,
                metrics=metrics,
                microbatch_size=loop.microbatch,
                replay_manifest=replay_manifest,
                league_state=dict(champion=str(champion)),
                dataset_hashes=replay_manifest["supervised_hashes"],
                sampler="python_rng_replacement",
            )
            print(f"Iteration {iteration}: update {loop.update}/{updates}", flush=True)
    atomic_json(directory / "candidate.json", dict(path=path.name, sha256=sha256(path)))
    return path


def run_league(root, cfg, initial_checkpoint, dataset_manifest):
    root = Path(root)
    league = League(root, cfg, initial_checkpoint)
    if league.state["completed_iterations"] >= cfg["self_play"]["iterations"]:
        atomic_json(
            league.path.parent / "complete.json",
            dict(
                status="complete",
                champion=league.state["champion"],
                iterations=league.state["completed_iterations"],
            ),
        )
        return league.champion
    forbidden = {
        identity(row["fen"])
        for split in ("val", "test")
        for row in read_jsonl(root / dataset_manifest["paths"][split])
    }
    forbidden.update(
        identity(row["fen"])
        for suite in ("development", "heldout")
        for row in read_jsonl(root / f"datasets/openings/{suite}.jsonl")
    )
    supervised = list(read_jsonl(root / dataset_manifest["paths"]["train"]))
    for iteration in range(
        league.state["completed_iterations"] + 1, cfg["self_play"]["iterations"] + 1
    ):
        collection = root / "results" / cfg["run_id"] / "self_play" / f"iteration-{iteration:03d}"
        collection.mkdir(parents=True, exist_ok=True)
        for game_index in range(cfg["self_play"]["games_per_iteration"]):
            path = collection / f"game-{game_index:05d}.json"
            if path.exists():
                continue
            game = collect_game(root, cfg, league, iteration, game_index, forbidden)
            atomic_json(path, game)
            print(
                f"Self-play {iteration}: game {game_index + 1}/{cfg['self_play']['games_per_iteration']}, "
                f"{game['result']} ({game['termination']})",
                flush=True,
            )
        replay, files = [], {}
        for path in sorted(
            (root / "results" / cfg["run_id"] / "self_play").glob("iteration-*/game-*.json")
        ):
            game = read_json(path)
            replay.extend(game["records"])
            files[str(path.relative_to(root))] = sha256(path)
            capacity = cfg["self_play"]["replay_capacity_positions"]
            if len(replay) > capacity:
                replay = replay[-capacity:]
        replay_manifest = dict(
            files=files,
            capacity=cfg["self_play"]["replay_capacity_positions"],
            positions=len(replay),
            supervised_hashes=dataset_manifest["hashes"],
        )
        candidate = train_iteration(
            root, cfg, league.champion, replay, supervised, iteration, replay_manifest
        )
        candidate_agent = build_research_agent(root, candidate, cfg, f"iteration-{iteration}")
        champion_agent = build_research_agent(root, league.champion, cfg, "champion")
        summary = run_matchup(
            root,
            candidate_agent,
            champion_agent,
            root / "datasets/openings/development.jsonl",
            cfg,
            f"promotion-{iteration:03d}",
            cfg["evaluation"]["development_games"],
        )
        promoted = promotion_allowed(summary, cfg["evaluation"]["promotion_threshold"])
        league.finish_iteration(iteration, candidate, promoted, summary)
    atomic_json(
        league.path.parent / "complete.json",
        dict(
            status="complete",
            champion=league.state["champion"],
            iterations=league.state["completed_iterations"],
        ),
    )
    return league.champion
