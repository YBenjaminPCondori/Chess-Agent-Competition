"""Versioned records, leakage-controlled splits, and on-demand tensor encoding."""

import hashlib
import json
import math
import random
from pathlib import Path
import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
from .action_encoding import encode_move, legal_mask, ACTION_SIZE
from .board_encoding import encode_board
from .reproducibility import atomic_json, read_json, sha256
from .transposition import position_key


def identity(fen):
    return repr(position_key(chess.Board(fen)))


def game_split(game_id, seed=42):
    value = int(hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()[:8], 16) % 10
    return "train" if value < 8 else "val" if value == 8 else "test"


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def resolve_source_paths(root, paths, label):
    resolved = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = Path(root) / path
        if not path.is_file():
            raise FileNotFoundError(f"Broad {label} source not found: {path}")
        resolved.append(path)
    return resolved


def prepare_openings(root, seed=42, development=128, heldout=256):
    directory = Path(root) / "datasets/openings"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest["seed"], manifest["development"], manifest["heldout"]) != (
            seed,
            development,
            heldout,
        ):
            raise ValueError(
                "Opening suite already exists with different settings; version it separately"
            )
        for name, digest in manifest["hashes"].items():
            if sha256(directory / name) != digest:
                raise ValueError("Immutable opening suite has changed")
        return manifest
    rng = random.Random(seed)
    seen = set()
    rows = []
    while len(rows) < development + heldout:
        board = chess.Board()
        moves = []
        for _ in range(rng.randint(12, 40)):
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            if not legal or board.outcome() is not None:
                break
            move = rng.choice(legal)
            moves.append(move.uci())
            board.push(move)
        key = identity(board.fen())
        if board.outcome() is not None or not board.is_valid() or key in seen:
            continue
        seen.add(key)
        index = len(rows)
        rows.append(
            dict(
                fen=board.fen(),
                opening_id=f"generated-{seed}-{index}",
                family_id=f"generated-{seed}-{index}",
                source="legal_random_playout",
                source_moves=moves,
                source_start=chess.STARTING_FEN,
            )
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "development.jsonl", rows[:development])
    write_jsonl(directory / "heldout.jsonl", rows[development:])
    manifest = dict(
        seed=seed,
        development=development,
        heldout=heldout,
        quality="Valid legal playouts; not certified balanced or platform openings.",
        hashes={name: sha256(directory / name) for name in ("development.jsonl", "heldout.jsonl")},
    )
    atomic_json(manifest_path, manifest)
    return manifest


def split_records(records, seed=42, forbidden=()):
    forbidden = set(forbidden)
    seen = set()
    splits = {"train": [], "val": [], "test": []}
    dropped = 0
    for record in records:
        key = identity(record["fen"])
        if key in forbidden or key in seen:
            dropped += 1
            continue
        seen.add(key)
        row = dict(record, split=game_split(record["game_id"], seed))
        splits[row["split"]].append(row)
    return splits, dropped


def pgn_positions(paths, burn_in=12):
    for path in paths:
        path = Path(path)
        digest = sha256(path)
        with path.open() as stream:
            game_index = 0
            while (game := chess.pgn.read_game(stream)) is not None:
                if game.errors:
                    raise ValueError(f"PGN parser errors in {path}: {game.errors}")
                board = game.board()
                game_id = f"{digest}:{game_index}"
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= burn_in and board.outcome() is None:
                        yield dict(
                            fen=board.fen(),
                            game_id=game_id,
                            opening_id=game_id,
                            ply=board.ply(),
                            source_uri_or_path=str(path),
                            game_result=game.headers.get("Result")
                            if game.headers.get("Result") != "*"
                            else None,
                        )
                    board.push(move)
                game_index += 1


def make_record(position, label):
    board = chess.Board(position["fen"])
    move = chess.Move.from_uci(label["best_move_uci"]) if label.get("best_move_uci") else None
    row = dict(
        schema_version=1,
        fen=board.fen(),
        legal_moves_uci=[m.uci() for m in board.legal_moves],
        best_move_uci=move.uci() if move else None,
        played_move_uci=None,
        policy_target_index=encode_move(board, move) if move else None,
        policy_target_distribution=None,
        value_target=None,
        value_target_kind=None,
        game_result=None,
        termination_reason=None,
        source_label="unknown",
        source_uri_or_path=None,
        source_version=None,
        game_id=None,
        opening_id=None,
        ply=board.ply(),
        split=None,
        board_encoder_version="board_v1",
        action_encoder_version="action_v1",
        model_version=None,
        teacher_search_depth=None,
        teacher_node_count=None,
        raw_engine_cp=None,
        raw_engine_mate=None,
    )
    row.update(position)
    row.update(label)
    row["fen"] = board.fen()
    return row


def prepare_dataset(root, cfg):
    from .teacher_labelling import Teacher
    from .reservations import enabled, load_reservations, evidence, audit_dataset, excluded

    root = Path(root)
    reservation = load_reservations(root, cfg) if enabled(cfg) else None
    if reservation:
        reservation = dict(
            reservation,
            reserved_positions=set(reservation["reserved_positions"]),
            reserved_groups=set(reservation["reserved_groups"]),
        )
    prepare_openings(root, cfg["seed"])
    manifest_path = root / "datasets/manifests" / (cfg["run_id"] + ".json")
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["dataset_config"] != cfg["dataset"]:
            raise ValueError("Dataset config changed under the same run_id")
        for split, relative in manifest["paths"].items():
            if sha256(root / relative) != manifest["hashes"][split]:
                raise ValueError("Processed dataset hash mismatch")
        if reservation:
            audit_dataset(root, cfg, manifest)
        return manifest
    forbidden = {
        identity(row["fen"])
        for name in ("development", "heldout")
        for row in read_jsonl(root / f"datasets/openings/{name}.jsonl")
    }
    if reservation:
        forbidden.update(reservation["reserved_positions"])
    settings = cfg["dataset"]
    imported = settings.get("jsonl_paths", [])
    pgn_paths = settings.get("pgn_paths", [])
    if not imported and not pgn_paths:
        raise ValueError(
            "Broad training requires external data. Put converted Lichess Eval DB JSONL under "
            "datasets/raw/external/ and list it in configs/default.yaml dataset.jsonl_paths, "
            "or provide PGNs in dataset.pgn_paths for teacher labelling."
        )
    imported = resolve_source_paths(root, imported, "JSONL")
    pgn_paths = resolve_source_paths(root, pgn_paths, "PGN")
    cache = root / "datasets/raw" / cfg["run_id"]
    cache.mkdir(parents=True, exist_ok=True)
    labelled = {
        identity(row["fen"]): row
        for file in sorted(cache.glob("labels-*.jsonl"))
        for row in read_jsonl(file)
    }
    cache_config = cache / "config.json"
    if cache_config.exists() and read_json(cache_config) != settings:
        raise ValueError("Raw cache config differs; use a new run_id")
    atomic_json(cache_config, settings)
    shard_number = len(list(cache.glob("labels-*.jsonl")))
    pending = []
    records = []
    seen = set()
    source = (
        (row for p in imported for row in read_jsonl(p))
        if imported
        else pgn_positions(pgn_paths, settings["burn_in_plies"])
    )
    teacher = None
    try:
        for position in source:
            if reservation and excluded(position, reservation):
                continue
            board = chess.Board(position["fen"])
            if not board.is_valid() or board.outcome() is not None:
                continue
            key = identity(board.fen())
            if key in seen or key in forbidden:
                continue
            seen.add(key)
            if imported:
                row = dict(position)
                if not row.get("game_id"):
                    raise ValueError("Imported records require a source game_id")
                validate_record(row)
            elif key in labelled:
                row = labelled[key]
            else:
                if teacher is None:
                    teacher = Teacher(settings)
                row = make_record(position, teacher.label(board))
                labelled[key] = row
                pending.append(row)
                if len(pending) >= 1000:
                    write_jsonl(cache / f"labels-{shard_number:06d}.jsonl", pending)
                    pending, shard_number = [], shard_number + 1
            records.append(row)
            if len(records) >= settings["target_positions"]:
                break
            if len(records) % 1000 == 0:
                print(f"Prepared {len(records):,} labelled positions", flush=True)
    finally:
        if pending:
            write_jsonl(cache / f"labels-{shard_number:06d}.jsonl", pending)
        if teacher:
            teacher.close()
    splits, dropped = split_records(records, cfg["seed"], forbidden)
    if any(not splits[s] for s in ("train", "val", "test")):
        raise ValueError("Every split needs independent source games; increase corpus size")
    manifest = dict(
        dataset_config=settings,
        seed=cfg["seed"],
        paths={},
        hashes={},
        counts={},
        duplicate_or_reserved_dropped=dropped,
        target_reached=len(records) >= settings["target_positions"],
        encoder_versions=["board_v1", "action_v1"],
    )
    if reservation:
        manifest["reservations"] = evidence(root, cfg)
    for split, rows in splits.items():
        path = root / "datasets/processed" / cfg["run_id"] / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["paths"][split] = str(path.relative_to(root))
        manifest["hashes"][split] = sha256(path)
        manifest["counts"][split] = len(rows)
    atomic_json(manifest_path, manifest)
    return manifest


def validate_record(row):
    board = chess.Board(row["fen"])
    if not board.is_valid():
        raise ValueError("Invalid dataset FEN")
    if (
        row.get("board_encoder_version", "board_v1") != "board_v1"
        or row.get("action_encoder_version", "action_v1") != "action_v1"
    ):
        raise ValueError("Incompatible encoder version")
    if row.get("value_target") is not None and not -1 <= row["value_target"] <= 1:
        raise ValueError("Invalid value target")
    move = row.get("best_move_uci")
    if move and encode_move(board, chess.Move.from_uci(move)) != row.get("policy_target_index"):
        raise ValueError("Move and policy target disagree")
    return board


class PositionDataset(Dataset):
    def __init__(self, records):
        self.records = (
            list(read_jsonl(records)) if isinstance(records, (str, Path)) else list(records)
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        board = validate_record(row)
        mask = legal_mask(board)
        policy = torch.zeros(ACTION_SIZE)
        distribution = row.get("policy_target_distribution")
        target_index = row.get("policy_target_index")
        if distribution:
            for action, probability in distribution:
                if (
                    not isinstance(action, int)
                    or not 0 <= action < ACTION_SIZE
                    or not math.isfinite(probability)
                    or probability < 0
                ):
                    raise ValueError("Invalid policy distribution")
                policy[action] += probability
        elif target_index is not None:
            if not 0 <= target_index < ACTION_SIZE:
                raise ValueError("Invalid target index")
            policy[target_index] = 1
        if bool((policy[~mask] != 0).any()):
            raise ValueError("Probability assigned to illegal action")
        if policy.sum() > 0:
            policy /= policy.sum()
        value = row.get("value_target")
        return dict(
            board=encode_board(board),
            mask=mask,
            policy=policy,
            value=torch.tensor(0.0 if value is None else value),
            policy_valid=torch.tensor(bool(policy.sum() > 0)),
            value_valid=torch.tensor(value is not None),
            source=str(row.get("value_target_kind", "unknown")),
        )
