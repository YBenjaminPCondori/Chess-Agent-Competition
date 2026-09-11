"""User-owned PGN/FEN/CSV/JSONL inputs, optional UCI labels, and grouped splits."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import chess
import chess.engine
import chess.pgn
from .action_encoding import encode_move
from .dataset import identity, make_record, read_jsonl, write_jsonl
from .reproducibility import atomic_json, read_json, sha256
from .strategy_features import extract_features
from .strategy_taxonomy import validate_theme

SCHEMA = (
    "fen",
    "legal_moves",
    "best_move",
    "policy_target",
    "value_target",
    "theme",
    "subtheme",
    "source",
    "source_id",
    "ply",
    "result",
    "engine",
    "engine_depth",
    "engine_score_cp",
    "engine_score_mate",
    "split",
    "created_at",
    "notes",
)
EXTRA_FIELDS = (
    "schema_version",
    "board_encoder_version",
    "action_encoder_version",
    "source_sha256",
    "source_game_id",
    "source_line_id",
    "label_provenance",
    "acceptable_moves",
    "history_dependent",
    "played_move",
    "line",
    "history_start_fen",
    "history_moves",
    "features",
    "label_kind",
)
JSON_FIELDS = (
    "legal_moves",
    "policy_target",
    "line",
    "history_moves",
    "features",
    "acceptable_moves",
    "label_provenance",
    "history_dependent",
)


def normalize_row(raw):
    row = {key: raw.get(key) for key in SCHEMA + EXTRA_FIELDS}
    for key, value in row.items():
        if value == "":
            row[key] = None
    for key in JSON_FIELDS:
        if isinstance(row[key], str):
            row[key] = json.loads(row[key])
    for key in ("value_target", "engine_score_cp"):
        if row[key] is not None:
            row[key] = float(row[key])
    for key in ("ply", "engine_depth", "engine_score_mate"):
        if row[key] is not None:
            number = float(row[key])
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError(f"{key} must be an integer")
            row[key] = int(number)
    board = chess.Board(row["fen"])
    if board.chess960 or not board.is_valid():
        raise ValueError("Invalid standard-chess FEN")
    row["fen"] = board.fen()
    row["theme"], row["subtheme"] = validate_theme(row["theme"], row["subtheme"])
    legal = sorted(m.uci() for m in board.legal_moves)
    if row["legal_moves"] is not None and sorted(row["legal_moves"]) != legal:
        raise ValueError("legal_moves does not match the FEN")
    row["legal_moves"] = legal
    for key, expected in (
        ("schema_version", 2),
        ("board_encoder_version", "board_v1"),
        ("action_encoder_version", "action_v1"),
    ):
        if row[key] is not None and str(row[key]) != str(expected):
            raise ValueError(f"Unsupported {key}")
        row[key] = expected
    acceptable = row["acceptable_moves"]
    if acceptable is not None:
        if (
            not isinstance(acceptable, list)
            or not acceptable
            or any(m not in legal for m in acceptable)
        ):
            raise ValueError("acceptable_moves must be a nonempty set of legal UCI moves")
        row["acceptable_moves"] = sorted(set(acceptable))
        if row["best_move"] and row["best_move"] not in acceptable:
            raise ValueError("best_move is outside acceptable_moves")
    if row["history_dependent"] not in (None, True, False):
        raise ValueError("history_dependent must be a boolean")
    for key in ("best_move", "played_move"):
        if row[key] is not None and row[key] not in legal:
            raise ValueError(f"Illegal {key}: {row[key]}")
    policy = row["policy_target"]
    if policy is not None:
        if not isinstance(policy, dict) or not policy:
            raise ValueError("policy_target must be a nonempty UCI-to-probability object")
        if any(
            move not in legal or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
            for move, p in policy.items()
        ):
            raise ValueError("Invalid or illegal policy_target")
        if not math.isclose(sum(policy.values()), 1.0, abs_tol=1e-6):
            raise ValueError("policy_target probabilities must sum to one")
        if row["best_move"] and policy.get(row["best_move"], 0) != max(policy.values()):
            raise ValueError("best_move disagrees with policy_target")
    elif row["best_move"]:
        row["policy_target"] = {row["best_move"]: 1.0}
    value = row["value_target"]
    if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
        raise ValueError("value_target must be finite and in [-1, 1], side-to-move perspective")
    if row["engine_score_cp"] is not None and not math.isfinite(row["engine_score_cp"]):
        raise ValueError("Nonfinite engine score")
    if row["engine_score_cp"] is not None and row["engine_score_mate"] is not None:
        raise ValueError("An engine score is either cp or mate")
    if row["engine_depth"] is not None and row["engine_depth"] < 0:
        raise ValueError("Negative engine depth")
    if row["split"] == "val":
        row["split"] = "validation"
    if row["split"] not in (None, "train", "validation", "test"):
        raise ValueError("Unknown split")
    if row["theme"] == "annotated_benchmark_games" and row["split"] == "train":
        raise ValueError("Annotated benchmark games are evaluation-only")
    if row["result"] not in (None, "*", "1-0", "0-1", "1/2-1/2"):
        raise ValueError("Unknown result")
    if not row["source"] or not row["source_id"]:
        raise ValueError("source and source_id are required for provenance and grouped splits")
    row["ply"] = board.ply() if row["ply"] is None else row["ply"]
    if row["ply"] < 0:
        raise ValueError("Negative ply")
    row["created_at"] = row["created_at"] or datetime.now(timezone.utc).isoformat()
    datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
    row["notes"] = row["notes"] or ""
    row["label_kind"] = row["label_kind"] or (
        "user_supplied" if row["policy_target"] or value is not None else "unlabelled"
    )
    row["label_provenance"] = row["label_provenance"] or {"kind": row["label_kind"]}
    replay = board.copy()
    for token in row["line"] or []:
        move = chess.Move.from_uci(token)
        if move not in replay.legal_moves:
            raise ValueError(f"Illegal continuation move: {token}")
        replay.push(move)
    if row["line"] and row["best_move"] and row["line"][0] != row["best_move"]:
        raise ValueError("Labelled continuation and best_move disagree")
    board_with_history(row)
    return row


def validate_schema(row):
    missing = set(SCHEMA) - row.keys()
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    return normalize_row(row)


def board_with_history(row):
    if row.get("history_moves") is None:
        return chess.Board(row["fen"])
    if not row.get("history_start_fen"):
        raise ValueError("history_moves requires history_start_fen")
    board = chess.Board(row["history_start_fen"])
    if not board.is_valid():
        raise ValueError("Invalid history start")
    for token in row["history_moves"]:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("Illegal history move")
        board.push(move)
    if board.fen() != chess.Board(row["fen"]).fen():
        raise ValueError("History does not reproduce FEN")
    return board


def load_source(root, spec):
    path = Path(root) / spec["path"]
    digest = sha256(path)
    defaults = {k: spec[k] for k in ("theme", "subtheme", "split") if k in spec}
    defaults["source"] = spec.get("source", str(spec["path"]))
    defaults["source_sha256"] = digest
    suffix = path.suffix.lower()
    if suffix == ".pgn":
        with path.open(encoding="utf-8-sig") as stream:
            game_index = 0
            while (game := chess.pgn.read_game(stream)) is not None:
                if game.errors or game.headers.get("Variant", "Standard") not in (
                    "Standard",
                    "Chess",
                ):
                    raise ValueError(f"Invalid standard PGN in {path}")
                board = game.board()
                moves = list(game.mainline_moves())
                history = []
                for move in moves:
                    raw = dict(
                        defaults,
                        fen=board.fen(),
                        source_id=f"{digest}:{game_index}",
                        source_game_id=f"{digest}:{game_index}",
                        played_move=move.uci(),
                        result=game.headers.get("Result"),
                        history_start_fen=game.board().fen(),
                        history_moves=list(history),
                    )
                    yield normalize_row(raw)
                    if move not in board.legal_moves:
                        raise ValueError("Illegal PGN mainline")
                    history.append(move.uci())
                    board.push(move)
                yield normalize_row(
                    dict(
                        defaults,
                        fen=board.fen(),
                        source_id=f"{digest}:{game_index}",
                        source_game_id=f"{digest}:{game_index}",
                        result=game.headers.get("Result"),
                        history_start_fen=game.board().fen(),
                        history_moves=history,
                    )
                )
                game_index += 1
        return
    if suffix in (".fen", ".txt"):
        with path.open(encoding="utf-8-sig") as stream:
            raw_rows = [
                dict(fen=line.strip())
                for line in stream
                if line.strip() and not line.startswith("#")
            ]
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            raw_rows = list(csv.DictReader(stream))
    elif suffix == ".jsonl":
        raw_rows = list(read_jsonl(path))
    else:
        raise ValueError("Strategy sources must be .pgn, .fen, .txt, .csv, or .jsonl")
    for index, raw in enumerate(raw_rows):
        merged = dict(defaults, source_id=f"{digest}:row-{index}")
        merged.update({k: v for k, v in raw.items() if v is not None and v != ""})
        merged.update(source_sha256=digest, source_line_id=f"{digest}:row-{index}")
        yield normalize_row(merged)


def split_rows(rows, seed=42, quarantine=True):
    """Split whole source games, then discard positions shared across partitions."""
    rows = [normalize_row(row) for row in rows]
    forced = {}
    for row in rows:
        group = str(row["source_game_id"] or row["source_id"])
        requested = row["split"] or (
            "test" if row["theme"] == "annotated_benchmark_games" else None
        )
        if requested and group in forced and forced[group] != requested:
            raise ValueError("One source game requests conflicting splits")
        if requested:
            forced[group] = requested
    result = {"train": [], "validation": [], "test": []}
    partitions = {}
    for row in rows:
        group = str(row["source_game_id"] or row["source_id"])
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 10
        split = forced.get(
            group, "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        )
        row["split"] = split
        partitions.setdefault(identity(row["fen"]), set()).add(split)
    seen = set()
    for row in rows:
        if quarantine and len(partitions[identity(row["fen"])]) > 1:
            continue
        key = json.dumps({k: v for k, v in row.items() if k != "created_at"}, sort_keys=True)
        if key not in seen:
            result[row["split"]].append(row)
            seen.add(key)
    return result


def to_training_record(row):
    row = validate_schema(row)
    board = chess.Board(row["fen"])
    return make_record(
        dict(
            fen=row["fen"],
            game_id=row["source_game_id"] or row["source_id"],
            split=row["split"],
            source_version=row["source_sha256"],
            source_uri_or_path=row["source"],
            source_line_id=row["source_line_id"],
            label_provenance=row["label_provenance"],
        ),
        dict(
            best_move_uci=row["best_move"],
            policy_target_distribution=[
                (encode_move(board, chess.Move.from_uci(m)), p)
                for m, p in (row["policy_target"] or {}).items()
            ]
            or None,
            value_target=row["value_target"],
            value_target_kind=row["label_kind"],
            source_label=row["source"],
        ),
    )


def label_with_engine(row, engine, settings):
    board = board_with_history(row)
    outcome = board.outcome()
    if outcome is not None:
        value = 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
        return normalize_row(
            dict(
                row,
                value_target=value,
                label_kind="rule_outcome",
                label_provenance={"kind": "rule_outcome"},
            )
        )
    info = engine.analyse(
        board, chess.engine.Limit(nodes=settings.get("nodes", 50000), depth=settings.get("depth"))
    )
    if not info.get("pv") or "score" not in info:
        raise ValueError("Engine returned no principal variation or score")
    score = info["score"].pov(board.turn)
    cp, mate = score.score(), score.mate()
    return normalize_row(
        dict(
            row,
            best_move=info["pv"][0].uci(),
            policy_target=None,
            line=[m.uci() for m in info["pv"]],
            value_target=math.tanh(cp / 600) if cp is not None else (1.0 if mate > 0 else -1.0),
            engine=engine.id.get("name", "uci"),
            engine_depth=info.get("depth"),
            engine_score_cp=cp,
            engine_score_mate=mate,
            label_kind="engine",
            label_provenance=dict(kind="engine", settings=settings, engine=engine.id),
        )
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA + EXTRA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(row.get(k)) if k in JSON_FIELDS else row.get(k)
                    for k in SCHEMA + EXTRA_FIELDS
                }
            )
    temporary.replace(path)


def load_strategy_manifest(root, path):
    root = Path(root)
    manifest = read_json(root / path)
    if manifest.get("schema_version") not in (1, 2) or set(manifest.get("paths", {})) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Unsupported strategy manifest schema or partitions")
    for split, relative in manifest["paths"].items():
        if sha256(root / relative) != manifest["hashes"][split]:
            raise ValueError("Strategy dataset changed after preparation")
    if (
        "quarantine" in manifest
        and sha256(root / manifest["quarantine"]["path"]) != manifest["quarantine"]["sha256"]
    ):
        raise ValueError("Strategy quarantine changed after preparation")
    return manifest


def build_strategy_datasets(root, cfg):
    from .reservations import enabled, load_reservations, evidence, safe_id, assert_independent

    root = Path(root)
    settings = cfg["strategy_dataset"]
    if not settings["sources"]:
        raise ValueError("Configure strategy_dataset.sources with your own PGN/FEN/CSV/JSONL files")
    destination = root / "data/strategy" / safe_id(settings.get("dataset_id", cfg["run_id"]))
    reservation = load_reservations(root, cfg) if enabled(cfg) else None
    manifest_path = destination / "manifest.json"
    inputs = {s["path"]: sha256(root / s["path"]) for s in settings["sources"]}
    engine_settings = settings.get("engine", {})
    if engine_settings.get("enabled"):
        inputs[engine_settings["path"]] = sha256(root / engine_settings["path"])
    signature = dict(settings=settings, seed=cfg["seed"], inputs=inputs)
    if reservation:
        signature["reservations"] = evidence(root, cfg)
    if manifest_path.exists():
        manifest = load_strategy_manifest(root, manifest_path)
        if manifest["signature"] != signature:
            raise ValueError("Strategy inputs changed; choose a new dataset_id")
        if reservation:
            assert_independent(read_jsonl(root / manifest["paths"]["train"]), reservation)
        return manifest
    rows = [row for spec in settings["sources"] for row in load_source(root, spec)]
    if not rows:
        raise ValueError("Strategy sources contain no positions")
    reserved = set()
    opening_paths = (
        [root / p for p in reservation["suites"].values()]
        if reservation
        else (root / "datasets/openings").glob("*.jsonl")
    )
    for path in opening_paths:
        reserved.update(identity(r["fen"]) for r in read_jsonl(path))
    broad_path = root / "datasets/manifests" / (cfg["run_id"] + ".json")
    if not reservation and broad_path.exists():
        broad = read_json(broad_path)
        # Broad evaluation/training overlaps cannot provide an independent strategy suite.
        for split in ("train", "val", "test"):
            reserved.update(identity(r["fen"]) for r in read_jsonl(root / broad["paths"][split]))
    input_count = len(rows)
    rows = [r for r in rows if identity(r["fen"]) not in reserved]
    reserved_dropped = input_count - len(rows)
    if not rows:
        raise ValueError(
            "All strategy positions overlap reserved broad/opening evaluation positions"
        )
    engine = None
    try:
        if engine_settings.get("enabled"):
            engine = chess.engine.SimpleEngine.popen_uci(str(root / engine_settings["path"]))
            engine.configure({"Threads": 1, "Hash": engine_settings.get("hash_mb", 128)})
        for index, row in enumerate(rows):
            if engine and (
                engine_settings.get("overwrite_labels", False)
                or not (row["policy_target"] or row["value_target"] is not None)
            ):
                row = label_with_engine(row, engine, engine_settings)
            row["features"] = extract_features(
                board_with_history(row), row["history_moves"] is not None
            )
            rows[index] = row
    finally:
        if engine:
            engine.quit()
    splits = split_rows(rows, cfg["seed"])
    unfiltered = split_rows(rows, cfg["seed"], quarantine=False)
    membership = {}
    for split, records in unfiltered.items():
        for row in records:
            membership.setdefault(identity(row["fen"]), set()).add(split)
    quarantined = [
        dict(row, quarantine_reason="cross_split_position")
        for records in unfiltered.values()
        for row in records
        if len(membership[identity(row["fen"])]) > 1
    ]
    quarantine_path = destination / "quarantine.jsonl"
    write_jsonl(quarantine_path, quarantined)
    if reservation:
        for split, records in splits.items():
            for row in records:
                if (
                    reservation["strategy_groups"].get(row["source_game_id"] or row["source_id"])
                    != split
                ):
                    raise ValueError("Strategy partition differs from notebook 01 reservation")
        assert_independent(splits["train"], reservation)
    manifest = dict(
        schema_version=2,
        dataset_id=settings.get("dataset_id", cfg["run_id"]),
        encoder_versions=["board_v1", "action_v1"],
        signature=signature,
        paths={},
        hashes={},
        counts={},
        themes={},
        label_coverage={},
        quarantine=dict(
            path=str(quarantine_path.relative_to(root)),
            sha256=sha256(quarantine_path),
            rows=len(quarantined),
        ),
        reserved_positions_dropped=reserved_dropped,
        duplicate_or_cross_split_rows_dropped=len(rows) - sum(map(len, splits.values())),
    )
    for split, records in splits.items():
        path = destination / f"{split}.jsonl"
        write_jsonl(path, records)
        write_csv(destination / f"{split}.csv", records)
        manifest["paths"][split] = str(path.relative_to(root))
        manifest["hashes"][split] = sha256(path)
        manifest["counts"][split] = len(records)
        manifest["label_coverage"][split] = dict(
            policy=sum(bool(r["policy_target"]) for r in records),
            value=sum(r["value_target"] is not None for r in records),
            unlabelled=sum(not r["policy_target"] and r["value_target"] is None for r in records),
        )
        for row in records:
            counts = manifest["themes"].setdefault(row["theme"], {s: 0 for s in splits})
            counts[split] += 1
    atomic_json(manifest_path, manifest)
    return manifest
