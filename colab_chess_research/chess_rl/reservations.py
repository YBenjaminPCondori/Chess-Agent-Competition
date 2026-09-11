"""Immutable, pre-training evaluation reservations shared by research stages."""

from pathlib import Path
import json
import re
from .dataset import identity, prepare_openings, read_jsonl, write_jsonl
from .reproducibility import atomic_json, read_json, sha256


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Use a filename-safe version identifier")
    return value


def claim_once(path, record):
    """Exclusive creation prevents simultaneous runs claiming one confirmation suite."""
    path = Path(path)
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing = read_json(path)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Confirmation claim is incomplete or being written; no games started"
            ) from error
        if existing != record:
            raise ValueError("Confirmation suite already assigned; never try a second challenger")


def reservation_path(root, cfg):
    return (
        Path(root)
        / "data/reservations"
        / safe_id(cfg["research_workflow"]["reservation_id"])
        / "manifest.json"
    )


def enabled(cfg):
    return bool(cfg.get("research_workflow", {}).get("reservation_id"))


def _inputs(root, cfg):
    specs = cfg["strategy_dataset"]["sources"]
    if not specs and cfg["research_workflow"].get("stages", {}).get("strategy") != "skipped":
        raise ValueError(
            "Configure real strategy sources before notebook 01, or explicitly skip strategy"
        )
    return dict(
        seed=cfg["seed"],
        sources=specs,
        source_hashes={s["path"]: sha256(Path(root) / s["path"]) for s in specs},
        dataset_id=safe_id(cfg["strategy_dataset"]["dataset_id"]),
        games={
            name: cfg["evaluation"][name + "_games"]
            for name in ("development", "confirmation", "heldout")
        },
    )


def prepare_reservations(root, cfg):
    from .strategy_dataset import load_source, split_rows

    root = Path(root)
    path = reservation_path(root, cfg)
    if path.exists():
        return load_reservations(root, cfg)
    inputs = _inputs(root, cfg)
    if any(n <= 0 or n % 2 for n in inputs["games"].values()):
        raise ValueError("Each general suite requires a positive even number of games")
    rows = [r for spec in inputs["sources"] for r in load_source(root, spec)]
    if inputs["sources"] and not rows:
        raise ValueError("Strategy sources are empty")
    partitions = split_rows(rows, cfg["seed"], quarantine=False)
    groups = {
        r["source_game_id"] or r["source_id"]: split for split, rs in partitions.items() for r in rs
    }
    strategy_positions = {identity(r["fen"]) for r in rows}
    reserved_groups = {g for g, split in groups.items() if split != "train"}
    reserved_positions = {
        identity(r["fen"])
        for r in rows
        if (r["source_game_id"] or r["source_id"]) in reserved_groups
    }
    # Versioned legal opening fixtures are independent of the no-RL opening store.
    count = sum(inputs["games"].values()) // 2
    openings, seen, batch = [], set(strategy_positions), 0
    while len(openings) < count:
        scratch = path.parent / "opening_source" / f"batch-{batch:03d}"
        prepare_openings(scratch, cfg["seed"] + batch, development=count, heldout=0)
        batch += 1
        for row in read_jsonl(scratch / "datasets/openings/development.jsonl"):
            key = identity(row["fen"])
            if key not in seen:
                openings.append(row)
                seen.add(key)
            if len(openings) == count:
                break
    suites, files, offset = {}, {}, 0
    for name, games in inputs["games"].items():
        selected = openings[offset : offset + games // 2]
        offset += games // 2
        destination = path.parent / (name + ".jsonl")
        write_jsonl(destination, selected)
        relative = str(destination.relative_to(root))
        suites[name] = relative
        files[relative] = sha256(destination)
        reserved_positions.update(identity(r["fen"]) for r in selected)
        reserved_groups.update(r["family_id"] for r in selected)
    manifest = dict(
        schema_version=1,
        inputs=inputs,
        suites=suites,
        files=files,
        strategy_groups=groups,
        reserved_groups=sorted(reserved_groups),
        reserved_positions=sorted(reserved_positions),
    )
    atomic_json(path, manifest)
    atomic_json(path.with_name("integrity.json"), {"sha256": sha256(path)})
    return manifest


def load_reservations(root, cfg):
    path = reservation_path(root, cfg)
    if not path.exists():
        raise ValueError("Run notebook 01 to reserve evaluation inputs before training")
    if sha256(path) != read_json(path.with_name("integrity.json"))["sha256"]:
        raise ValueError("Immutable reservation manifest changed")
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or manifest["inputs"] != _inputs(root, cfg):
        raise ValueError("Reservation inputs changed; choose a new reservation_id and training run")
    for relative, digest in manifest["files"].items():
        if sha256(Path(root) / relative) != digest:
            raise ValueError("Immutable reservation suite changed")
    return manifest


def evidence(root, cfg):
    load_reservations(root, cfg)
    path = reservation_path(root, cfg)
    return dict(path=str(path.relative_to(root)), sha256=sha256(path))


def excluded(row, manifest):
    return (
        identity(row["fen"]) in manifest["reserved_positions"]
        or str(
            row.get("source_game_id")
            or row.get("source_id")
            or row.get("game_id")
            or row.get("family_id")
        )
        in manifest["reserved_groups"]
    )


def assert_independent(rows, reservations):
    positions, groups = (
        set(reservations["reserved_positions"]),
        set(reservations["reserved_groups"]),
    )
    for row in rows:
        if (
            identity(row["fen"]) in positions
            or str(row.get("source_game_id") or row.get("source_id") or row.get("game_id"))
            in groups
        ):
            raise ValueError(
                "Training data overlaps reserved evaluation inputs; independent evaluation rejected"
            )


def audit_dataset(root, cfg, manifest):
    reservation = load_reservations(root, cfg)
    if manifest.get("reservations") != evidence(root, cfg):
        raise ValueError(
            "Dataset lacks matching pre-training reservation evidence; rebuild under a new run"
        )
    for split, relative in manifest["paths"].items():
        if sha256(Path(root) / relative) != manifest["hashes"][split]:
            raise ValueError("Recorded training dataset changed")
        assert_independent(read_jsonl(Path(root) / relative), reservation)
    return reservation


def audit_base(root, cfg, checkpoint, manifest):
    from .checkpoints import load_checkpoint

    audit_dataset(root, cfg, manifest)
    payload = load_checkpoint(checkpoint)
    if payload.get("dataset_hashes") != manifest["hashes"]:
        raise ValueError("Base checkpoint training provenance does not match the audited dataset")
    return payload
