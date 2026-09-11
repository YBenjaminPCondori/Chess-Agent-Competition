"""One immutable candidate cohort, created without running either evaluation stage."""

from pathlib import Path
from .checkpoints import load_checkpoint
from .dataset import read_jsonl
from .reproducibility import atomic_json, read_json, sha256
from .reservations import load_reservations, evidence, audit_base, assert_independent


def audit_checkpoint(root, cfg, path, visited=None):
    root, path = Path(root), Path(path)
    visited = set() if visited is None else visited
    digest = sha256(path)
    if digest in visited:
        raise ValueError("Cyclic checkpoint ancestry")
    visited.add(digest)
    payload = load_checkpoint(path)
    broad_path = root / "datasets/manifests" / (cfg["run_id"] + ".json")
    broad = read_json(broad_path)
    if "inputs" in payload:
        from .strategy_dataset import load_strategy_manifest

        inputs = payload["inputs"]
        if inputs["reservations"] != evidence(root, cfg):
            raise ValueError("Strategy checkpoint reservation provenance differs")
        for relative, expected in inputs["data_hashes"].items():
            if sha256(root / relative) != expected:
                raise ValueError("Checkpoint training data changed")
            if read_json(root / relative).get("schema_version") == 2:
                load_strategy_manifest(root, relative)
        for relative, expected in inputs["candidate_hashes"].items():
            if sha256(root / relative) != expected:
                raise ValueError("Candidate training rows changed")
        rows = read_jsonl(path.parent / "data/train.jsonl")
        assert_independent(rows, load_reservations(root, cfg))
        parent = inputs["parent"]
    elif "replay_manifest" in payload:
        from .reservations import audit_dataset

        reserved = audit_dataset(root, cfg, broad)
        replay_manifest = payload["replay_manifest"]
        if replay_manifest.get("reservations") != evidence(root, cfg):
            raise ValueError("Self-play lacks matching reservation evidence")
        forbidden = set(replay_manifest["forbidden_positions"])
        if not set(reserved["reserved_positions"]) <= forbidden:
            raise ValueError("Self-play did not exclude all reserved positions")
        parent = payload.get("parent")
        if not parent:
            raise ValueError("Self-play checkpoint lacks auditable parent provenance")
        for relative, expected in payload["replay_manifest"]["files"].items():
            if sha256(root / relative) != expected:
                raise ValueError("Self-play replay changed")
            # Collection may contain filtered positions; audit precisely the eligible replay.
            from .dataset import identity

            assert_independent(
                (
                    r
                    for r in read_json(root / relative)["records"]
                    if identity(r["fen"]) not in forbidden
                ),
                reserved,
            )
    else:
        audit_base(root, cfg, path, broad)
        return
    if sha256(root / parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("Parent checkpoint changed")
    audit_checkpoint(root, cfg, root / parent["checkpoint"], visited)


def load_candidate_registry(root, cfg):
    from .research_workflow import discover_candidates, validate_candidate, source_hashes

    root = Path(root)
    path = root / "results" / cfg["run_id"] / "candidate_registry.json"
    fixed = dict(config=cfg, reservations=evidence(root, cfg), sources=source_hashes(root))
    if path.exists():
        record = read_json(path)
        if any(record[key] != value for key, value in fixed.items()):
            raise ValueError("Immutable candidate registry inputs changed; use a new run")
        if record["candidates"] != discover_candidates(root, cfg):
            raise ValueError("Candidate cohort changed after registration")
    else:
        candidates = discover_candidates(root, cfg)
        stages = {}
        for name in ("broad", "strategy", "self_play"):
            state = cfg["research_workflow"]["stages"][name]
            if state not in ("required", "skipped"):
                raise ValueError("Neural stages must be required or explicitly skipped")
            stages[name] = dict(status="skipped" if state == "skipped" else "completed")
        if stages["broad"]["status"] == "skipped" and any(
            stages[s]["status"] == "completed" for s in ("strategy", "self_play")
        ):
            raise ValueError("Strategy/self-play require a completed broad parent")
        record = dict(schema_version=1, **fixed, candidates=candidates, stages=stages)
    for candidate in record["candidates"]:
        validate_candidate(root, candidate)
        if candidate["checkpoint"]:
            audit_checkpoint(root, cfg, root / candidate["checkpoint"])
    if not path.exists():
        atomic_json(path, record)
    return record
