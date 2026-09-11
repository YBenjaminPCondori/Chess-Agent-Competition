"""Candidate discovery, comparable research matches, and immutable winner selection."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from .evaluation import build_research_agent, run_matchup, promotion_allowed
from .non_rl import classical_candidate, source_hashes, variant_config, selected_no_rl
from .reproducibility import atomic_json, read_json, sha256


def _candidate(root, cfg, name, kind, checkpoint=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Candidate id must be a filename-safe identifier")
    checkpoint = (Path(root) / checkpoint).resolve() if checkpoint else None
    if checkpoint and (checkpoint.parent / "complete.json").exists():
        if read_json(checkpoint.parent / "complete.json").get("status") not in (
            "complete",
            "completed",
        ):
            raise ValueError(f"Candidate {name} lacks a completed training record")
    if checkpoint and not (checkpoint.parent / "complete.json").exists():
        if not (kind == "self_play" and (checkpoint.parent / "candidate.json").exists()):
            raise ValueError(f"Candidate {name} has not completed training")
    return dict(
        id=name,
        kind=kind,
        checkpoint=str(checkpoint.relative_to(Path(root).resolve())) if checkpoint else None,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        config=deepcopy(cfg),
    )


def validate_candidate(root, candidate):
    path = candidate["checkpoint"]
    if path and sha256(Path(root) / path) != candidate["checkpoint_sha256"]:
        raise ValueError(f"Candidate checkpoint changed: {candidate['id']}")


def discover_candidates(root, cfg):
    root = Path(root)
    stages = cfg["research_workflow"]["stages"]
    candidates = []
    if stages["broad"] != "skipped":
        initial = read_json(root / "results" / cfg["run_id"] / "initial_selection.json")
        if sha256(root / initial["checkpoint"]) != initial["sha256"]:
            raise ValueError("Broad checkpoint changed")
        candidates.append(_candidate(root, cfg, "broad", "supervised", initial["checkpoint"]))
    strategy_dir = root / "models/strategy" / cfg["run_id"]
    strategy_paths = []
    if stages["strategy"] != "skipped":
        completed = read_json(strategy_dir / "completed.json")
        expected = {cfg["strategy_finetuning"]["combined_id"]} | {
            c["id"] for c in cfg["strategy_finetuning"]["specialists"]
        }
        if set(completed["candidates"]) != expected:
            raise ValueError("Complete all configured strategy candidates before registering")
        strategy_paths = [strategy_dir / name / "specialist.json" for name in sorted(expected)]
    for path in strategy_paths:
        specialist = read_json(path)
        if specialist != completed["candidates"][path.parent.name]:
            raise ValueError("Completed strategy cohort differs from its candidate records")
        if sha256(root / specialist["checkpoint"]) != specialist["sha256"]:
            raise ValueError("Specialist checkpoint changed")
        candidates.append(
            _candidate(
                root,
                specialist["config"],
                path.parent.name,
                specialist["kind"],
                specialist["checkpoint"],
            )
        )
    league_path = root / "checkpoints/self_play" / cfg["run_id"] / "league.json"
    if stages["self_play"] != "skipped":
        if not (league_path.parent / "complete.json").exists():
            raise ValueError("Complete self-play or explicitly mark it skipped")
        league = read_json(league_path)
        candidates.append(
            _candidate(root, league["config"], "self_play", "self_play", league["champion"])
        )
    control = deepcopy(cfg)
    control["non_rl"] = dict(variant="classical", final_games=16)
    control = variant_config(control, "classical")
    candidates.append(_candidate(root, control, "classical", "classical"))
    for run in cfg["research_workflow"]["no_rl_run_ids"]:
        record = selected_no_rl(root, run)
        candidates.append(_candidate(root, record["config"], run, "no_rl", record["checkpoint"]))
    for item in cfg["research_workflow"]["additional_candidates"]:
        candidates.append(
            _candidate(root, item.get("config", cfg), item["id"], item["kind"], item["checkpoint"])
        )
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate ids")
    return candidates


def self_play_inputs(root, cfg):
    root = Path(root)
    configured = cfg["research_workflow"]["self_play_initial_checkpoint"]
    if configured:
        path = root / configured
        if cfg["research_workflow"]["stages"]["strategy"] != "skipped":
            broad = read_json(root / "results" / cfg["run_id"] / "initial_selection.json")
            if sha256(path) == broad["sha256"]:
                raise ValueError("Broad initialization requires strategy explicitly skipped")
        if not (path.parent / "complete.json").exists():
            raise ValueError("Self-play initialization must come from completed training")
    else:
        if cfg["research_workflow"]["stages"]["strategy"] == "skipped":
            initial = read_json(root / "results" / cfg["run_id"] / "initial_selection.json")
        else:
            directory = root / "models/strategy" / cfg["run_id"]
            completed = read_json(directory / "completed.json")
            initial = completed["candidates"][cfg["strategy_finetuning"]["combined_id"]]
        path = root / initial["checkpoint"]
        if sha256(path) != initial["sha256"]:
            raise ValueError("Self-play initialization changed")
    manifest = read_json(root / "datasets/manifests" / (cfg["run_id"] + ".json"))
    from .candidate_registry import audit_checkpoint

    if not (path.parent / "complete.json").exists():
        raise ValueError("Self-play requires completed initialization")
    audit_checkpoint(root, cfg, path)
    return path, manifest


def _adapter(root, candidate, run_id):
    cfg = deepcopy(candidate["config"])
    cfg["run_id"] = run_id
    if candidate["checkpoint"]:
        return build_research_agent(
            root, Path(root) / candidate["checkpoint"], cfg, candidate["id"]
        )
    return classical_candidate(root, cfg)


def general_evaluation(root, cfg, candidates=None):
    from .candidate_registry import load_candidate_registry
    from .reservations import load_reservations

    root = Path(root)
    registry = load_candidate_registry(root, cfg)
    if candidates is not None and candidates != registry["candidates"]:
        raise ValueError("Evaluation must use the immutable registry")
    candidates = registry["candidates"]
    openings = root / load_reservations(root, cfg)["suites"]["development"]
    directory = root / "results" / cfg["run_id"]
    signature = dict(
        candidates=candidates,
        evaluation=cfg["evaluation"],
        sources=source_hashes(root),
        openings_sha256=sha256(openings),
        registry_sha256=sha256(directory / "candidate_registry.json"),
    )
    lock = directory / "general_inputs.json"
    if lock.exists() and read_json(lock) != signature:
        raise ValueError("Evaluated inputs changed; use a new run or restore the fixed candidates")
    atomic_json(lock, signature)
    results = {}
    for candidate in candidates:
        validate_candidate(root, candidate)
        adapter = _adapter(root, candidate, cfg["run_id"])
        results[candidate["id"]] = {}
        for name, opponent in (
            ("greedy", "reference/starter/baselines/greedy"),
            ("minimax", "reference/starter/baselines/minimax"),
            ("classical", "reference/classical_agent"),
        ):
            results[candidate["id"]][name] = run_matchup(
                root,
                adapter,
                root / opponent,
                openings,
                cfg,
                f"general-{candidate['id']}-{name}",
            )
    record = dict(signature=signature, results=results)
    atomic_json(directory / "general_evaluation.json", record)
    return record


def select_winner(candidates, general, strategy, settings):
    ranking = []
    for candidate in candidates:
        summaries = list(general["results"].get(candidate["id"], {}).values())
        themes = [r for r in strategy["summaries"] if r["candidate"] == candidate["id"]]
        usable = bool(summaries) and all(
            s.get("score") is not None
            and math.isfinite(s["score"])
            and s.get("games", 0) > 0
            and s.get("candidate_failures", 0) == 0
            and s.get("model_errors", 0) == 0
            and s.get("search_errors", 0) == 0
            and s.get("opponent_failures", 0) == 0
            and s.get("opponent_errors", 0) == 0
            and s.get("void_games", 0) == 0
            for s in summaries
        )
        score = sum(s["score"] for s in summaries) / len(summaries) if usable else None
        labelled = [r for r in themes if r["status"] in ("pass", "fail")]
        strategy_pass = bool(labelled) and all(r["status"] == "pass" for r in labelled)
        accuracy = [r["accuracy"] for r in themes if r["accuracy"] is not None]
        usable = (
            usable
            and score >= settings["min_general_score"]
            and not any(r["errors"] for r in themes)
        )
        if settings["require_strategy_pass"]:
            usable = usable and strategy_pass
        ranking.append(
            dict(
                id=candidate["id"],
                eligible=bool(usable),
                general_score=score,
                strategy_accuracy=sum(accuracy) / len(accuracy) if accuracy else None,
                strategy_pass=strategy_pass,
            )
        )
    eligible = [r for r in ranking if r["eligible"]]
    if not eligible:
        raise ValueError("No candidate satisfies the documented selection criteria")
    winner = sorted(
        eligible,
        key=lambda r: (
            -r["general_score"],
            r["id"],
        ),
    )[0]
    return next(c for c in candidates if c["id"] == winner["id"]), ranking


def confirm_challenger(root, cfg, challenger, incumbent, general_hash, registry_hash):
    from .reservations import load_reservations, reservation_path, claim_once
    from .dataset import identity, read_jsonl

    root = Path(root)
    reservations = load_reservations(root, cfg)
    claim_path = reservation_path(root, cfg).parent / "confirmation_use.json"
    suite_identity = hashlib.sha256(
        json.dumps(
            sorted(
                identity(r["fen"])
                for r in read_jsonl(root / reservations["suites"]["confirmation"])
            )
        ).encode()
    ).hexdigest()
    shared_claim = root / "data/reservations/confirmation_uses" / (suite_identity + ".json")
    claim = dict(
        run_id=cfg["run_id"],
        challenger=challenger,
        incumbent=incumbent,
        general_sha256=general_hash,
        registry_sha256=registry_hash,
        openings_sha256=sha256(root / reservations["suites"]["confirmation"]),
        evaluation=cfg["evaluation"],
        selection=cfg["research_workflow"]["selection"],
    )
    if any(p.exists() and read_json(p) != claim for p in (claim_path, shared_claim)):
        raise ValueError("Confirmation suite already assigned; never try a second challenger")
    claim_once(shared_claim, claim)
    claim_once(claim_path, claim)
    result_path = root / "results" / cfg["run_id"] / "confirmation.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["claim"] != claim:
            raise ValueError("Confirmation evidence changed")
        return result
    if challenger["id"] == incumbent["id"]:
        result = dict(claim=claim, status="not_applicable", accepted=False, summary=None)
    else:
        summary = run_matchup(
            root,
            _adapter(root, challenger, cfg["run_id"]),
            _adapter(root, incumbent, cfg["run_id"]),
            root / reservations["suites"]["confirmation"],
            cfg,
            "incumbent-confirmation",
            cfg["evaluation"]["confirmation_games"],
        )
        accepted = (
            summary["games"] == cfg["evaluation"]["confirmation_games"]
            and promotion_allowed(summary, 0.55)
            and summary.get("search_errors", 0) == 0
            and summary.get("opponent_failures", 0) == 0
        )
        accepted = accepted and summary.get("opponent_errors", 0) == 0
        result = dict(claim=claim, status="completed", accepted=accepted, summary=summary)
    atomic_json(result_path, result)
    return result


def freeze_winner(root, cfg):
    from .candidate_registry import load_candidate_registry
    from .reservations import load_reservations, evidence
    from .strategy_dataset import load_strategy_manifest
    from .classical_evaluation import resolve_weights

    root = Path(root)
    directory = root / "results" / cfg["run_id"]
    lock = directory / "winner_lock.json"
    if lock.exists() and read_json(lock)["workflow_config"] != cfg:
        raise ValueError("Winner already frozen; held-out results cannot be used to reselect")
    registry = load_candidate_registry(root, cfg)
    reservations = load_reservations(root, cfg)
    registry_path = directory / "candidate_registry.json"
    general_path = directory / "general_evaluation.json"
    strategy_path = root / "results/strategy_evaluation" / cfg["run_id"] / "validation/summary.json"
    general, strategy = read_json(general_path), read_json(strategy_path)
    candidates, sources = registry["candidates"], source_hashes(root)
    if general["signature"]["candidates"] != candidates or strategy["candidates"] != candidates:
        raise ValueError("Both evaluation stages must use the immutable registry")
    if general["signature"]["sources"] != sources or strategy["sources"] != sources:
        raise ValueError("Research source changed since evaluation")
    if (
        general["signature"]["evaluation"] != cfg["evaluation"]
        or strategy["settings"] != cfg["strategy_evaluation"]
    ):
        raise ValueError("Evaluation settings changed since comparison")
    if general["signature"]["openings_sha256"] != sha256(
        root / reservations["suites"]["development"]
    ):
        raise ValueError("Development openings changed since comparison")
    if general["signature"]["registry_sha256"] != sha256(registry_path) or strategy[
        "registry_sha256"
    ] != sha256(registry_path):
        raise ValueError("Evaluations used a different registry")
    for candidate in candidates:
        summaries = general["results"].get(candidate["id"], {})
        if set(summaries) != {"greedy", "minimax", "classical"} or any(
            s["games"] != cfg["evaluation"]["development_games"] for s in summaries.values()
        ):
            raise ValueError("Complete identical development comparisons before selection")
    for relative, digest in strategy["dataset_hashes"].items():
        if sha256(root / relative) != digest:
            raise ValueError("Strategy suites changed since evaluation")
        load_strategy_manifest(root, relative)
    challenger, ranking = select_winner(
        candidates, general, strategy, cfg["research_workflow"]["selection"]
    )
    incumbent = next(c for c in candidates if c["id"] == "classical" and not c["checkpoint"])
    confirmation = confirm_challenger(
        root, cfg, challenger, incumbent, sha256(general_path), sha256(registry_path)
    )
    winner = challenger if confirmation["accepted"] else incumbent
    selected_cfg = deepcopy(winner["config"])
    selected_cfg["run_id"] = cfg["run_id"]
    selected_cfg["evaluation"] = cfg["evaluation"]
    selected_cfg.setdefault(
        "non_rl",
        dict(variant="supervised" if winner["checkpoint"] else "classical", final_games=16),
    )
    stages = deepcopy(registry["stages"])
    stages.update(
        general_evaluation=dict(status="completed"),
        strategy_evaluation=dict(status="completed"),
        confirmation=dict(status=confirmation["status"]),
    )
    record = dict(
        status="frozen",
        workflow="strategy_research",
        workflow_config=cfg,
        selected_id=winner["id"],
        kind=winner["kind"],
        checkpoint=winner["checkpoint"],
        checkpoint_sha256=winner["checkpoint_sha256"],
        config=selected_cfg,
        sources=sources,
        classical_coefficients=resolve_weights(selected_cfg["search"].get("evaluation_weights"))
        if not winner["checkpoint"]
        else None,
        stages=stages,
        reservations=evidence(root, cfg),
        evaluation_complete=False,
        selection_criteria=cfg["research_workflow"]["selection"],
        ranking=ranking,
        confirmation=confirmation,
        evaluation_summaries=dict(general=general, strategy=strategy),
        evaluation_hashes={
            str(p.relative_to(root)): sha256(p)
            for p in (general_path, strategy_path, registry_path, directory / "confirmation.json")
        },
        opening_hashes=reservations["files"],
    )
    record["config_sha256"] = hashlib.sha256(
        json.dumps(selected_cfg, sort_keys=True).encode()
    ).hexdigest()
    if lock.exists() and read_json(lock) != record:
        raise ValueError("Winner already frozen; held-out results cannot be used to reselect")
    atomic_json(lock, record)
    heldout = run_matchup(
        root,
        _adapter(root, winner, cfg["run_id"]),
        root / "reference/classical_agent",
        root / reservations["suites"]["heldout"],
        cfg,
        f"winner-heldout-{winner['id']}",
        cfg["evaluation"]["heldout_games"],
    )
    if (
        heldout["candidate_failures"]
        or heldout.get("model_errors", 0)
        or heldout["void_games"]
        or heldout.get("search_errors", 0)
    ):
        raise RuntimeError(
            "Held-out operational failures; inspect the frozen run without reselection"
        )
    from .strategy_evaluation import evaluate_candidates

    heldout_cfg = deepcopy(cfg)
    heldout_cfg["strategy_evaluation"]["split"] = "test"
    strategy_test = evaluate_candidates(root, heldout_cfg, [winner])
    if any(row["errors"] for row in strategy_test["summaries"]):
        raise RuntimeError("Held-out strategy prediction errors; inspect the frozen run")
    record.update(
        evaluation_complete=True,
        heldout_results=heldout,
        strategy_test=strategy_test,
        final_stages=dict(heldout=dict(status="completed")),
        export_deferred=True,
    )
    path = directory / "final_selection.json"
    atomic_json(path, record)
    return path
