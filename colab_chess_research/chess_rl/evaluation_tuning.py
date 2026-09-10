"""Seeded, match-based tuning of classical evaluation coefficients. No RL or model training."""

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import chess
from .classical_evaluation import DEFAULT_WEIGHTS, resolve_weights
from .dataset import identity, read_jsonl
from .environment import terminal
from .evaluation import run_matchup, telemetry
from .non_rl import classical_candidate, source_hashes, variant_config
from .reproducibility import atomic_json, read_json, sha256


def tuning_enabled(cfg):
    return cfg["non_rl"]["variant"] == "classical" and bool(
        cfg["non_rl"].get("tuning", {}).get("enabled", False)
    )


def propose_weights(baseline, ranges, count, seed):
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("tuning.trials must be a positive integer")
    if not ranges or set(ranges) - set(DEFAULT_WEIGHTS):
        raise ValueError("Tuning ranges must name existing evaluation features")
    for name, bounds in ranges.items():
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in bounds
            )
            or bounds[0] > bounds[1]
        ):
            raise ValueError(f"Invalid tuning range for {name}")
    baseline = resolve_weights(baseline)
    rng = random.Random(seed)
    seen = {json.dumps(baseline, sort_keys=True)}
    proposals = []
    for _ in range(count * 100):
        weights = dict(baseline)
        for name in sorted(ranges):
            weights[name] = round(rng.uniform(*ranges[name]), 4)
        key = json.dumps(weights, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(weights)
        if len(proposals) == count:
            return proposals
    raise ValueError("Tuning ranges do not produce enough distinct candidates")


def split_tuning_openings(development, heldout, screening_games, confirmation_games):
    for count in (screening_games, confirmation_games):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or count % 2:
            raise ValueError("Tuning game counts must be positive and even (colour-swapped pairs)")
    first, second = screening_games // 2, confirmation_games // 2
    if first + second > len(development):
        raise ValueError("Development suite is too small for disjoint screening and confirmation")
    groups = [development[:first], development[first : first + second], heldout]
    keys, families = [], []
    for index, group in enumerate(groups):
        positions = [identity(row["fen"]) for row in group]
        if len(set(positions)) != len(positions):
            raise ValueError("Duplicate chess positions in a tuning/held-out partition")
        keys.append(set(positions))
        families.append({row.get("family_id", row["opening_id"]) for row in group})
        if index < 2:
            for row in group:
                board = chess.Board(row["fen"])
                if not board.is_valid() or terminal(board) is not None:
                    raise ValueError("Tuning requires valid, non-terminal starting positions")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if keys[i] & keys[j] or families[i] & families[j]:
            raise ValueError("Screening, confirmation and held-out openings must be disjoint")
    return groups[0], groups[1]


def _inputs(root, cfg):
    root = Path(root)
    settings = cfg["non_rl"]["tuning"]
    threshold = settings["minimum_confirmation_score"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.5 <= threshold <= 1
    ):
        raise ValueError("minimum_confirmation_score must be between 0.5 and 1")
    paths = {name: root / f"datasets/openings/{name}.jsonl" for name in ("development", "heldout")}
    screening, confirmation = split_tuning_openings(
        list(read_jsonl(paths["development"])),
        list(read_jsonl(paths["heldout"])),
        settings["games_per_trial"],
        settings["confirmation_games"],
    )
    proposals = propose_weights(
        cfg["search"].get("evaluation_weights"), settings["ranges"], settings["trials"], cfg["seed"]
    )
    manifest = dict(
        config=cfg,
        sources=source_hashes(root),
        opening_hashes={name: sha256(path) for name, path in paths.items()},
        screening=screening,
        confirmation=confirmation,
        proposals=proposals,
        objective="Screen against fixed baseline; confirm one candidate on unused development openings",
    )
    return manifest


def _directory(root, cfg):
    return Path(root) / "results" / cfg["run_id"] / "no_rl/weight_tuning"


def usable(summary):
    return (
        summary["score"] is not None
        and summary["candidate_failures"] == 0
        and summary["void_games"] == 0
        and summary.get("search_errors", 0) == 0
        and summary.get("model_errors", 0) == 0
        and summary.get("failed_terminations", 0) == 0
        and summary.get("opponent_search_errors", 0) == 0
    )


def confirmed_improvement(summary, threshold):
    lower = summary["interval95"][0]
    return usable(summary) and summary["score"] >= threshold and lower is not None and lower > 0.5


def _match(root, cfg, candidate, baseline, suite, name):
    result = run_matchup(root, candidate, baseline, suite, cfg, name, len(suite) * 2)
    records = Path(root) / "results" / cfg["run_id"] / "matches" / name / "games.jsonl"
    rows = list(read_jsonl(records))
    result["search_errors"] = sum(row["telemetry"].get("search_errors", 0) for row in rows)
    result["failed_terminations"] = sum(
        row["termination"] in {"illegal", "crash", "flag", "init"} for row in rows
    )
    result["opponent_search_errors"] = sum(
        telemetry(path.read_text()).get("search_errors", 0)
        for path in records.parent.glob("game-*-opponent.log")
    )
    return result


def _save_csv(directory, trials):
    path = directory / "trials.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = ["trial", "score", "wins", "draws", "losses", "games", "lower95", "upper95", "usable"]
    fields += list(DEFAULT_WEIGHTS)
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            summary = trial["summary"]
            writer.writerow(
                dict(
                    trial=trial["trial"],
                    **{name: summary[name] for name in fields[1:6]},
                    lower95=summary["interval95"][0],
                    upper95=summary["interval95"][1],
                    usable=usable(summary),
                    **trial["weights"],
                )
            )
    temporary.replace(path)


def tune_classical_weights(root, cfg):
    """Run/resume screening and confirmation; preserve baseline if the evidence is insufficient."""
    if not tuning_enabled(cfg):
        return None
    root = Path(root)
    directory = _directory(root, cfg)
    manifest = _inputs(root, cfg)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Weight-tuning inputs changed; use a new no_rl_ run_id")
    if (directory / "selection.json").exists():
        selected_config(root, cfg)
        return read_json(directory / "selection.json")
    if (directory.parent / "heldout_lock.json").exists() or (
        directory.parent / "development.json"
    ).exists():
        raise ValueError("Tune before evaluation/held-out selection, or start a new run_id")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest)
    baseline_weights = resolve_weights(cfg["search"].get("evaluation_weights"))
    baseline = classical_candidate(root, cfg)
    trials = []
    for index, weights in enumerate(manifest["proposals"], 1):
        path = directory / f"trial-{index:03d}.json"
        if path.exists():
            trial = read_json(path)
            if trial["weights"] != weights:
                raise ValueError("Saved trial does not match its planned weights")
        else:
            trial_cfg = variant_config(cfg, "classical")
            trial_cfg["search"]["evaluation_weights"] = weights
            candidate = classical_candidate(root, trial_cfg)
            summary = _match(
                root,
                trial_cfg,
                candidate,
                baseline,
                manifest["screening"],
                f"weight-trial-{index:03d}",
            )
            trial = dict(trial=index, weights=weights, summary=summary)
            atomic_json(path, trial)
        trials.append(trial)
        _save_csv(directory, trials)
        print(
            f"Weight trial {index}/{len(manifest['proposals'])}: score={trial['summary']['score']}, usable={usable(trial['summary'])}",
            flush=True,
        )
    eligible = [
        trial for trial in trials if usable(trial["summary"]) and trial["summary"]["score"] > 0.5
    ]
    best = (
        max(eligible, key=lambda trial: (trial["summary"]["score"], -trial["trial"]))
        if eligible
        else None
    )
    confirmation = None
    accepted = False
    if best is not None:
        trial_cfg = variant_config(cfg, "classical")
        trial_cfg["search"]["evaluation_weights"] = best["weights"]
        candidate = classical_candidate(root, trial_cfg)
        confirmation = _match(
            root, trial_cfg, candidate, baseline, manifest["confirmation"], "weight-confirmation"
        )
        accepted = confirmed_improvement(
            confirmation, cfg["non_rl"]["tuning"]["minimum_confirmation_score"]
        )
    result = dict(
        status="complete",
        manifest_sha256=sha256(manifest_path),
        baseline_weights=baseline_weights,
        selected_weights=best["weights"] if accepted else baseline_weights,
        accepted=accepted,
        best_screening_trial=best["trial"] if best else None,
        confirmation=confirmation,
        trials=trials,
        decision="confirmed_on_unused_development_openings"
        if accepted
        else "baseline_retained_insufficient_evidence",
    )
    atomic_json(directory / "selected_weights.json", result["selected_weights"])
    atomic_json(directory / "selection.json", result)
    return result


def selected_config(root, cfg):
    directory = _directory(root, cfg)
    selection_path = directory / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("Run weight tuning in no-RL notebook 03 before evaluating this run")
    manifest = directory / "manifest.json"
    result = read_json(selection_path)
    if (
        result["status"] != "complete"
        or result["manifest_sha256"] != sha256(manifest)
        or read_json(manifest) != _inputs(root, cfg)
    ):
        raise ValueError("Weight-tuning inputs changed; use a new no_rl_ run_id")
    resolved = deepcopy(cfg)
    resolved["search"]["evaluation_weights"] = resolve_weights(result["selected_weights"])
    return resolved
