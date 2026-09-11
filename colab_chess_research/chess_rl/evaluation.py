"""Paired CPU games through the preserved harness; no export or package checks."""

import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import hashlib
import numpy as np
from .dataset import read_jsonl, write_jsonl
from .reproducibility import atomic_json, read_json, sha256, append_csv


def harness_modules(root):
    location = str(Path(root).resolve() / "reference/starter")
    if location not in sys.path:
        sys.path.insert(0, location)
    referee = importlib.import_module("harness.referee")
    sandbox = importlib.import_module("harness.sandbox")
    if not Path(referee.__file__).resolve().is_relative_to(Path(location)):
        raise RuntimeError("Another harness is shadowing the preserved reference")
    return referee, sandbox


def build_research_agent(root, checkpoint, cfg, label="candidate"):
    root = Path(root).resolve()
    digest = sha256(checkpoint) if checkpoint else "classical_research"
    settings = dict(
        project_root=str(root),
        checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
        search=cfg["search"],
        version=digest,
    )
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    directory = root / "results" / cfg["run_id"] / "research_agents" / f"{label}-{identity}"
    directory.mkdir(parents=True, exist_ok=True)
    settings_path = directory / "research.json"
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError("Research adapter identity collision")
    atomic_json(settings_path, settings)
    shutil.copy2(root / "templates/research_agent.py", directory / "agent.py")
    return directory


class MeasuredAgent:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.times = []
        self.clocks = []
        self.failure = None
        self.last_move_started = -1.0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def start(self, budget):
        started = time.monotonic()
        try:
            return self.wrapped.start(budget)
        except Exception as error:
            self.failure = getattr(error, "reason", "init")
            raise
        finally:
            self.init_ms = (time.monotonic() - started) * 1000

    def move(self, fen, time_left_ms):
        self.clocks.append(time_left_ms)
        started = time.monotonic()
        self.last_move_started = started
        try:
            return self.wrapped.move(fen, time_left_ms)
        except Exception as error:
            self.failure = getattr(error, "reason", "crash")
            raise
        finally:
            self.times.append((time.monotonic() - started) * 1000)


def telemetry(log):
    metrics = {}
    for line in log.splitlines():
        if line.startswith("CHESS_METRICS "):
            try:
                metrics = json.loads(line[len("CHESS_METRICS ") :])
            except json.JSONDecodeError:
                continue
    return metrics


def summarize(rows, samples=10000, seed=42):
    valid = [r for r in rows if r["score"] is not None]
    scores = [r["score"] for r in valid]
    groups = {}
    for row in valid:
        groups.setdefault(row["family_id"], []).append(row["score"])
    groups = [values for values in groups.values() if len(values) % 2 == 0]
    if len(groups) >= 2:
        rng = np.random.default_rng(seed)
        totals = np.array([sum(g) for g in groups])
        counts = np.array([len(g) for g in groups])
        selected = rng.integers(len(groups), size=(samples, len(groups)))
        bootstrap = totals[selected].sum(1) / counts[selected].sum(1)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    else:
        interval = [None, None]
    times = [v for row in rows for v in row["move_times_ms"]]
    metrics = dict(
        games=len(rows),
        scored_games=len(scores),
        void_games=len(rows) - len(scores),
        wins=scores.count(1.0),
        draws=scores.count(0.5),
        losses=scores.count(0.0),
        score=float(np.mean(scores)) if scores else None,
        interval95=interval,
        candidate_failures=sum(bool(row["candidate_failure"]) for row in rows),
        bootstrap_groups=len(groups),
        bootstrap_samples=samples,
    )
    if times:
        metrics["move_time_ms"] = dict(
            mean=float(np.mean(times)),
            p50=float(np.median(times)),
            p95=float(np.quantile(times, 0.95)),
            maximum=max(times),
        )
    metrics["fallback_moves"] = sum(r["telemetry"].get("fallback_moves", 0) for r in rows)
    metrics["model_errors"] = sum(r["telemetry"].get("model_errors", 0) for r in rows)
    metrics["search_errors"] = sum(r["telemetry"].get("search_errors", 0) for r in rows)
    metrics["opponent_failures"] = sum(bool(r.get("opponent_failure")) for r in rows)
    metrics["opponent_errors"] = sum(
        r.get("opponent_telemetry", {}).get(key, 0)
        for r in rows
        for key in ("model_errors", "search_errors")
    )
    metrics["nodes"] = sum(r["telemetry"].get("nodes", 0) for r in rows)
    metrics["inference_calls"] = sum(r["telemetry"].get("inference_calls", 0) for r in rows)
    metrics["inference_ms"] = sum(r["telemetry"].get("inference_ms", 0) for r in rows)
    return metrics


def run_matchup(root, candidate, opponent, openings, cfg, name, games=None):
    root, candidate, opponent = Path(root), Path(candidate), Path(opponent)
    suite = list(read_jsonl(openings)) if isinstance(openings, (str, Path)) else list(openings)
    games = games or cfg["evaluation"]["development_games"]
    if games % 2 or games > len(suite) * 2:
        raise ValueError("Use one colour-swapped pair per opening and an even supported game count")
    directory = root / "results" / cfg["run_id"] / "matches" / name
    directory.mkdir(parents=True, exist_ok=True)

    def fingerprint(path):
        return {
            str(p.relative_to(path)): sha256(p)
            for p in path.rglob("*")
            if p.is_file() and p.suffix in (".py", ".json", ".pt")
        }

    manifest = dict(
        candidate=fingerprint(candidate),
        opponent=fingerprint(opponent),
        candidate_path=str(candidate.resolve()),
        opponent_path=str(opponent.resolve()),
        openings=suite[: games // 2],
        evaluation=cfg["evaluation"],
        seed=cfg["seed"],
    )
    if (directory / "manifest.json").exists() and read_json(
        directory / "manifest.json"
    ) != manifest:
        raise ValueError("Match inputs changed; use a new match name")
    atomic_json(directory / "manifest.json", manifest)
    referee, sandbox = harness_modules(root)
    rows = []
    for game_index in range(games):
        saved = directory / f"game-{game_index:04d}.json"
        if saved.exists():
            rows.append(read_json(saved))
            continue
        opening = suite[game_index // 2]
        candidate_white = game_index % 2 == 0
        first = MeasuredAgent(sandbox.local(candidate, cfg["seed"] + game_index))
        second = MeasuredAgent(sandbox.local(opponent, cfg["seed"] + game_index))
        white, black = (first, second) if candidate_white else (second, first)
        outcome = referee.play_match(
            white,
            black,
            cfg["evaluation"]["base_ms"],
            cfg["evaluation"]["increment_ms"],
            start_fen=opening["fen"],
        )
        score = (
            None
            if outcome.result == "void"
            else 0.5
            if outcome.result == "draw"
            else float(outcome.result == ("white" if candidate_white else "black"))
        )
        failure = first.failure
        opponent_failure = second.failure
        if outcome.termination in {"illegal", "crash", "flag", "init"}:
            if score == 0:
                failure = outcome.termination
            elif score == 1:
                opponent_failure = outcome.termination
            elif (
                outcome.termination == "flag"
                and score == 0.5
                and first.last_move_started > second.last_move_started
            ):
                failure = "flag"
            elif outcome.termination == "flag" and score == 0.5:
                opponent_failure = "flag"
        row = dict(
            game_index=game_index,
            opening_id=opening["opening_id"],
            family_id=opening.get("family_id", opening["opening_id"]),
            candidate_white=candidate_white,
            score=score,
            result=outcome.result,
            termination=outcome.termination,
            candidate_failure=failure,
            opponent_failure=opponent_failure,
            init_ms=first.init_ms,
            move_times_ms=first.times,
            clocks_before_ms=first.clocks,
            telemetry=telemetry(first.stderr_log),
            opponent_telemetry=telemetry(second.stderr_log),
        )
        (directory / f"game-{game_index:04d}.pgn").write_text(outcome.pgn + "\n")
        (directory / f"game-{game_index:04d}-candidate.log").write_text(first.stderr_log)
        (directory / f"game-{game_index:04d}-opponent.log").write_text(second.stderr_log)
        atomic_json(saved, row)
        rows.append(row)
        print(
            f"{name}: {game_index + 1}/{games}, {outcome.result}, {outcome.termination}", flush=True
        )
    write_jsonl(directory / "games.jsonl", rows)
    summary = summarize(rows, cfg["evaluation"]["bootstrap_samples"], cfg["seed"])
    atomic_json(directory / "summary.json", summary)
    table = directory / "results.csv"
    if not table.exists():
        for row in rows:
            append_csv(
                table,
                {
                    key: row[key]
                    for key in (
                        "game_index",
                        "opening_id",
                        "candidate_white",
                        "score",
                        "result",
                        "termination",
                        "candidate_failure",
                    )
                },
            )
    return summary


def promotion_allowed(summary, threshold=0.55):
    lower = summary["interval95"][0]
    return bool(
        summary["score"] is not None
        and summary["score"] >= threshold
        and lower is not None
        and lower > 0.5
        and summary["candidate_failures"] == 0
        and summary["void_games"] == 0
        and summary.get("model_errors", 0) == 0
    )


def compare_candidates(root, checkpoint, cfg, previous=None, heldout=False):
    root = Path(root)
    suite = root / f"datasets/openings/{'heldout' if heldout else 'development'}.jsonl"
    count = cfg["evaluation"]["heldout_games" if heldout else "development_games"]
    if heldout:
        lock = root / "results" / cfg["run_id"] / "heldout_selection.json"
        selection = dict(checkpoint_sha256=sha256(checkpoint), search=cfg["search"])
        if lock.exists() and read_json(lock) != selection:
            raise ValueError("Held-out model is already frozen; do not retune on these results")
        atomic_json(lock, selection)
    candidate = build_research_agent(root, checkpoint, cfg)
    opponents = dict(
        greedy=root / "reference/starter/baselines/greedy",
        minimax=root / "reference/starter/baselines/minimax",
        classical=root / "reference/classical_agent",
    )
    if previous:
        opponents["previous"] = build_research_agent(root, previous, cfg, "previous")
    summaries = {}
    for name, opponent in opponents.items():
        match_name = f"{'heldout' if heldout else 'development'}-{name}-{sha256(checkpoint)[:12]}"
        summaries[name] = run_matchup(root, candidate, opponent, suite, cfg, match_name, count)
    return summaries


def run_ablations(root, checkpoint, cfg, supervised=None):
    from copy import deepcopy

    root = Path(root)
    variants = dict(
        classical=(None, False, 0.0),
        policy=(checkpoint, True, 0.0),
        value=(checkpoint, False, 1.0),
        hybrid=(checkpoint, True, 0.5),
    )
    if supervised:
        variants["supervised"] = (supervised, True, 0.5)
    results = {}
    for name, (weights, policy, mix) in variants.items():
        variant = deepcopy(cfg)
        variant["search"].update(policy_ordering=policy, value_eval_mix=mix)
        agent = build_research_agent(root, weights, variant, name)
        results[name] = run_matchup(
            root,
            agent,
            root / "reference/classical_agent",
            root / "datasets/openings/development.jsonl",
            variant,
            f"ablation-{name}-{sha256(checkpoint)[:12]}",
        )
    return results


def finalize_selection(root, checkpoint, cfg, heldout_results):
    root = Path(root)
    for stage in ("supervised", "self_play"):
        complete = root / "checkpoints" / stage / cfg["run_id"] / "complete.json"
        if not complete.exists():
            raise RuntimeError(f"{stage} training has not recorded completion")
    record = dict(
        status="frozen",
        checkpoint=str(Path(checkpoint).relative_to(root)),
        checkpoint_sha256=sha256(checkpoint),
        config=cfg,
        heldout_results=heldout_results,
        training_complete=True,
        evaluation_complete=True,
    )
    path = root / "results" / cfg["run_id"] / "final_selection.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("Final selection already exists and differs")
    atomic_json(path, record)
    return path
