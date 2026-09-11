"""Independent classical/supervised research workflow. No self-play dependencies."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from .config import load_config
from .classical_evaluation import resolve_weights
from .evaluation import build_research_agent, run_matchup
from .reproducibility import atomic_json, read_json, sha256

CLASSICAL_FILES = (
    "__init__.py",
    "piece_tables.py",
    "classical_evaluation.py",
    "environment.py",
    "time_management.py",
    "transposition.py",
    "search.py",
)


def load_no_rl_config(root):
    cfg = load_config(root, "no_rl.yaml")
    cfg.pop("self_play", None)
    if not cfg["run_id"].startswith("no_rl_"):
        raise ValueError("Use a separate run_id starting with no_rl_")
    if cfg["non_rl"]["variant"] not in {"classical", "supervised"}:
        raise ValueError("non_rl.variant must be classical or supervised")
    cfg["search"]["evaluation_weights"] = resolve_weights(cfg["search"].get("evaluation_weights"))
    return variant_config(cfg, cfg["non_rl"]["variant"])


def variant_config(cfg, variant):
    if variant not in {"classical", "supervised"}:
        raise ValueError("Unknown no-RL variant")
    result = deepcopy(cfg)
    result["non_rl"]["variant"] = variant
    if variant == "classical":
        result["search"].update(policy_ordering=False, value_eval_mix=0.0)
    return result


def source_hashes(root):
    root = Path(root)
    paths = list((root / "chess_rl").glob("*.py"))
    paths += list((root / "templates").glob("*.py"))
    paths += list((root / "reference/starter/harness").glob("*.py"))
    paths += list((root / "reference/starter/baselines").glob("*/agent.py"))
    paths += [root / "reference/classical_agent/agent.py"]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def candidate_spec(root, cfg):
    root = Path(root)
    from .evaluation_tuning import tuning_enabled, selected_config

    if tuning_enabled(cfg):
        cfg = selected_config(root, cfg)
    variant = cfg["non_rl"]["variant"]
    checkpoint = None
    if variant == "supervised":
        from .checkpoints import checkpoint_path

        directory = root / "checkpoints/supervised" / cfg["run_id"]
        if not (directory / "complete.json").exists():
            raise RuntimeError("Complete no-RL notebook 02 for the supervised variant")
        checkpoint = checkpoint_path(directory, "best_validation")
    return dict(
        workflow="no_rl",
        variant=variant,
        config=variant_config(cfg, variant),
        checkpoint=str(checkpoint.relative_to(root)) if checkpoint else None,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        sources=source_hashes(root),
        opening_hashes={
            name: sha256(root / "datasets/openings" / name)
            for name in ("development.jsonl", "heldout.jsonl")
        },
    )


def classical_candidate(root, cfg):
    root = Path(root)
    cfg = variant_config(cfg, "classical")
    identity = dict(search=cfg["search"], sources=source_hashes(root))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = root / "results" / cfg["run_id"] / "research_agents" / f"classical-{digest}"
    (directory / "chess_runtime").mkdir(parents=True, exist_ok=True)
    for name in CLASSICAL_FILES:
        shutil.copy2(root / "chess_rl" / name, directory / "chess_runtime" / name)
    shutil.copy2(root / "templates/classical_agent.py", directory / "agent.py")
    atomic_json(directory / "config.json", dict(search=cfg["search"], model_version=digest))
    return directory


def _candidate(root, spec):
    if spec["variant"] == "classical":
        return classical_candidate(root, spec["config"])
    return build_research_agent(
        root, Path(root) / spec["checkpoint"], spec["config"], "no-rl-supervised"
    )


def _opponents(root):
    root = Path(root)
    return dict(
        greedy=root / "reference/starter/baselines/greedy",
        minimax=root / "reference/starter/baselines/minimax",
        original_classical=root / "reference/classical_agent",
    )


def evaluate_no_rl(root, cfg):
    """Development games only. Supervised runs also measure the untrained classical control."""
    root = Path(root)
    spec = candidate_spec(root, cfg)
    directory = root / "results" / cfg["run_id"] / "no_rl"
    prior = directory / "development.json"
    if prior.exists() and read_json(prior)["candidate"] != spec:
        raise ValueError("Development inputs changed; choose a new no_rl_ run_id")
    candidate = _candidate(root, spec)
    suite = root / "datasets/openings/development.jsonl"
    summaries = {
        name: run_matchup(root, candidate, opponent, suite, spec["config"], f"no-rl-dev-{name}")
        for name, opponent in _opponents(root).items()
    }
    controls = {}
    if spec["variant"] == "supervised":
        control_cfg = variant_config(cfg, "classical")
        control = classical_candidate(root, control_cfg)
        for name, opponent in _opponents(root).items():
            controls[name] = run_matchup(
                root, control, opponent, suite, control_cfg, f"no-rl-control-{name}"
            )
        summaries["matched_classical"] = run_matchup(
            root, candidate, control, suite, spec["config"], "no-rl-dev-matched-classical"
        )
    record = dict(candidate=spec, summaries=summaries, classical_control=controls)
    atomic_json(prior, record)
    return record


def evaluate_and_freeze_no_rl(root, cfg):
    """Freeze the chosen candidate before held-out games; never requires self-play completion."""
    root = Path(root)
    directory = root / "results" / cfg["run_id"] / "no_rl"
    development = read_json(directory / "development.json")
    spec = candidate_spec(root, cfg)
    if development["candidate"] != spec:
        raise ValueError("Candidate changed since development evaluation; use a new run_id")
    lock = directory / "heldout_lock.json"
    if lock.exists() and read_json(lock) != spec:
        raise ValueError("Held-out selection is already locked")
    atomic_json(lock, spec)
    candidate = _candidate(root, spec)
    suite = root / "datasets/openings/heldout.jsonl"
    summaries = {
        name: run_matchup(
            root,
            candidate,
            opponent,
            suite,
            spec["config"],
            f"no-rl-heldout-{name}",
            cfg["evaluation"]["heldout_games"],
        )
        for name, opponent in _opponents(root).items()
    }
    record = dict(
        status="frozen",
        **spec,
        evaluation_complete=True,
        training_required=spec["variant"] == "supervised",
        training_status="completed" if spec["checkpoint"] else "not_applicable",
        development=development["summaries"],
        heldout=summaries,
    )
    atomic_json(directory / "final_selection.json", record)
    return record


def selected_no_rl(root, run_id):
    root = Path(root)
    selection = read_json(root / "results" / run_id / "no_rl/final_selection.json")
    if (
        selection.get("workflow"),
        selection.get("status"),
        selection.get("evaluation_complete"),
    ) != ("no_rl", "frozen", True):
        raise RuntimeError("Complete no-RL notebooks 03 and 04 first")
    if selection["sources"] != source_hashes(root):
        raise ValueError("Source changed after selection; evaluate a new run before exporting")
    if any(
        sha256(root / "datasets/openings" / name) != digest
        for name, digest in selection["opening_hashes"].items()
    ):
        raise ValueError("Opening suite changed after selection")
    checkpoint = selection["checkpoint"]
    if checkpoint and sha256(root / checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected supervised checkpoint changed")
    return selection
