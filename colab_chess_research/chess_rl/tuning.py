"""Sequential, resumable Optuna studies; separate run directories for parallel sessions."""

from copy import deepcopy
import pickle
from pathlib import Path
import optuna
import torch
from .training import fit_supervised
from .checkpoints import load_checkpoint
from .reproducibility import atomic_json
from .self_play import run_league
from .evaluation import build_research_agent, run_matchup


def suggest(trial, name, specification):
    if isinstance(specification, list):
        return trial.suggest_categorical(name, specification)
    return trial.suggest_float(
        name, specification["low"], specification["high"], log=specification.get("log", False)
    )


def study_for(directory, seed, direction):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "study.pickle"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # Only this project's trusted, single-writer study.
    return optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed), direction=direction)


def persist_study(directory, study):
    temporary = directory / "study.pickle.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(study, stream)
    temporary.replace(directory / "study.pickle")
    records = [
        dict(
            number=t.number,
            state=t.state.name,
            value=t.value,
            params=t.params,
            user_attrs=t.user_attrs,
        )
        for t in study.trials
    ]
    atomic_json(directory / "trials.json", records)


def run_model_study(root, cfg, manifest, spaces):
    root = Path(root)
    directory = root / "logs" / cfg["run_id"] / "model_study"
    study = study_for(directory, cfg["seed"], "minimize")

    def objective(trial):
        variant = deepcopy(cfg)
        variant["run_id"] += f"-model-{trial.number:03d}"
        for section in ("model", "training"):
            for key, values in spaces[section].items():
                variant[section][key] = suggest(trial, section + "." + key, values)
        variant["training"]["value_weight"] = 1.0
        checkpoint = fit_supervised(root, variant, manifest)
        trial.set_user_attr("checkpoint", str(checkpoint.relative_to(root)))
        trial.set_user_attr("config", variant)
        return load_checkpoint(checkpoint)["metrics"]["total_loss"]

    def callback(study, trial):
        persist_study(directory, study)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    remaining = max(0, spaces["model_trials"] - len(study.trials))
    study.optimize(
        objective,
        n_trials=remaining,
        callbacks=[callback],
        catch=(RuntimeError, ValueError, OSError),
    )
    persist_study(directory, study)
    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )
    shortlist = [t.user_attrs for t in completed[:3]]
    atomic_json(directory / "shortlist.json", shortlist)
    return shortlist


def run_self_play_search_study(root, cfg, initial_checkpoint, manifest, spaces, search_only=False):
    root = Path(root)
    name = "search_study" if search_only else "self_play_study"
    directory = root / "logs" / cfg["run_id"] / name
    study = study_for(directory, cfg["seed"], "maximize")
    reference = build_research_agent(root, initial_checkpoint, cfg, "fixed-study-reference")

    def objective(trial):
        variant = deepcopy(cfg)
        variant["run_id"] += f"-{name}-{trial.number:03d}"
        for key, specification in spaces["search"].items():
            variant["search"][key] = suggest(trial, "search." + key, specification)
        if search_only:
            checkpoint = initial_checkpoint
        else:
            for key, specification in spaces["self_play"].items():
                variant["self_play"][key] = suggest(trial, "self_play." + key, specification)
            variant["training"]["value_weight"] = suggest(
                trial, "training.value_weight", spaces["value_weight"]
            )
            variant["self_play"]["iterations"] = 3
            checkpoint = run_league(root, variant, initial_checkpoint, manifest)
        agent = build_research_agent(root, checkpoint, variant, "trial")
        result = run_matchup(
            root,
            agent,
            reference,
            root / "datasets/openings/development.jsonl",
            variant,
            "study-development",
            games=128,
        )
        trial.set_user_attr("checkpoint", str(Path(checkpoint).relative_to(root)))
        trial.set_user_attr("config", variant)
        trial.set_user_attr("summary", result)
        return (
            result["score"]
            if result["score"] is not None and not result["candidate_failures"]
            else -1.0
        )

    remaining = max(0, spaces["self_play_search_trials"] - len(study.trials))
    study.optimize(
        objective,
        n_trials=remaining,
        callbacks=[lambda study, trial: persist_study(directory, study)],
        catch=(RuntimeError, ValueError, OSError),
    )
    persist_study(directory, study)
    winner = study.best_trial.user_attrs
    # A full paired development evaluation follows the short trial-ranking suite.
    candidate = build_research_agent(
        root, root / winner["checkpoint"], winner["config"], "study-winner"
    )
    winner["full_evaluation"] = run_matchup(
        root,
        candidate,
        reference,
        root / "datasets/openings/development.jsonl",
        winner["config"],
        "study-winner-full",
        games=256,
    )
    atomic_json(directory / "winner.json", winner)
    return winner
