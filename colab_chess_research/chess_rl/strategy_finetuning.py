"""Independent combined/specialist transfer runs with deterministic mixed replay."""

from copy import deepcopy
import math
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from .checkpoints import load_model, load_checkpoint, save_checkpoint, checkpoint_path, set_pointer
from .config import load_config, merge
from .dataset import identity, PositionDataset, read_jsonl, write_jsonl
from .reproducibility import atomic_json, read_json, sha256, seed_all, resolve_device, restore_rng
from .reservations import safe_id, load_reservations, evidence, audit_base, assert_independent
from .strategy_dataset import load_strategy_manifest, to_training_record
from .strategy_taxonomy import validate_theme
from .training import OptimizerLoop, evaluate_dataset, make_loader


def load_finetuning_config(root):
    cfg = load_config(root, "strategy.yaml")
    settings = cfg["strategy_finetuning"]
    safe_id(settings["combined_id"])
    ids = [settings["combined_id"]]
    for candidate in settings["specialists"]:
        ids.append(safe_id(candidate["id"]))
        if not candidate["themes"]:
            raise ValueError("Specialists require at least one theme")
        for theme in candidate["themes"]:
            validate_theme(theme)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate strategy candidate ids")
    for name in ("heads", "full"):
        phase = settings[name]
        if type(phase["epochs"]) is not int or phase["epochs"] < 1 or phase["learning_rate"] <= 0:
            raise ValueError("Each phase requires positive epochs and learning rate")
    if not 0 < settings["replay_fraction"] < 1:
        raise ValueError("Replay fraction must be between zero and one")
    return cfg


class MixedBatchSampler:
    """Every batch has an exact replay quota; themes cycle with equal exposure."""

    def __init__(self, themes, replay_count, batch_size, replay_fraction, seed, steps=None):
        quota = float(batch_size * replay_fraction)
        if not quota.is_integer() or not 0 < quota < batch_size or not themes or replay_count < 1:
            raise ValueError("Batch size must support exact nonzero strategy/replay quotas")
        self.replay_size, self.strategy_size = int(quota), batch_size - int(quota)
        self.themes, self.replay_count, self.seed = list(themes), replay_count, seed
        self.steps = steps or math.ceil(len(themes) / self.strategy_size)
        if self.steps < 1:
            raise ValueError("At least one batch is required")

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed)
        pools = {}
        for index, theme in enumerate(self.themes):
            pools.setdefault(theme, []).append(index)
        names = sorted(pools)
        rng.shuffle(names)
        cursor = 0
        for _ in range(self.steps):
            batch = []
            for _ in range(self.strategy_size):
                batch.append(rng.choice(pools[names[cursor % len(names)]]))
                cursor += 1
            batch.extend(
                len(self.themes) + rng.randrange(self.replay_count) for _ in range(self.replay_size)
            )
            rng.shuffle(batch)
            yield batch


def train_candidate(
    root, cfg, initial, directory, strategy, replay, validation, signature, epoch_callback=None
):
    """Resume completed epochs; optimizer/cosine schedules restart at the full phase."""
    directory = Path(directory)
    lock = directory / "inputs.json"
    if lock.exists() and read_json(lock) != signature:
        raise ValueError("Candidate inputs changed; use a new candidate/run id")
    atomic_json(lock, signature)
    if (directory / "complete.json").exists():
        winner = checkpoint_path(directory, "best_validation")
        if load_checkpoint(winner)["inputs"] != signature:
            raise ValueError("Completed candidate provenance changed")
        return winner
    seed_all(cfg["seed"], cfg.get("deterministic", False))
    device = resolve_device(cfg["device"])
    model, _ = load_model(initial, device)
    settings = cfg["strategy_finetuning"]
    training = merge(cfg["training"], settings["training"])
    training_cfg = dict(cfg, training=training)
    batch_size = training["batch_size_cuda" if device.startswith("cuda") else "batch_size_cpu"]
    data = PositionDataset([to_training_record(row) for row in strategy] + replay)
    val_data = PositionDataset([to_training_record(row) for row in validation])
    saved = None
    if (directory / "latest.json").exists():
        saved = load_checkpoint(checkpoint_path(directory))
        if saved["inputs"] != signature:
            raise ValueError("Resume parent/data/config changed")
        model.load_state_dict(saved["model_state_dict"])
        restore_rng(saved["rng"])
        if saved.get("improved"):
            set_pointer(directory, "best_validation", checkpoint_path(directory))
    global_epoch = saved["global_epoch"] if saved else 0
    for phase_name in ("heads", "full"):
        if saved and saved["phase"] == "full" and phase_name == "heads":
            continue
        phase = settings[phase_name]
        sampler = MixedBatchSampler(
            [r["theme"] for r in strategy],
            len(replay),
            batch_size,
            settings["replay_fraction"],
            cfg["seed"],
            settings.get("steps_per_epoch"),
        )
        phase_settings = dict(training, learning_rate=phase["learning_rate"])
        loop = OptimizerLoop(model, phase_settings, device, len(sampler) * phase["epochs"])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                phase_name == "full" or name.split(".")[0] not in ("stem", "trunk")
            )
        loop.frozen_modules = [model.stem, model.trunk] if phase_name == "heads" else []
        start, stale, best, progress_best = 0, 0, math.inf, math.inf
        if saved and saved["phase"] == phase_name:
            loop.restore(saved)
            start, stale, best = saved["epoch"], saved["stale"], saved["best_loss"]
            progress_best = saved["progress_loss"]
        for epoch in range(start, phase["epochs"]):
            if phase_name == "full" and stale >= training["early_stopping_patience"]:
                break
            sampler.seed = cfg["seed"] + global_epoch
            loader = DataLoader(
                data,
                batch_sampler=sampler,
                num_workers=0,
                generator=torch.Generator().manual_seed(sampler.seed),
            )
            for batch in loader:
                loop.train_group([batch])
            metrics = evaluate_dataset(
                model, make_loader(val_data, training_cfg, device), device, training
            )
            improved = phase_name == "full" and metrics["total_loss"] < best
            if metrics["total_loss"] < progress_best - training["early_stopping_min_delta"]:
                stale, progress_best = 0, metrics["total_loss"]
            else:
                stale += 1
            best = min(best, metrics["total_loss"]) if phase_name == "full" else math.inf
            global_epoch += 1
            path = save_checkpoint(
                directory,
                model,
                loop.optimizer,
                loop.scheduler,
                loop.scaler,
                inputs=signature,
                config=cfg,
                phase=phase_name,
                epoch=epoch + 1,
                global_epoch=global_epoch,
                update=loop.update,
                best_loss=best,
                progress_loss=progress_best,
                stale=stale,
                improved=improved,
                metrics=metrics,
                microbatch_size=loop.microbatch,
                sampler=dict(
                    next_seed=cfg["seed"] + global_epoch,
                    steps=len(sampler),
                    strategy_per_batch=sampler.strategy_size,
                    replay_per_batch=sampler.replay_size,
                ),
                dataset_hashes=signature["data_hashes"],
                parent_sha256=sha256(initial),
            )
            if improved:
                set_pointer(directory, "best_validation", path)
            if epoch_callback:
                epoch_callback(path)
        saved = None
    winner = checkpoint_path(directory, "best_validation")
    atomic_json(
        directory / "complete.json",
        dict(status="completed", inputs=signature, checkpoint=winner.name, sha256=sha256(winner)),
    )
    return winner


def finetune_strategy(root, cfg):
    root = Path(root)
    settings = cfg["strategy_finetuning"]
    reservations = load_reservations(root, cfg)
    initial = read_json(root / "results" / cfg["run_id"] / "initial_selection.json")
    checkpoint = root / initial["checkpoint"]
    if (
        sha256(checkpoint) != initial["sha256"]
        or not (checkpoint.parent / "complete.json").exists()
    ):
        raise ValueError("Complete broad supervised notebook 02 and retain its original checkpoint")
    broad_path = root / "datasets/manifests" / (cfg["run_id"] + ".json")
    broad = read_json(broad_path)
    audit_base(root, cfg, checkpoint, broad)
    replay = list(read_jsonl(root / broad["paths"]["train"]))
    if not replay or any(r["split"] != "train" for r in replay):
        raise ValueError("Replay must contain only the broad training split")
    selected = {s: [] for s in ("train", "validation", "test")}
    paths = settings["manifests"] or [
        f"data/strategy/{cfg['strategy_dataset']['dataset_id']}/manifest.json"
    ]
    data_hashes = {str(broad_path.relative_to(root)): sha256(broad_path)}
    seen, groups = {}, {}
    for path in paths:
        manifest = load_strategy_manifest(root, path)
        if manifest["signature"].get("reservations") != evidence(root, cfg):
            raise ValueError("Strategy dataset lacks matching reservation evidence")
        data_hashes[str(path)] = sha256(root / path)
        for split, relative in manifest["paths"].items():
            for row in read_jsonl(root / relative):
                key, group = identity(row["fen"]), row["source_game_id"] or row["source_id"]
                if (
                    row["split"] != split
                    or seen.get(key, split) != split
                    or groups.get(group, split) != split
                ):
                    raise ValueError("Cross-manifest partition leakage")
                seen[key], groups[group] = split, split
                if row["policy_target"] or row["value_target"] is not None:
                    selected[split].append(row)
    assert_independent(selected["train"] + replay, reservations)
    heldout = {
        identity(r["fen"]) for s in ("val", "test") for r in read_jsonl(root / broad["paths"][s])
    }
    heldout_groups = {
        r["game_id"] for s in ("val", "test") for r in read_jsonl(root / broad["paths"][s])
    }
    selected["train"] = [
        r
        for r in selected["train"]
        if identity(r["fen"]) not in heldout
        and (r["source_game_id"] or r["source_id"]) not in heldout_groups
    ]
    candidates = [dict(id=settings["combined_id"], themes=[], kind="combined")] + [
        dict(c, kind="specialist") for c in settings["specialists"]
    ]
    winners = {}
    for candidate in candidates:
        safe_id(candidate["id"])
        chosen = {
            s: [r for r in rows if not candidate["themes"] or r["theme"] in candidate["themes"]]
            for s, rows in selected.items()
        }
        if not chosen["train"] or not chosen["validation"]:
            raise ValueError(
                f"{candidate['id']} requires real labelled independent train and validation data"
            )
        directory = root / "models/strategy" / cfg["run_id"] / candidate["id"]
        signature = dict(
            config=cfg,
            candidate=candidate,
            parent=initial,
            data_hashes=data_hashes,
            reservations=evidence(root, cfg),
            broad_hashes=broad["hashes"],
            candidate_hashes={},
        )
        for split, rows in chosen.items():
            path = directory / "data" / (split + ".jsonl")
            if path.exists():
                if list(read_jsonl(path)) != rows:
                    raise ValueError("Candidate training rows changed")
            else:
                write_jsonl(path, rows)
            signature["candidate_hashes"][str(path.relative_to(root))] = sha256(path)
        winner = train_candidate(
            root,
            cfg,
            checkpoint,
            directory,
            chosen["train"],
            replay,
            chosen["validation"],
            signature,
        )
        record = dict(
            checkpoint=str(winner.relative_to(root)),
            sha256=sha256(winner),
            config=deepcopy(cfg),
            themes=candidate["themes"],
            kind=candidate["kind"],
            inputs=signature,
        )
        atomic_json(directory / "specialist.json", record)
        winners[candidate["id"]] = record
    atomic_json(
        root / "models/strategy" / cfg["run_id"] / "completed.json", dict(candidates=winners)
    )
    return winners
