"""Supervised and replay optimization, with legal losses and resumable optimizer state."""

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .checkpoints import save_checkpoint, checkpoint_path, set_pointer, load_checkpoint
from .dataset import PositionDataset
from .model import ChessPolicyValueNetSmall
from .reproducibility import (
    seed_all,
    worker_seed,
    restore_rng,
    capture_rng,
    atomic_json,
    append_csv,
    resolve_device,
    metadata,
)


def loss_components(logits, values, batch, smoothing=0.0, huber_delta=1.0):
    mask = batch["mask"]
    targets = batch["policy"]
    policy_valid = batch["policy_valid"]
    value_valid = batch["value_valid"]
    if bool((targets.masked_select(~mask) != 0).any()):
        raise ValueError("Policy target assigns mass to illegal actions")
    if bool((policy_valid & ~mask.any(dim=1)).any()):
        raise ValueError("Terminal position cannot have a policy target")
    safe_mask = mask.clone()
    safe_mask[~safe_mask.any(dim=1), 0] = True
    log_probs = F.log_softmax(logits.float().masked_fill(~safe_mask, -torch.inf), dim=1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    smooth = mask.float() / mask.sum(dim=1, keepdim=True).clamp_min(1)
    mixed = (1 - smoothing) * targets + smoothing * smooth
    policy_each = -(mixed * log_probs).sum(dim=1)
    value_each = F.huber_loss(
        values.float(), batch["value"].float(), reduction="none", delta=huber_delta
    )
    policy_sum = (policy_each * policy_valid).sum()
    value_sum = (value_each * value_valid).sum()
    return policy_sum, value_sum


class OptimizerLoop:
    def __init__(self, model, settings, device, total_steps=1, constant_lr=False):
        self.model = model.to(device)
        self.settings, self.device = settings, device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=settings["weight_decay"],
        )
        self.scheduler = None
        if not constant_lr:
            if settings["scheduler"] == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, total_steps),
                    eta_min=settings["learning_rate"] * 0.01,
                )
            elif settings["scheduler"] == "onecycle":
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=settings["learning_rate"],
                    total_steps=total_steps,
                    pct_start=0.1,
                    div_factor=10,
                    final_div_factor=100,
                )
            else:
                raise ValueError("Unsupported scheduler")
        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.use_amp = device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self.microbatch = (
            settings.get("microbatch_size")
            or settings["batch_size_cuda" if self.use_amp else "batch_size_cpu"]
        )
        self.update = 0
        self.frozen_modules = []

    def autocast(self):
        return torch.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()

    def train_group(self, batches):
        policy_count = sum(int(b["policy_valid"].sum()) for b in batches)
        value_count = sum(int(b["value_valid"].sum()) for b in batches)
        buffers = {name: value.detach().clone() for name, value in self.model.named_buffers()}
        rng = capture_rng()
        overflow_retries = 0
        while True:
            step_started = False
            self.optimizer.zero_grad(set_to_none=True)
            totals = [0.0, 0.0]
            self.model.train()
            for module in self.frozen_modules:
                module.eval()
            try:
                for original in batches:
                    for start in range(0, len(original["board"]), self.microbatch):
                        batch = {
                            key: value[start : start + self.microbatch].to(self.device)
                            for key, value in original.items()
                            if isinstance(value, torch.Tensor)
                        }
                        with self.autocast():
                            logits, values = self.model(batch["board"])
                            ps, vs = loss_components(
                                logits,
                                values,
                                batch,
                                self.settings["policy_label_smoothing"],
                                self.settings["huber_delta"],
                            )
                            loss = ps / max(1, policy_count) + self.settings[
                                "value_weight"
                            ] * vs / max(1, value_count)
                        if not torch.isfinite(loss):
                            raise ValueError("Training loss is not finite")
                        self.scaler.scale(loss).backward()
                        totals[0] += float(ps.detach())
                        totals[1] += float(vs.detach())
                        del logits, values, loss, batch, ps, vs
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.settings["gradient_clip_norm"],
                    error_if_nonfinite=not self.scaler.is_enabled(),
                )
                step_started = True
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < previous_scale:
                    overflow_retries += 1
                    if overflow_retries > 20:
                        raise RuntimeError(
                            "Repeated FP16 gradient overflow; use BF16 or full precision"
                        )
                    with torch.no_grad():
                        for name, value in self.model.named_buffers():
                            value.copy_(buffers[name])
                    restore_rng(rng)
                    print(
                        "FP16 overflow: reduced scale and retrying the optimizer update.",
                        flush=True,
                    )
                    continue
                if self.scheduler:
                    self.scheduler.step()
                self.update += 1
                return dict(
                    policy_loss=totals[0] / max(1, policy_count),
                    value_loss=totals[1] / max(1, value_count),
                    microbatch_size=self.microbatch,
                )
            except torch.cuda.OutOfMemoryError:
                self.optimizer.zero_grad(set_to_none=True)
                if step_started or not self.use_amp or self.microbatch <= 1:
                    raise
                with torch.no_grad():
                    for name, value in self.model.named_buffers():
                        value.copy_(buffers[name])
                restore_rng(rng)
                if self.scaler.is_enabled():
                    self.scaler.update(new_scale=self.scaler.get_scale())
                self.microbatch = max(1, self.microbatch // 2)
                torch.cuda.empty_cache()
                print(
                    f"CUDA OOM: retrying with microbatch={self.microbatch}; effective batch unchanged.",
                    flush=True,
                )

    def restore(self, payload):
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler and payload["scheduler_state_dict"]:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload["scaler_state_dict"]:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.update = payload["update"]
        self.microbatch = payload.get("microbatch_size", self.microbatch)
        restore_rng(payload["rng"])


def evaluate_dataset(model, loader, device, settings):
    model.eval()
    sums = dict(
        policy_loss=0.0,
        value_loss=0.0,
        policy_count=0,
        value_count=0,
        top1=0,
        top3=0,
        value_absolute_error=0.0,
    )
    sources = {}
    with torch.inference_mode():
        for batch in loader:
            labels = batch["source"]
            moved = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            logits, values = model(moved["board"])
            # Always compare unsmoothed validation loss across smoothing trials.
            ps, vs = loss_components(logits, values, moved, 0.0, settings["huber_delta"])
            sums["policy_loss"] += float(ps)
            sums["value_loss"] += float(vs)
            pvalid, vvalid = moved["policy_valid"], moved["value_valid"]
            sums["policy_count"] += int(pvalid.sum())
            sums["value_count"] += int(vvalid.sum())
            legal_logits = logits.masked_fill(~moved["mask"], -torch.inf)
            target = moved["policy"].argmax(dim=1)
            top = legal_logits.topk(3, dim=1).indices
            sums["top1"] += int(((top[:, 0] == target) & pvalid).sum())
            sums["top3"] += int(((top == target[:, None]).any(dim=1) & pvalid).sum())
            errors = (values - moved["value"]).abs()
            sums["value_absolute_error"] += float((errors * vvalid).sum())
            for label, error, valid in zip(labels, errors.cpu(), vvalid.cpu()):
                if valid:
                    stat = sources.setdefault(label, [0.0, 0])
                    stat[0] += float(error)
                    stat[1] += 1
    pc, vc = max(1, sums["policy_count"]), max(1, sums["value_count"])
    return dict(
        policy_loss=sums["policy_loss"] / pc,
        value_loss=sums["value_loss"] / vc,
        total_loss=sums["policy_loss"] / pc + settings["value_weight"] * sums["value_loss"] / vc,
        top1=sums["top1"] / pc,
        top3=sums["top3"] / pc,
        value_mae=sums["value_absolute_error"] / vc,
        source_mae={key: value[0] / value[1] for key, value in sources.items()},
    )


def make_loader(dataset, cfg, device, epoch=0, shuffle=False):
    training = cfg["training"]
    generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
    return DataLoader(
        dataset,
        batch_size=training["batch_size_cuda" if device.startswith("cuda") else "batch_size_cpu"],
        shuffle=shuffle,
        num_workers=training["num_workers"],
        pin_memory=device.startswith("cuda"),
        worker_init_fn=worker_seed,
        generator=generator,
    )


def fit_supervised(
    root,
    cfg,
    manifest,
    *,
    initial_checkpoint=None,
    checkpoint_directory=None,
    log_name="supervised",
    freeze_prefixes=(),
    unfreeze_epoch=None,
):
    from .reproducibility import sha256
    from .checkpoints import load_model
    from .reservations import enabled, audit_dataset

    root = Path(root)
    if enabled(cfg) and initial_checkpoint is None:
        audit_dataset(root, cfg, manifest)
    directory = (
        Path(checkpoint_directory)
        if checkpoint_directory
        else root / "checkpoints/supervised" / cfg["run_id"]
    )
    transfer = (
        dict(
            initial_sha256=sha256(initial_checkpoint),
            freeze_prefixes=list(freeze_prefixes),
            unfreeze_epoch=unfreeze_epoch,
        )
        if initial_checkpoint
        else None
    )
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / "complete.json"
    if completion.exists():
        selected = checkpoint_path(directory, "best_validation")
        payload = load_checkpoint(selected)
        if (
            payload["config"] != cfg
            or payload["dataset_hashes"] != manifest["hashes"]
            or payload.get("transfer") != transfer
        ):
            raise ValueError("Completed run has different config/data; choose a new run_id")
        return selected
    seed_all(cfg["seed"], cfg.get("deterministic", False))
    device = resolve_device(cfg["device"])
    training = cfg["training"]
    train_data = PositionDataset(root / manifest["paths"]["train"])
    val_data = PositionDataset(root / manifest["paths"]["val"])
    loader = make_loader(train_data, cfg, device, shuffle=True)
    steps_per_epoch = math.ceil(len(loader) / training["gradient_accumulation_steps"])
    model = (
        load_model(initial_checkpoint, device)[0]
        if initial_checkpoint
        else ChessPolicyValueNetSmall(**cfg["model"])
    )
    loop = OptimizerLoop(model, training, device, steps_per_epoch * training["max_epochs"])
    start_epoch, best_loss, stale = 0, math.inf, 0
    best_progress = math.inf
    if (directory / "latest.json").exists():
        payload = load_checkpoint(checkpoint_path(directory))
        if (
            payload["config"] != cfg
            or payload["dataset_hashes"] != manifest["hashes"]
            or payload.get("transfer") != transfer
        ):
            raise ValueError(
                "Resume config/data differs; use a new run_id for a changed experiment"
            )
        loop.restore(payload)
        start_epoch, best_loss, stale = payload["epoch"], payload["best_loss"], payload["stale"]
        best_progress = payload.get("progress_loss", best_loss)
    atomic_json(
        directory / "run_metadata.json",
        dict(
            config=cfg,
            metadata=metadata(),
            device=device,
            parameter_count=sum(p.numel() for p in model.parameters()),
            amp_dtype=str(loop.amp_dtype) if loop.use_amp else "float32",
        ),
    )
    if stale >= training["early_stopping_patience"]:
        atomic_json(completion, dict(status="complete", reason="early_stopping"))
        return checkpoint_path(directory, "best_validation")
    epoch = start_epoch - 1
    for epoch in range(start_epoch, training["max_epochs"]):
        frozen = freeze_prefixes if unfreeze_epoch is None or epoch < unfreeze_epoch else ()
        modules = dict(model.named_modules())
        if any(prefix not in modules for prefix in frozen):
            raise ValueError("Unknown module in freeze_prefixes")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                not any(name == prefix or name.startswith(prefix + ".") for prefix in frozen)
            )
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Fine-tuning must leave at least one trainable parameter")
        loop.frozen_modules = [modules[prefix] for prefix in frozen]
        started = time.monotonic()
        batches, epoch_policy, epoch_value, updates = [], 0.0, 0.0, 0
        for batch in make_loader(train_data, cfg, device, epoch, shuffle=True):
            batches.append(batch)
            if len(batches) == training["gradient_accumulation_steps"]:
                metrics = loop.train_group(batches)
                epoch_policy += metrics["policy_loss"]
                epoch_value += metrics["value_loss"]
                updates += 1
                batches = []
        if batches:
            metrics = loop.train_group(batches)
            epoch_policy += metrics["policy_loss"]
            epoch_value += metrics["value_loss"]
            updates += 1
        validation = evaluate_dataset(model, make_loader(val_data, cfg, device), device, training)
        improved = validation["total_loss"] < best_loss
        if improved:
            best_loss = validation["total_loss"]
        if validation["total_loss"] < best_progress - training["early_stopping_min_delta"]:
            best_progress, stale = validation["total_loss"], 0
        else:
            stale += 1
        path = save_checkpoint(
            directory,
            model,
            loop.optimizer,
            loop.scheduler,
            loop.scaler,
            epoch=epoch + 1,
            update=loop.update,
            config=cfg,
            metrics=validation,
            best_loss=best_loss,
            progress_loss=best_progress,
            stale=stale,
            microbatch_size=loop.microbatch,
            dataset_hashes=manifest["hashes"],
            sampler=dict(next_epoch=epoch + 1, seed=cfg["seed"] + epoch + 1),
            replay_manifest=None,
            league_state=None,
            transfer=transfer,
        )
        if improved or not (directory / "best_validation.json").exists():
            set_pointer(directory, "best_validation", path)
        row = dict(
            epoch=epoch + 1,
            update=loop.update,
            train_policy_loss=epoch_policy / max(1, updates),
            train_value_loss=epoch_value / max(1, updates),
            **validation,
            learning_rate=loop.optimizer.param_groups[0]["lr"],
            microbatch_size=loop.microbatch,
            epoch_seconds=time.monotonic() - started,
        )
        row["source_mae"] = json.dumps(row["source_mae"], sort_keys=True)
        append_csv(root / "logs" / cfg["run_id"] / (log_name + ".csv"), row)
        atomic_json(directory / "metrics.json", row)
        print(
            f"Epoch {epoch + 1}: val_loss={validation['total_loss']:.5f}, "
            f"legal_top1={validation['top1']:.3f}, checkpoint={path.name}",
            flush=True,
        )
        if stale >= training["early_stopping_patience"]:
            break
    atomic_json(
        completion,
        dict(
            status="complete",
            epochs=epoch + 1,
            selected=checkpoint_path(directory, "best_validation").name,
        ),
    )
    return checkpoint_path(directory, "best_validation")
