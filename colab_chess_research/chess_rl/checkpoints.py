"""Trusted training checkpoints; tensor-only deployment export is a separate final step."""

from pathlib import Path
import uuid
import torch
from .board_encoding import VERSION as BOARD_VERSION
from .action_encoding import VERSION as ACTION_VERSION
from .model import ChessPolicyValueNetSmall
from .reproducibility import atomic_json, read_json, sha256, capture_rng, metadata


def save_checkpoint(directory, model, optimizer=None, scheduler=None, scaler=None, **state):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(
        model_state_dict=model.state_dict(),
        architecture=model.architecture,
        optimizer_state_dict=optimizer.state_dict() if optimizer else None,
        scheduler_state_dict=scheduler.state_dict() if scheduler else None,
        scaler_state_dict=scaler.state_dict() if scaler else None,
        rng=capture_rng(),
        board_encoder_version=BOARD_VERSION,
        action_encoder_version=ACTION_VERSION,
        metadata=metadata(),
        parameter_count=sum(p.numel() for p in model.parameters()),
        **state,
    )
    name = f"epoch-{state.get('epoch', 0):04d}-update-{state.get('update', 0):07d}-{uuid.uuid4().hex[:8]}.pt"
    path = directory / name
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    digest = sha256(temporary)
    temporary.replace(path)
    if sha256(path) != digest:
        raise OSError("Checkpoint checksum mismatch")
    reference = dict(path=name, sha256=digest)
    atomic_json(directory / "latest.json", reference)
    return path


def checkpoint_path(directory, role="latest"):
    directory = Path(directory)
    pointer = read_json(directory / (role + ".json"))
    path = directory / pointer["path"]
    if sha256(path) != pointer["sha256"]:
        raise ValueError("Checkpoint checksum does not match pointer")
    return path


def set_pointer(directory, role, path):
    directory, path = Path(directory), Path(path)
    if path.parent.resolve() != directory.resolve():
        raise ValueError("Checkpoint role pointers must reference their own run directory")
    atomic_json(directory / (role + ".json"), dict(path=path.name, sha256=sha256(path)))


def load_checkpoint(path):
    # Full RNG/optimizer checkpoints contain Python objects. Only load this project's trusted files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["board_encoder_version"] != BOARD_VERSION
        or payload["action_encoder_version"] != ACTION_VERSION
    ):
        raise ValueError("Checkpoint encoder mismatch")
    return payload


def load_model(path, device="cpu"):
    payload = load_checkpoint(path)
    model = ChessPolicyValueNetSmall(**payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload
