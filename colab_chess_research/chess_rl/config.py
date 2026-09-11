from copy import deepcopy
from pathlib import Path
import re
import yaml

COLAB_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess"
)


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(root, override=None):
    root = Path(root)
    config = yaml.safe_load((root / "configs/default.yaml").read_text())
    if override:
        config = merge(config, yaml.safe_load((root / "configs" / override).read_text()) or {})
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["run_id"]):
        raise ValueError("run_id must be a single filename-safe identifier")
    for field in ("policy_top_k_root", "policy_top_k_internal"):
        if config["search"][field] != "all":
            raise ValueError("Candidate exclusion is not implemented in this baseline")
    return config


def prepare_directories(root):
    root = Path(root)
    for path in (
        "datasets/raw",
        "datasets/processed",
        "datasets/openings",
        "datasets/manifests",
        "checkpoints/supervised",
        "checkpoints/self_play",
        "data/strategy_sources",
        "data/strategy",
        "models/strategy",
        "results/strategy_evaluation",
        "logs",
        "results",
        "exports",
    ):
        (root / path).mkdir(parents=True, exist_ok=True)
    return root
