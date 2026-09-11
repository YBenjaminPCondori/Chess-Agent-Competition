"""Call only from notebook 06b after winner selection and held-out evaluation."""

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
import torch
from .board_encoding import encode_board
from .checkpoints import load_model
from .dataset import read_jsonl
from .evaluation import harness_modules, run_matchup
from .reproducibility import atomic_json, read_json, sha256, metadata
from .runtime_loader import reconstruct

RUNTIME_FILES = (
    "__init__.py",
    "board_encoding.py",
    "action_encoding.py",
    "model.py",
    "piece_tables.py",
    "classical_evaluation.py",
    "environment.py",
    "time_management.py",
    "transposition.py",
    "inference.py",
    "search.py",
    "runtime_loader.py",
)


def freeze_requirement(root, run_id):
    selection = read_json(Path(root) / "results" / run_id / "final_selection.json")
    if (
        selection["status"] != "frozen"
        or (selection.get("workflow") != "strategy_research" and not selection["training_complete"])
        or not selection["evaluation_complete"]
    ):
        raise RuntimeError("Complete training, evaluation, and winner selection through 06a first")
    if selection.get("workflow") == "strategy_research":
        from .non_rl import source_hashes

        if selection["sources"] != source_hashes(root):
            raise ValueError("Source changed after winner selection")
        from .candidate_registry import load_candidate_registry

        registry = load_candidate_registry(root, selection["workflow_config"])
        if any(selection["stages"][name] != state for name, state in registry["stages"].items()):
            raise ValueError("Frozen training stage evidence changed")
        for name, digest in selection["opening_hashes"].items():
            if sha256(Path(root) / name) != digest:
                raise ValueError("Opening suite changed after selection")
        for relative, digest in selection["evaluation_hashes"].items():
            if sha256(Path(root) / relative) != digest:
                raise ValueError("Evaluation summary changed after selection")
        lock = read_json(Path(root) / "results" / run_id / "winner_lock.json")
        expected = {key: selection[key] for key in lock}
        expected["evaluation_complete"] = False
        if expected != lock:
            raise ValueError("Frozen winner config or evidence changed")
    if selection["checkpoint"] is None:
        return selection, None
    checkpoint = Path(root) / selection["checkpoint"]
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after model selection")
    return selection, checkpoint


def export_selected(root, run_id):
    root = Path(root).resolve()
    selection, checkpoint = freeze_requirement(root, run_id)
    if checkpoint is None:
        raise ValueError("Use final_selected_checks for a classical winner")
    cfg = selection["config"]
    model_version = selection["checkpoint_sha256"][:16]
    directory = root / "exports" / model_version
    candidate = root / "submission_candidate"
    if candidate.exists() and any(candidate.iterdir()):
        prior = candidate / "model_manifest.json"
        if (
            not prior.exists()
            or read_json(prior)["checkpoint_sha256"] != selection["checkpoint_sha256"]
        ):
            raise RuntimeError(
                "submission_candidate contains another model; archive it before finalizing this one"
            )
    directory.mkdir(parents=True, exist_ok=True)
    (candidate / "chess_runtime").mkdir(parents=True, exist_ok=True)
    (candidate / "weights").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source_model, payload = load_model(checkpoint, "cpu")
    source_model.eval()
    model = source_model
    quantization = cfg["export"]["quantization"]
    export_format = cfg["export"]["format"]
    if quantization == "dynamic_int8":
        if export_format != "pytorch_state_dict":
            raise ValueError("INT8 and ONNX are separate final export experiments")
        model = torch.ao.quantization.quantize_dynamic(
            source_model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif quantization != "none":
        raise ValueError("Unsupported quantization")
    if export_format == "pytorch_state_dict":
        torch.save(model.state_dict(), directory / "best_model_cpu.pt")
        shutil.copy2(directory / "best_model_cpu.pt", candidate / "weights/best_model_cpu.pt")
    elif export_format == "onnx":
        # Optional dependencies are installed only by the final notebook's ONNX cell.
        torch.onnx.export(
            model,
            torch.zeros(1, 21, 8, 8),
            str(directory / "best_model.onnx"),
            input_names=["board"],
            output_names=["policy", "value"],
            dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
            opset_version=cfg["export"]["onnx_opset"],
            dynamo=False,
        )
        shutil.copy2(directory / "best_model.onnx", candidate / "weights/best_model.onnx")
    else:
        raise ValueError("Unsupported export format")
    runtime_config = dict(
        architecture=payload["architecture"],
        search=cfg["search"],
        format=export_format,
        quantization=quantization,
        model_version=model_version,
        board_encoder_version="board_v1",
        action_encoder_version="action_v1",
    )
    for path in (directory / "config.json", candidate / "config.json"):
        atomic_json(path, runtime_config)
    for name in RUNTIME_FILES:
        shutil.copy2(root / "chess_rl" / name, candidate / "chess_runtime" / name)
    shutil.copy2(root / "templates/submission_agent.py", candidate / "agent.py")
    manifest = dict(
        checkpoint_sha256=selection["checkpoint_sha256"],
        checkpoint=selection["checkpoint"],
        architecture=payload["architecture"],
        config=cfg,
        software=metadata(),
        encoder_versions=["board_v1", "action_v1"],
        initialization="random, trained by this project; provenance requires human review",
        provenance_review="not_verified",
        runtime_sources={name: sha256(root / "chess_rl" / name) for name in RUNTIME_FILES},
        reference_manifest=read_json(root / "docs/source_manifest.json"),
    )
    for path in (directory / "model_manifest.json", candidate / "model_manifest.json"):
        atomic_json(path, manifest)
    return candidate, directory, source_model, runtime_config


def final_checks(root, run_id, live_games=16):
    root = Path(root).resolve()
    selection, _ = freeze_requirement(root, run_id)
    report_path = root / "results" / run_id / "final_compliance.json"
    report = dict(status="running", checks={}, platform_acceptance="not_verified")
    atomic_json(report_path, report)
    try:
        candidate, directory, source_model, runtime_config = export_selected(root, run_id)
        exported = reconstruct(candidate, runtime_config)
        import chess

        positions = [
            chess.Board(row["fen"])
            for row in list(read_jsonl(root / "datasets/openings/development.jsonl"))[:32]
        ]
        tensor = torch.stack([encode_board(board) for board in positions])
        with torch.inference_mode():
            reference_outputs = source_model(tensor)
            actual_outputs = exported(tensor)
        tolerance = 0.15 if runtime_config["quantization"] != "none" else 1e-4
        for expected, actual in zip(reference_outputs, actual_outputs):
            torch.testing.assert_close(expected, actual, atol=tolerance, rtol=tolerance)
        report["checks"]["output_round_trip"] = "pass"
        elapsed = []
        with torch.inference_mode():
            for index in range(220):
                board = positions[index % len(positions)]
                started = time.perf_counter()
                exported(encode_board(board).unsqueeze(0))
                milliseconds = (time.perf_counter() - started) * 1000
                if index >= 20:
                    elapsed.append(milliseconds)
        report["inference_ms"] = dict(
            median=float(np.median(elapsed)),
            p95=float(np.quantile(elapsed, 0.95)),
            maximum=max(elapsed),
            warmup=20,
            measured=len(elapsed),
        )
        report["inference_target_ms"] = selection["config"]["evaluation"]["target_inference_p95_ms"]
        report["checks"]["cpu_inference"] = "pass"
        # Inspect source imports without confusing optional runtime adapters with training imports.
        allowed = set(sys.stdlib_module_names) | {"torch", "chess", "chess_runtime", "onnxruntime"}
        for path in candidate.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [n.name.split(".")[0] for n in node.names]
                    if isinstance(node, ast.Import)
                    else (
                        [node.module.split(".")[0]]
                        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                        else []
                    )
                )
                if set(names) - allowed:
                    raise ValueError(f"Unapproved imports in {path}: {set(names) - allowed}")
        report["checks"]["runtime_imports"] = "pass"
        harness_modules(root)
        from harness.package import build

        archive = directory / "submission.zip"
        build(
            candidate, archive, ("weights", "chess_runtime", "config.json", "model_manifest.json")
        )
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if bundle.testzip() is not None:
                raise ValueError("Archive integrity check failed")
            if "agent.py" not in bundle.namelist():
                raise ValueError("Missing agent.py at ZIP root")
            total = sum(item.file_size for item in entries)
            if total > 50000000:
                raise ValueError("Uncompressed submission exceeds 50,000,000 bytes")
            unexpected = [
                i.filename
                for i in entries
                if Path(i.filename).suffix not in {".py", ".json", ".pt", ".onnx"}
            ]
            if unexpected:
                raise ValueError(f"Unexpected packaged files: {unexpected}")
            with tempfile.TemporaryDirectory(prefix="chess-final-") as temporary:
                bundle.extractall(temporary)
                checker = root / "scripts/check_candidate_runtime.py"
                result = subprocess.run(
                    [sys.executable, "-I", str(checker), temporary],
                    text=True,
                    capture_output=True,
                    timeout=90,
                    cwd=temporary,
                )
                if result.returncode:
                    raise RuntimeError(result.stderr or result.stdout)
                report["isolated_import_output"] = result.stdout[-4096:]
                report["runtime_observations"] = json.loads(result.stdout.strip().splitlines()[-1])
                weights = (
                    Path(temporary)
                    / "weights"
                    / (
                        "best_model.onnx"
                        if runtime_config["format"] == "onnx"
                        else "best_model_cpu.pt"
                    )
                )
                weights.rename(weights.with_suffix(".missing"))
                fallback = subprocess.run(
                    [sys.executable, "-I", str(checker), temporary, "--fallback"],
                    text=True,
                    capture_output=True,
                    timeout=90,
                    cwd=temporary,
                )
                if fallback.returncode:
                    raise RuntimeError(fallback.stderr or fallback.stdout)
                report["fallback_observations"] = json.loads(
                    fallback.stdout.strip().splitlines()[-1]
                )
        report["checks"]["archive_and_cpu_import"] = "pass"
        for name in ("get_move_api", "legal_moves", "low_clock_timing", "zip_root_agent"):
            report["checks"][name] = "pass"
        report["archive"] = str(archive)
        report["archive_sha256"] = sha256(archive)
        report["unzipped_bytes"] = total
        report["checks"]["provenance_human_review"] = "not_verified"
        report["checks"]["platform_resource_isolation"] = "not_verified"
        report["checks"]["network_and_filesystem_enforcement"] = "not_verified"
        # These are actual games, after training, not a substitute for the platform's validation.
        cfg = selection["config"]
        report["live_results"] = run_matchup(
            root,
            candidate,
            root / "reference/starter/baselines/greedy",
            root / "datasets/openings/development.jsonl",
            cfg,
            f"final-live-{selection['checkpoint_sha256'][:12]}",
            games=live_games,
        )
        if report["live_results"]["candidate_failures"] or report["live_results"]["model_errors"]:
            raise RuntimeError("Candidate failed during final live games")
        report["checks"]["live_games"] = "pass"
        edge_openings = [
            dict(
                fen="7k/P7/8/8/8/8/8/7K w - - 0 300",
                opening_id="white-promotion",
                family_id="white-promotion",
            ),
            dict(
                fen="7k/8/8/8/8/8/p7/7K b - - 0 300",
                opening_id="black-promotion",
                family_id="black-promotion",
            ),
            dict(
                fen="k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 300",
                opening_id="pinned-en-passant",
                family_id="pinned-en-passant",
            ),
            dict(
                fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 300",
                opening_id="castling-rights",
                family_id="castling-rights",
            ),
        ]
        report["edge_game_results"] = run_matchup(
            root,
            candidate,
            root / "reference/starter/baselines/greedy",
            edge_openings,
            cfg,
            f"final-edges-{selection['checkpoint_sha256'][:12]}",
            games=8,
        )
        if report["edge_game_results"]["candidate_failures"]:
            raise RuntimeError("Candidate failed final edge-position games")
        report["checks"]["edge_games"] = "pass"
        report["status"] = "local_checks_passed_with_unverified_platform_constraints"
    except Exception as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        atomic_json(report_path, report)
        raise
    atomic_json(report_path, report)
    return report


def final_selected_checks(root, run_id, live_games=16):
    """06b dispatches the frozen winner through its existing final export implementation."""
    selection, checkpoint = freeze_requirement(root, run_id)
    if checkpoint is not None:
        return final_checks(root, run_id, live_games)
    from .non_rl_export import final_no_rl_checks

    selection = dict(selection, workflow="no_rl", variant="classical")
    report = final_no_rl_checks(root, run_id, selection=selection)
    atomic_json(Path(root) / "results" / run_id / "final_compliance.json", report)
    return report
