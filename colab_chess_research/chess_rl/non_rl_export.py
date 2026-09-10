"""Final checks for no-RL notebook 05 only; no dependency on a league checkpoint."""

import ast
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from .evaluation import harness_modules, run_matchup
from .non_rl import CLASSICAL_FILES, selected_no_rl
from .reproducibility import atomic_json, read_json, sha256


def final_no_rl_checks(root, run_id):
    root = Path(root).resolve()
    selection = selected_no_rl(root, run_id)
    variant, cfg = selection["variant"], selection["config"]
    directory = root / "exports/no_rl" / run_id / variant
    candidate = directory / "candidate"
    report_path = root / "results" / run_id / "no_rl/final_compliance.json"
    report = dict(
        status="running",
        variant=variant,
        checks={},
        platform_acceptance="not_verified",
        platform_resource_isolation="not_verified",
    )
    atomic_json(report_path, report)
    try:
        if candidate.exists() and any(candidate.iterdir()):
            prior = candidate / "selection.json"
            if not prior.exists() or read_json(prior) != selection:
                raise ValueError("Candidate directory contains different work; use a new run_id")
        (candidate / "chess_runtime").mkdir(parents=True, exist_ok=True)
        atomic_json(candidate / "selection.json", selection)
        runtime_config = dict(search=cfg["search"], model_version=f"{run_id}-{variant}")
        files = CLASSICAL_FILES
        template = "classical_agent.py"
        if variant == "supervised":
            import torch
            from .checkpoints import load_model
            from .export import RUNTIME_FILES

            torch.set_num_threads(1)
            model, payload = load_model(root / selection["checkpoint"], "cpu")
            (candidate / "weights").mkdir(exist_ok=True)
            torch.save(model.state_dict(), candidate / "weights/best_model_cpu.pt")
            runtime_config.update(
                architecture=payload["architecture"],
                format="pytorch_state_dict",
                quantization="none",
                board_encoder_version="board_v1",
                action_encoder_version="action_v1",
            )
            files, template = RUNTIME_FILES, "submission_agent.py"
        for name in files:
            shutil.copy2(root / "chess_rl" / name, candidate / "chess_runtime" / name)
        shutil.copy2(root / "templates" / template, candidate / "agent.py")
        atomic_json(candidate / "config.json", runtime_config)
        atomic_json(
            candidate / "model_manifest.json",
            dict(
                workflow="no_rl",
                variant=variant,
                checkpoint_sha256=selection["checkpoint_sha256"],
                runtime_sources={name: sha256(root / "chess_rl" / name) for name in files},
                reference=read_json(root / "docs/source_manifest.json"),
                provenance_review="not_applicable"
                if variant == "classical"
                else "human_review_required",
            ),
        )
        allowed = set(sys.stdlib_module_names) | {"chess", "chess_runtime"}
        if variant == "supervised":
            allowed |= {"torch", "onnxruntime"}  # Optional ONNX loader is inactive in this path.
        for path in candidate.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else (
                        [node.module] if isinstance(node, ast.ImportFrom) and not node.level else []
                    )
                )
                if any(name and name.split(".")[0] not in allowed for name in names):
                    raise ValueError(f"Unexpected runtime import in {path.name}: {names}")
        report["checks"]["runtime_imports"] = "pass"
        harness_modules(root)
        package = importlib.import_module("harness.package")
        rules = importlib.import_module("harness.rules")
        archive_path = directory / "submission.zip"
        package.build(
            candidate,
            archive_path,
            (
                "chess_runtime",
                "weights",
                "config.json",
                "model_manifest.json",
            ),
        )
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            assert "agent.py" in archive.namelist()
            report["uncompressed_bytes"] = sum(member.file_size for member in members)
            if report["uncompressed_bytes"] > rules.MAX_UNZIPPED_BYTES:
                raise ValueError("Submission exceeds the reference size limit")
            if variant == "classical" and any(m.filename.startswith("weights/") for m in members):
                raise ValueError("Classical submission unexpectedly contains weights")
            assert archive.testzip() is None
            report["checks"]["archive_root_size_integrity"] = "pass"
            with tempfile.TemporaryDirectory(prefix="verify-", dir=directory) as temporary:
                extracted = Path(temporary)
                archive.extractall(extracted)
                script = (
                    "check_classical_runtime.py"
                    if variant == "classical"
                    else "check_candidate_runtime.py"
                )
                command = [sys.executable, str(root / "scripts" / script), str(extracted)]
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=90
                )
                report["runtime"] = json.loads(result.stdout.splitlines()[-1])
                if (
                    report["runtime"]["init_ms"] >= 90000
                    or report["runtime"]["peak_rss_mib"] >= 2048
                ):
                    raise ValueError("Local init time or peak RSS exceeds the recorded limit")
                if variant == "supervised":
                    from .board_encoding import encode_board
                    from .runtime_loader import reconstruct
                    import chess

                    reloaded = reconstruct(extracted, runtime_config)
                    boards = [chess.Board(), chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")]
                    batch = torch.stack([encode_board(board) for board in boards])
                    with torch.inference_mode():
                        for expected, actual in zip(model(batch), reloaded(batch)):
                            torch.testing.assert_close(expected, actual)
                    report["checks"]["supervised_weight_round_trip"] = "pass"
                    weight = extracted / "weights/best_model_cpu.pt"
                    backup = weight.with_suffix(".backup")
                    weight.rename(backup)
                    try:
                        subprocess.run(
                            command + ["--fallback"],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=90,
                        )
                    finally:
                        backup.rename(weight)
                    report["checks"]["missing_model_fallback"] = "pass"
                report["checks"]["fresh_process_legal_moves"] = "pass"
        games = run_matchup(
            root,
            candidate,
            root / "reference/classical_agent",
            root / "datasets/openings/development.jsonl",
            cfg,
            f"no-rl-final-{variant}",
            cfg["non_rl"]["final_games"],
        )
        report["final_games"] = games
        if games["candidate_failures"] or games["void_games"] or games.get("model_errors", 0):
            raise RuntimeError("Final games recorded an agent or infrastructure failure")
        report.update(
            status="local_checks_passed", submission=str(archive_path), sha256=sha256(archive_path)
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(report_path, report)
    return report
