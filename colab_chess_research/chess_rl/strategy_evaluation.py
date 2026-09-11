"""CPU research suites with explicit label coverage and missing-data states."""

import math
from pathlib import Path
import chess
import torch
from .action_encoding import encode_move
from .board_encoding import encode_board
from .checkpoints import load_model
from .classical_evaluation import evaluate_cp
from .dataset import read_jsonl
from .reproducibility import atomic_json, sha256
from .search import ResearchAgent
from .strategy_dataset import load_strategy_manifest, validate_schema
from .strategy_taxonomy import TAXONOMY


def load_suites(root, settings, run_id):
    split = settings.get("split", "validation")
    if split not in ("validation", "test"):
        raise ValueError("Evaluation requires validation or test data")
    paths = settings.get("manifests") or [f"data/strategy/{run_id}/manifest.json"]
    suites = {theme: [] for theme in TAXONOMY}
    hashes = {}
    for path in paths:
        manifest = load_strategy_manifest(root, path)
        hashes[str(path)] = sha256(Path(root) / path)
        for raw in read_jsonl(Path(root) / manifest["paths"][split]):
            row = validate_schema(raw)
            if row["split"] != split:
                raise ValueError("Suite row is stored under the wrong split")
            suites[row["theme"]].append(row)
    return suites, hashes


class CandidatePredictor:
    def __init__(self, root, candidate, top_k=3, time_left_ms=10000):
        torch.set_num_threads(1)
        self.model = (
            load_model(Path(root) / candidate["checkpoint"], "cpu")[0]
            if candidate["checkpoint"]
            else None
        )
        self.agent = ResearchAgent(
            self.model, candidate["config"]["search"], "cpu", candidate["id"]
        )
        self.evaluation_weights = candidate["config"]["search"].get("evaluation_weights")
        self.top_k, self.time_left_ms = top_k, time_left_ms

    def __call__(self, fen):
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        ranked = None
        if self.model is not None:
            with torch.inference_mode():
                logits, value = self.model(encode_board(board).unsqueeze(0))
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite model predictions")
            ranked = [
                m.uci()
                for m in sorted(
                    legal, key=lambda m: float(logits[0, encode_move(board, m)]), reverse=True
                )[: self.top_k]
            ]
            value = float(value[0])
        else:
            value = math.tanh(evaluate_cp(board, self.evaluation_weights) / 600)
        before = dict(self.agent.engine.cumulative)
        move = self.agent.get_move(fen, self.time_left_ms) if legal else None
        if any(
            self.agent.engine.cumulative.get(key, 0) > before.get(key, 0)
            for key in ("model_errors", "search_errors")
        ):
            raise RuntimeError("Search used an error fallback")
        return dict(move=move, ranked=ranked, value=value)


def evaluate_suite(rows, predictor, thresholds, top_k=3):
    details = []
    for raw in rows:
        record = dict(
            fen=raw.get("fen"),
            source_id=raw.get("source_id"),
            theme=raw.get("theme"),
            subtheme=raw.get("subtheme"),
            move=None,
            legal=None,
            correct=None,
            policy_top1_correct=None,
            policy_top3_correct=None,
            top_k_correct=None,
            value_error=None,
            error=None,
            history_dependent=False,
            history_available=False,
            legal_applicable=False,
            status="not_assessable",
            policy_labelled=False,
            value_labelled=False,
        )
        try:
            row = validate_schema(raw)
            record["history_available"] = row["history_moves"] is not None
            record["history_dependent"] = bool(
                row.get("history_dependent")
                or "repetition" in (row["subtheme"] or "")
                or (row.get("features") or {}).get("threefold_repetition")
            )
            legal = row["legal_moves"]
            record["legal_applicable"] = bool(legal)
            targets = set(row.get("acceptable_moves") or [])
            if not targets and row["best_move"]:
                targets = {row["best_move"]}
            if not targets and row["policy_target"]:
                best = max(row["policy_target"].values())
                targets = {m for m, p in row["policy_target"].items() if p == best}
            record["policy_labelled"] = bool(targets)
            record["value_labelled"] = row["value_target"] is not None
            predicted = predictor(row["fen"])
            move, ranked = predicted.get("move"), predicted.get("ranked")
            record.update(move=move, legal=(move in legal) if legal else move is None)
            if not record["legal"] or ranked is not None and any(m not in legal for m in ranked):
                raise ValueError("Illegal predicted move or policy ranking")
            value = predicted.get("value")
            if value is not None and (
                not math.isfinite(float(value)) or not -1 <= float(value) <= 1
            ):
                raise ValueError("Invalid evaluation value")
            # History-only labels cannot assess the deployed FEN-only agent.
            if not record["history_dependent"]:
                if targets:
                    record["correct"] = move in targets
                    if ranked is not None:
                        record["policy_top1_correct"] = bool(targets.intersection(ranked[:1]))
                        record["policy_top3_correct"] = bool(targets.intersection(ranked[:3]))
                        record["top_k_correct"] = bool(targets.intersection(ranked[:top_k]))
                if row["value_target"] is not None and value is not None:
                    record["value_error"] = abs(float(value) - row["value_target"])
                assessed = record["correct"] is not None or record["value_error"] is not None
                failed = record["correct"] is False or (
                    record["value_error"] is not None
                    and record["value_error"] > thresholds["max_value_mae"]
                )
                record["status"] = "fail" if failed else "pass" if assessed else "not_assessable"
        except Exception as error:
            record.update(legal=False, error=f"{type(error).__name__}: {error}", status="fail")
            if record["policy_labelled"] and not record["history_dependent"]:
                record["correct"] = False
        details.append(record)

    def mean(key, records=None):
        values = [r[key] for r in (details if records is None else records) if r[key] is not None]
        return sum(values) / len(values) if values else None

    counts = {
        status: sum(r["status"] == status for r in details)
        for status in ("pass", "fail", "not_assessable")
    }
    summary = dict(
        positions=len(rows),
        policy_labelled=sum(r.get("policy_labelled", False) for r in details),
        value_labelled=sum(r.get("value_labelled", False) for r in details),
        accuracy=mean("correct"),
        search_move_agreement=mean("correct"),
        policy_top1_agreement=mean("policy_top1_correct"),
        policy_top3_agreement=mean("policy_top3_correct"),
        top_k_agreement=mean("top_k_correct"),
        value_mae=mean("value_error"),
        value_predictions=sum(r["value_error"] is not None for r in details),
        policy_predictions=sum(r["policy_top1_correct"] is not None for r in details),
        legal_move_rate=mean("legal", [r for r in details if r["legal_applicable"]]),
        terminal_positions=sum(not r["legal_applicable"] for r in details),
        errors=sum(r["error"] is not None for r in details),
        history_positions=sum(r["history_dependent"] for r in details),
        history_counts={
            status: sum(r["history_dependent"] and r["status"] == status for r in details)
            for status in counts
        },
        fen_only_counts={
            status: sum(not r["history_dependent"] and r["status"] == status for r in details)
            for status in counts
        },
        counts=counts,
    )
    failed = bool(summary["errors"])
    for metric, threshold, minimum in (
        ("accuracy", "min_accuracy", True),
        ("value_mae", "max_value_mae", False),
        ("legal_move_rate", "min_legal_rate", True),
    ):
        value = summary[metric]
        if value is not None:
            failed |= value < thresholds[threshold] if minimum else value > thresholds[threshold]
    covered = summary["accuracy"] is not None or summary["value_predictions"] > 0
    summary["status"] = "fail" if failed else "pass" if covered else "not_assessable"
    return summary, details


def evaluate_candidates(root, cfg, candidates=None):
    from copy import deepcopy
    import csv
    from .candidate_registry import load_candidate_registry
    from .research_workflow import validate_candidate, source_hashes
    from .reservations import evidence
    from .reproducibility import read_json

    root = Path(root)
    settings = cfg["strategy_evaluation"]
    registry_cfg = deepcopy(cfg)
    registry_cfg["strategy_evaluation"]["split"] = "validation"
    registry = load_candidate_registry(root, registry_cfg)
    if settings["split"] == "test":
        lock = read_json(root / "results" / cfg["run_id"] / "winner_lock.json")
        selected = [c for c in registry["candidates"] if c["id"] == lock["selected_id"]]
    else:
        selected = registry["candidates"]
    if candidates is not None and candidates != selected:
        raise ValueError("Evaluation candidates differ from registry/frozen winner")
    candidates = selected
    top_k = settings["top_k"]
    if type(top_k) is not int or top_k < 3:
        raise ValueError("top_k must include at least the top three policy moves")
    if (
        not cfg["strategy_dataset"]["sources"]
        and cfg["research_workflow"]["stages"]["strategy"] == "skipped"
    ):
        suites, hashes = {theme: [] for theme in TAXONOMY}, {}
    else:
        suites, hashes = load_suites(root, settings, cfg["strategy_dataset"]["dataset_id"])
        for path in hashes:
            manifest = load_strategy_manifest(root, path)
            if manifest["signature"].get("reservations") != evidence(root, cfg):
                raise ValueError("Strategy suites lack matching pre-training reservations")
    directory = root / "results/strategy_evaluation" / cfg["run_id"] / settings["split"]
    signature = dict(
        candidates=candidates,
        dataset_hashes=hashes,
        settings=settings,
        sources=source_hashes(root),
        registry_sha256=sha256(root / "results" / cfg["run_id"] / "candidate_registry.json"),
    )
    lock = directory / "inputs.json"
    if lock.exists() and read_json(lock) != signature:
        raise ValueError("Immutable strategy evaluation inputs changed")
    atomic_json(lock, signature)
    if (directory / "summary.json").exists():
        record = read_json(directory / "summary.json")
        if any(record[k] != v for k, v in signature.items()) or record[
            "positions_sha256"
        ] != sha256(directory / "positions.json"):
            raise ValueError("Strategy evaluation evidence changed")
        return record
    summaries, details = [], []
    for candidate in candidates:
        validate_candidate(root, candidate)
        try:
            predictor = CandidatePredictor(root, candidate, top_k, settings["time_left_ms"])
        except Exception as error:
            message = f"{type(error).__name__}: {error}"

            def predictor(fen, message=message):
                raise RuntimeError(message)

        for theme, rows in suites.items():
            summary, records = evaluate_suite(rows, predictor, settings["thresholds"], top_k)
            summaries.append(dict(candidate=candidate["id"], theme=theme, **summary))
            details.extend(dict(candidate=candidate["id"], **r) for r in records)
        del predictor
    atomic_json(directory / "positions.json", details)
    record = dict(
        **signature, summaries=summaries, positions_sha256=sha256(directory / "positions.json")
    )
    atomic_json(directory / "summary.json", record)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return record
