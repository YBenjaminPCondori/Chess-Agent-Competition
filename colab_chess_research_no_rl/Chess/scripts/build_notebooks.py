"""Generate five Colab notebooks with nbformat; shared logic lives in chess_rl."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MOUNT = """from google.colab import drive
drive.mount("/content/drive")"""
SETUP = """from pathlib import Path
import sys
import subprocess

PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess")
if not (PROJECT_ROOT / "chess_rl").is_dir():
    raise FileNotFoundError(f"Project files are missing from {PROJECT_ROOT}. See README.md.")
sys.path.insert(0, str(PROJECT_ROOT))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "-r", str(PROJECT_ROOT / "requirements_colab.txt")])"""
CONFIG = """from chess_rl.config import load_config, prepare_directories
from chess_rl.reproducibility import metadata, read_json, atomic_json, sha256

prepare_directories(PROJECT_ROOT)
cfg = load_config(PROJECT_ROOT)
print("Run:", cfg["run_id"])
print("Runtime:", metadata())"""


def notebook(title, intro, sections):
    cells = [
        nbf.v4.new_markdown_cell("# " + title + "\n\n" + intro),
        nbf.v4.new_code_cell(MOUNT),
        nbf.v4.new_markdown_cell(
            "## Project and Dependencies\n\nKeep Colab's existing CUDA PyTorch. Restart only if pip explicitly requires it."
        ),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_markdown_cell(
            "## Run Configuration\n\nEdit configs/default.yaml once for the workflow. Use a new run_id for a changed experiment."
        ),
        nbf.v4.new_code_cell(CONFIG),
    ]
    for heading, markdown, code in sections:
        cells.append(nbf.v4.new_markdown_cell("## " + heading + "\n\n" + markdown))
        if code:
            cells.append(nbf.v4.new_code_cell(code))
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    nb.metadata.language_info = dict(name="python", version="3.12")
    nb.metadata.colab = dict(name=title, provenance=[])
    nbf.validate(nb)
    return nb


books = {}
books["01_chess_environment_and_encoding.ipynb"] = notebook(
    "01 - Chess Environment and Encoding",
    "Define the observable chess task, inspect the encodings, and run ordinary correctness tests. "
    "No trained model or measured playing strength is claimed by this notebook.",
    [
        (
            "Sources and Environment Contract",
            "The reference snapshot is included. The referee retains move history; the agent sees FEN and its remaining clock. The reserved repetition plane is zero in training and runtime.",
            """source = read_json(PROJECT_ROOT / "docs/source_manifest.json")
print("Source date:", source["inspected_at"])
print("Official runtime versions:", source["runtime_versions"])
print((PROJECT_ROOT / "docs/ENVIRONMENT_CONTRACT.md").read_text())""",
        ),
        (
            "A Standard Chess Position",
            "Every piece constraint comes from python-chess. These are legal UCI actions.",
            """import chess
from IPython.display import display
from chess_rl.environment import ChessEnvironment

environment = ChessEnvironment()
fen, remaining_ms = environment.observe()
display(environment.board)
print("FEN:", fen)
print("Remaining milliseconds:", remaining_ms)
print("Legal UCI moves:", [move.uci() for move in environment.board.legal_moves])""",
        ),
        (
            "Board and Action Representation",
            "Absolute White-oriented coordinates are used for both players. The policy index is plane * 64 + from_square.",
            """from chess_rl.board_encoding import encode_board
from chess_rl.action_encoding import encode_move, decode_move, legal_mask

board = environment.board
encoded = encode_board(board)
move = chess.Move.from_uci("e2e4")
index = encode_move(board, move)
print("Input shape:", tuple(encoded.shape))
print("e2e4 index:", index, "| round-trip:", decode_move(board, index).uci())
print("Legal-mask entries:", int(legal_mask(board).sum()))
print("History plane sum:", float(encoded[20].sum()))""",
        ),
        (
            "Correctness Tests",
            "These cover rules, encodings, legal losses, timeout restoration, and checkpoint recovery. They are not submission compliance or long training runs.",
            """subprocess.check_call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=PROJECT_ROOT)""",
        ),
        (
            "Versioned Opening Suites",
            "Reserve independent legal playouts before the corpus. These are valid research fixtures, not certified balanced positions or the hidden platform set. The public eight remain in the reference harness.",
            """from chess_rl.dataset import prepare_openings
opening_manifest = prepare_openings(PROJECT_ROOT, seed=cfg["seed"])
print(opening_manifest)""",
        ),
        (
            "Next Notebook",
            "Open notebook 02. An A100 accelerates gradient updates; Stockfish labels positions on CPU. Every notebook reloads persisted inputs from Drive.",
            None,
        ),
    ],
)

books["02_supervised_policy_value_training.ipynb"] = notebook(
    "02 - Supervised Policy and Value Training",
    "Acquire labelled positions, train the shared network, and persist validation-selected checkpoints. "
    "Plots use completed epochs; no example scores are pre-filled.",
    [
        (
            "Training Device",
            "Select a GPU runtime in Colab. CPU remains an explicit portability option. The default has six residual blocks and 128 channels.",
            """import torch
from chess_rl.reproducibility import resolve_device, seed_all
from chess_rl.model import ChessPolicyValueNetSmall

device = resolve_device(cfg["device"])
seed_all(cfg["seed"], cfg["deterministic"])
model_preview = ChessPolicyValueNetSmall(**cfg["model"])
print("Training device:", device)
print("Parameters:", sum(p.numel() for p in model_preview.parameters()))
print("Architecture:", model_preview.architecture)
del model_preview""",
        ),
        (
            "Offline Teacher Setup",
            "Stockfish labels training data only. Default: 50,000 nodes, one thread, 128 MB hash. It is excluded from packaging. An explicit teacher: classical configuration is also supported.",
            """teacher_path = Path(cfg["dataset"]["engine_path"])
if cfg["dataset"]["teacher"] == "stockfish" and not teacher_path.is_file():
    subprocess.check_call(["apt-get", "update", "-qq"])
    subprocess.check_call(["apt-get", "install", "-y", "-qq", "stockfish"])
if cfg["dataset"]["teacher"] == "stockfish":
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Set dataset.engine_path to installed Stockfish: {teacher_path}")
    print("Teacher executable:", teacher_path)
    print("Teacher hash:", sha256(teacher_path))""",
        ),
        (
            "Prepare the Dataset",
            "Supply PGNs or compatible JSONL through the config. Without a corpus, legal playout positions are generated and labelled. The target is 100,000 unique positions. Saved annotation shards support resuming.",
            """from chess_rl.dataset import prepare_dataset
dataset_manifest = prepare_dataset(PROJECT_ROOT, cfg)
print("Actual split counts:", dataset_manifest["counts"])
print("Requested target reached:", dataset_manifest["target_reached"])
print("Dataset hashes:", dataset_manifest["hashes"])""",
        ),
        (
            "Loss and Value Meaning",
            "Policy loss uses legal moves. Outcome values use the side to move: win +1, draw 0, loss -1. Engine centipawns use tanh(cp/600), identified separately as heuristic labels. Validation loss is unsmoothed for comparable trials.",
            None,
        ),
        (
            "Train or Resume",
            "This runs the configured experiment. Latest resumes completed epochs; best_validation selects lowest validation loss. A partial interrupted epoch restarts from its last complete checkpoint.",
            """from chess_rl.training import fit_supervised
selected_pretraining = fit_supervised(PROJECT_ROOT, cfg, dataset_manifest)
print("Selected supervised checkpoint:", selected_pretraining)""",
        ),
        (
            "Learning Curves",
            "Accuracy is against legal teacher targets. Lower loss does not itself prove stronger chess.",
            """from chess_rl.plots import plot_supervised
display(plot_supervised(PROJECT_ROOT, cfg["run_id"]))""",
        ),
        (
            "Optional Model Search",
            "Disabled by default. Trials have separate folders. Three lowest-loss trials play development matches against the preserved classical agent; their playing scores select the initialization.",
            """RUN_MODEL_SEARCH = False
if RUN_MODEL_SEARCH:
    import yaml
    from chess_rl.tuning import run_model_study
    from chess_rl.evaluation import build_research_agent, run_matchup
    spaces = yaml.safe_load((PROJECT_ROOT / "configs/search_spaces.yaml").read_text())
    shortlist = run_model_study(PROJECT_ROOT, cfg, dataset_manifest, spaces)
    comparisons = []
    for trial in shortlist:
        checkpoint = PROJECT_ROOT / trial["checkpoint"]
        candidate = build_research_agent(PROJECT_ROOT, checkpoint, trial["config"], "shortlist")
        result = run_matchup(PROJECT_ROOT, candidate, PROJECT_ROOT / "reference/classical_agent",
                            PROJECT_ROOT / "datasets/openings/development.jsonl",
                            trial["config"], "shortlist-development")
        comparisons.append((result["score"] if result["score"] is not None else -1, checkpoint))
    if not comparisons:
        raise RuntimeError("No successful model-search trials")
    selected_pretraining = max(comparisons, key=lambda item: item[0])[1]
    print("Shortlist match selection:", selected_pretraining)""",
        ),
        (
            "Persist Initialization Choice",
            "Notebook 03 reads this checkpoint reference. Independent experiments need different run identifiers.",
            """atomic_json(PROJECT_ROOT / "results" / cfg["run_id"] / "initial_selection.json",
            {"checkpoint": str(selected_pretraining.relative_to(PROJECT_ROOT)),
             "sha256": sha256(selected_pretraining)})
print("Notebook 02 complete. Open notebook 03.")""",
        ),
    ],
)

books["03_self_play_league_training.ipynb"] = notebook(
    "03 - Search-Guided Self-Play and League Training",
    "Learn policy targets from completed alpha-beta analyses and values from game outcomes. "
    "Collectors stay frozen during each iteration. CPU matches decide promotion.",
    [
        (
            "Load Prepared Inputs",
            "Notebook 02 must have completed. It does not need to remain open.",
            """initial = read_json(PROJECT_ROOT / "results" / cfg["run_id"] / "initial_selection.json")
initial_checkpoint = PROJECT_ROOT / initial["checkpoint"]
if sha256(initial_checkpoint) != initial["sha256"]:
    raise ValueError("Initialization checkpoint was modified")
dataset_manifest = read_json(PROJECT_ROOT / "datasets/manifests" / (cfg["run_id"] + ".json"))
print("Initialization:", initial_checkpoint)
print("Self-play settings:", cfg["self_play"])""",
        ),
        (
            "Targets and Exploration",
            "The teacher searches every root action at a common completed depth with full windows. Target probabilities use softmax(score/0.25). Exploration changes the played move, not the target. Missing completed analyses have no policy target. Failed games do not become draw labels.",
            None,
        ),
        (
            "Train or Resume the League",
            "Default: 10 iterations, 512 games each, 2,000 updates. Saved games and update checkpoints survive disconnects. GPU collection follows the rule clock but does not measure CPU tournament speed.",
            """from chess_rl.self_play import run_league
champion = run_league(PROJECT_ROOT, cfg, initial_checkpoint, dataset_manifest)
print("Retained champion:", champion)""",
        ),
        (
            "Inspect Promotions",
            "Promotion requires at least 55% score and a paired 95% interval above 50%, without operational failures. Rejected candidates and results are retained.",
            """from chess_rl.plots import plot_league
display(plot_league(PROJECT_ROOT, cfg["run_id"]))
league = read_json(PROJECT_ROOT / "checkpoints/self_play" / cfg["run_id"] / "league.json")
for item in league["iterations"]:
    print(item["iteration"], "promoted:", item["promoted"], "score:", item["summary"]["score"])""",
        ),
        (
            "Optional Self-Play or Search Study",
            "Disabled by default. Trials initialize from the same checkpoint in separate folders. Search-only trials reuse weights; self-play trials train for three iterations.",
            """RUN_SEARCH_STUDY = False
SEARCH_ONLY = True
if RUN_SEARCH_STUDY:
    import yaml
    from chess_rl.tuning import run_self_play_search_study
    spaces = yaml.safe_load((PROJECT_ROOT / "configs/search_spaces.yaml").read_text())
    study_winner = run_self_play_search_study(PROJECT_ROOT, cfg, initial_checkpoint,
                                             dataset_manifest, spaces, search_only=SEARCH_ONLY)
    print("Research branch winner, not automatically final:", study_winner)
print("Open notebook 04 after all selected training runs finish.")""",
        ),
    ],
)

books["04_evaluation_and_comparison.ipynb"] = notebook(
    "04 - Evaluation and Comparison",
    "Compare frozen candidates on CPU. Development games support selection; held-out games follow freezing. "
    "This notebook does not run submission compliance.",
    [
        (
            "Load the Retained Champion",
            "The default final candidate is notebook 03's champion. HPO branches remain separate unless explicitly selected through a documented run.",
            """league_directory = PROJECT_ROOT / "checkpoints/self_play" / cfg["run_id"]
print(read_json(league_directory / "complete.json"))
league = read_json(league_directory / "league.json")
champion = PROJECT_ROOT / league["champion"]
previous = PROJECT_ROOT / league["history"][-2] if len(league["history"]) > 1 else None
initial = read_json(PROJECT_ROOT / "results" / cfg["run_id"] / "initial_selection.json")
supervised = PROJECT_ROOT / initial["checkpoint"]
print("Candidate:", champion)
print("Hash:", sha256(champion))""",
        ),
        (
            "Development Opponents",
            "128 development positions with colours swapped give 256 games per opponent, at 120 seconds plus 0.5 seconds. The preserved harness uses separate agent processes.",
            """from chess_rl.evaluation import compare_candidates
development_results = compare_candidates(PROJECT_ROOT, champion, cfg, previous=previous)
print(development_results)""",
        ),
        (
            "Development Scores",
            "Intervals resample opening pairs or source families. The reference is 50%; draws contribute half a point.",
            """from chess_rl.plots import plot_matches
display(plot_matches(PROJECT_ROOT, cfg["run_id"], development_results, "development_matches"))""",
        ),
        (
            "Controlled Ablations",
            "Compare classical search, policy-only, value-only, hybrid, and supervised weights. Clocks and fixtures are shared.",
            """RUN_ABLATIONS = True
if RUN_ABLATIONS:
    from chess_rl.evaluation import run_ablations
    ablation_results = run_ablations(PROJECT_ROOT, champion, cfg, supervised)
    display(plot_matches(PROJECT_ROOT, cfg["run_id"], ablation_results, "ablations"))""",
        ),
        (
            "Freeze and Evaluate Held-Out Games",
            "Complete all selected training and tuning first. This locks the checkpoint and search config. A different model cannot reuse this suite as if it had been preselected.",
            """heldout_results = compare_candidates(PROJECT_ROOT, champion, cfg, previous=previous, heldout=True)
display(plot_matches(PROJECT_ROOT, cfg["run_id"], heldout_results, "heldout_matches"))""",
        ),
        (
            "Record Final Selection",
            "This manifest unlocks notebook 05, recording actual completed training and evaluation.",
            """from chess_rl.evaluation import finalize_selection
selection_path = finalize_selection(PROJECT_ROOT, champion, cfg, heldout_results)
print("Frozen selection:", selection_path)
print("Notebook 04 complete. Open notebook 05.")""",
        ),
    ],
)

books["05_final_export_and_compliance.ipynb"] = notebook(
    "05 - Final Export and Compliance",
    "Run after training and evaluation. Export selected weights, construct a separate candidate, and check its artifacts. "
    "The starter agent is preserved. Nothing is uploaded automatically.",
    [
        (
            "Require Frozen Selection",
            "Earlier workflow completion is required. Failure does not trigger retraining.",
            """from chess_rl.export import freeze_requirement
selection, checkpoint = freeze_requirement(PROJECT_ROOT, cfg["run_id"])
print("Frozen checkpoint:", checkpoint)
print("Hash:", selection["checkpoint_sha256"])""",
        ),
        (
            "Optional ONNX Dependencies",
            "PyTorch state_dict is the default. ONNX is final-only and has separate runtime thread settings.",
            """if selection["config"]["export"]["format"] == "onnx":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])""",
        ),
        (
            "Export, Inspect, and Play",
            "Build the ZIP, check CPU reconstruction, measure batch-one latency, and play real games. The 5 ms target is experimental. Unreplicated isolation and human provenance review are reported as not verified.",
            """from chess_rl.export import final_checks
report = final_checks(PROJECT_ROOT, cfg["run_id"], live_games=16)
print("Status:", report["status"])
print("Archive:", report.get("archive"))
print("Uncompressed bytes:", report.get("unzipped_bytes"))
print("CPU inference milliseconds:", report.get("inference_ms"))
print("Checks:", report["checks"])""",
        ),
        (
            "Inspect Files and Logs",
            "The candidate contains only its runtime source, needed weights, configuration, and manifest.",
            """print("Candidate directory:", PROJECT_ROOT / "submission_candidate")
print("Report:", PROJECT_ROOT / "results" / cfg["run_id"] / "final_compliance.json")
print("Live results:", report.get("live_results"))
print("Original starter repository was not replaced.")""",
        ),
        (
            "Platform Submission",
            "The reported ZIP can be uploaded on the dashboard. Only platform validation establishes acceptance. Inspect a failed final report; this workflow does not automatically rerun training.",
            None,
        ),
    ],
)

destination = ROOT / "notebooks"
destination.mkdir(exist_ok=True)
for name, nb in books.items():
    nbf.write(nb, destination / name)
    print(name, len(nb.cells), "cells")
