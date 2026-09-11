"""Generate the ten active research notebooks and preserve Git originals in an archive."""

from pathlib import Path
import hashlib
import json
import subprocess
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
cfg = load_config(PROJECT_ROOT, "strategy.yaml")
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
books["01_environment_and_encoding.ipynb"] = notebook(
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
from chess_rl.reservations import prepare_reservations
reservations = prepare_reservations(PROJECT_ROOT, cfg)
prepare_openings(PROJECT_ROOT, seed=cfg["seed"])
print("Reserved suites:", reservations["suites"])""",
        ),
        (
            "Next Notebook",
            "After 01, notebooks 02 and 03a may run independently or in parallel. Configure real strategy sources before 01; changing sources requires a new reservation_id. Every notebook reloads persisted inputs from Drive.",
            None,
        ),
    ],
)

books["02_supervised_broad_training.ipynb"] = notebook(
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
            "Supply compatible JSONL, such as converted Lichess Eval DB records, through dataset.jsonl_paths, or PGNs through dataset.pgn_paths for teacher labelling. Empty broad sources now stop with a clear error instead of generating synthetic training positions. The target is 100,000 unique positions. Saved annotation shards support resuming PGN teacher labelling.",
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
    from chess_rl.reservations import load_reservations
    spaces = yaml.safe_load((PROJECT_ROOT / "configs/search_spaces.yaml").read_text())
    shortlist = run_model_study(PROJECT_ROOT, cfg, dataset_manifest, spaces)
    comparisons = []
    for trial in shortlist:
        checkpoint = PROJECT_ROOT / trial["checkpoint"]
        candidate = build_research_agent(PROJECT_ROOT, checkpoint, trial["config"], "shortlist")
        result = run_matchup(PROJECT_ROOT, candidate, PROJECT_ROOT / "reference/classical_agent",
                            PROJECT_ROOT / load_reservations(PROJECT_ROOT, cfg)["suites"]["development"],
                            trial["config"], "shortlist-development")
        comparisons.append((result["score"] if result["score"] is not None else -1, checkpoint))
    if not comparisons:
        raise RuntimeError("No successful model-search trials")
    selected_pretraining = max(comparisons, key=lambda item: item[0])[1]
    print("Shortlist match selection:", selected_pretraining)""",
        ),
        (
            "Persist Initialization Choice",
            "Notebook 03b and optional stage 04a read this checkpoint reference. Independent experiments need different run identifiers.",
            """atomic_json(PROJECT_ROOT / "results" / cfg["run_id"] / "initial_selection.json",
            {"checkpoint": str(selected_pretraining.relative_to(PROJECT_ROOT)),
             "sha256": sha256(selected_pretraining)})
print("Notebook 02 complete. Open 03a_build_strategy_datasets.ipynb.")""",
        ),
    ],
)


# All research stages use the same reservation and dataset configuration.

books["03a_build_strategy_datasets.ipynb"] = notebook(
    "03a - Build Strategy Datasets",
    "CPU only. Supply your own strategy positions; this stage does not train a model. "
    "All eleven themes are supported. PGN comments and annotations are not imported.",
    [
        (
            "Configure Sources",
            "Edit configs/strategy.yaml: strategy_dataset.sources accepts PGN, one-FEN-per-line, CSV or JSONL. "
            "Set theme/subtheme there or in each CSV/JSONL row. A line is a JSON list of legal UCI moves. "
            "Converted Lichess Puzzle DB and STS-Rating files belong here; motif detector tags must be written before this notebook runs. "
            "Played moves and lines are observations unless best_move/policy_target is explicitly supplied. "
            "Optional UCI labelling uses one CPU thread and fails clearly if its configured executable is missing.",
            """from chess_rl.strategy_taxonomy import TAXONOMY
for theme, subthemes in TAXONOMY.items():
    print(theme, ":", ", ".join(subthemes))
print("Sources:", cfg["strategy_dataset"]["sources"])""",
        ),
        (
            "Build and Persist",
            "Whole source games share a split. Positions duplicated across partitions are removed. "
            "Notebook 01 fixes source partitions and reserved openings before either 02 or 03a. Annotated benchmark games default to test. "
            "PGNs with common early openings therefore lose those shared positions. Supply enough independent games.",
            """from chess_rl.strategy_dataset import build_strategy_datasets
manifest = build_strategy_datasets(PROJECT_ROOT, cfg)
print("Split counts:", manifest["counts"])
print("Saved to:", PROJECT_ROOT / "data/strategy" / cfg["strategy_dataset"]["dataset_id"])""",
        ),
        (
            "Summary Tables and Charts",
            "CSV and JSONL share the documented schema. Small corpora may have empty partitions; "
            "03b requires labelled train and validation rows. No labels are fabricated to fill a gap.",
            """from chess_rl.strategy_plots import dataset_summary
display(dataset_summary(PROJECT_ROOT, cfg["run_id"], manifest))""",
        ),
        (
            "History and Heuristic Limits",
            "FEN validity checks basic chess constraints, not historical reachability. PGN history is retained as "
            "moves for repetition diagnostics. FEN alone cannot reveal repetition or prior castling. "
            "Backward pawns, outposts, trapped pieces, bad bishops and fortress-like positions are diagnostics, "
            "not adjudications or training labels. Sacrifice, skewer and annotated-game themes require curated tags.",
            None,
        ),
    ],
)

books["03b_strategy_finetuning.ipynb"] = notebook(
    "03b - Strategy Fine-Tuning",
    "Supervised transfer learning from the completed broad checkpoint in 02. "
    "Use a GPU runtime for real training; CPU works for small experiments. Self-play is optional and separate.",
    [
        (
            "Fine-Tuning Configuration",
            "Edit strategy_finetuning in configs/strategy.yaml. Select themes and manifests; use a distinct "
            "candidate id for every specialist. The combined model and each selected specialist start independently from 02. "
            "Defaults: two head-only epochs at 1e-4 then up to eight full epochs at 3e-5; AdamW, cosine, "
            "weight decay 1e-4, patience 3 in the full phase, 512 CUDA/64 CPU. Frozen BatchNorm stays fixed. "
            "Each batch is 75% strategy and 25% broad training replay, with theme-balanced strategy sampling.",
            """import torch
from chess_rl.strategy_finetuning import load_finetuning_config, finetune_strategy
cfg = load_finetuning_config(PROJECT_ROOT)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Available training device:", device)
print(cfg["strategy_finetuning"])""",
        ),
        (
            "Load Base and Train or Resume",
            "initial_selection.json from 02 supplies the base checkpoint and checksum. The architecture is loaded "
            "from that checkpoint. No random replacement is made when the base is missing. "
            "Training uses legal policy targets and side-to-move values; unlabelled rows are excluded.",
            """checkpoints = finetune_strategy(PROJECT_ROOT, cfg)
print("Completed candidates:", checkpoints)""",
        ),
        (
            "Saved Metrics",
            "Candidates and resume pointers are under models/strategy/<run_id>/<candidate_id>/. "
            "Epoch checkpoints retain metrics, optimizer, scheduler, RNG and sampling state. Test data is never used for early stopping.",
            """directory = PROJECT_ROOT / "models/strategy" / cfg["run_id"]
print(read_json(directory / "completed.json"))
print("Next: optional 04a/04b, or 05a.")""",
        ),
    ],
)

SELF_PLAY_LOAD = """from chess_rl.research_workflow import self_play_inputs
initial_checkpoint, dataset_manifest = self_play_inputs(PROJECT_ROOT, cfg)
print("Frozen initialization:", initial_checkpoint)
print("Settings:", cfg["self_play"])"""

books["04a_self_play_generation.ipynb"] = notebook(
    "04a - Self-Play Generation",
    "Optional RL stage. Collect the next iteration with frozen models on CPU by default. "
    "Generation does not apply gradient updates; 04b training uses CUDA when available.",
    [
        (
            "Load Initialization",
            "Default is the completed combined strategy checkpoint. Broad initialization requires strategy explicitly skipped. To override, set "
            "research_workflow.self_play_initial_checkpoint before beginning a new league run.",
            SELF_PLAY_LOAD,
        ),
        (
            "Generate or Resume Games",
            "Records use engine-search policy targets and completed-game results. "
            "Failed games are excluded from training. Strategy validation/test positions and reserved broad/opening "
            "positions are excluded. Completed games and their hashes are persisted.",
            """from chess_rl.self_play import generate_self_play
generation = generate_self_play(PROJECT_ROOT, cfg, initial_checkpoint, dataset_manifest)
print("Generation:", generation.get("iteration", generation.get("status")))
print("Next: 04b_self_play_training_and_promotion.ipynb.")""",
        ),
    ],
)

books["04b_self_play_training_and_promotion.ipynb"] = notebook(
    "04b - Self-Play Training and Promotion",
    "Consume completed 04a games, train a candidate, and run CPU promotion matches. "
    "GPU is recommended for gradient updates. The champion changes only after successful promotion.",
    [
        (
            "Load Inputs",
            "Run 04a for the next iteration first. A missing or changed generation manifest stops training.",
            SELF_PLAY_LOAD,
        ),
        (
            "Train and Promote One Iteration",
            "Resume optimizer updates from complete checkpoints. "
            "Promotion retains the established score, paired-confidence and runtime-failure criteria.",
            """from chess_rl.self_play import train_and_promote
champion = train_and_promote(PROJECT_ROOT, cfg, initial_checkpoint, dataset_manifest)
print("Retained champion:", champion)
league = read_json(PROJECT_ROOT / "checkpoints/self_play" / cfg["run_id"] / "league.json")
print("Completed iterations:", league["completed_iterations"], "/", cfg["self_play"]["iterations"])""",
        ),
        (
            "Continue the League",
            "Repeat 04a then 04b until all configured iterations finish. "
            "This optional runner performs that same alternating sequence for the remaining iterations.",
            """RUN_REMAINING_ITERATIONS = False
if RUN_REMAINING_ITERATIONS:
    from chess_rl.self_play import run_league
    champion = run_league(PROJECT_ROOT, cfg, initial_checkpoint, dataset_manifest)
print("Next: 05a after the chosen training runs have completed.")""",
        ),
    ],
)

books["05a_evaluate_general_strength.ipynb"] = notebook(
    "05a - Evaluate General Strength",
    "CPU only. Compare completed broad, specialist, optional self-play and classical candidates. "
    "These are research matches, with no submission export checks.",
    [
        (
            "Discover Candidates",
            "Load or create the immutable candidate registry, independently of 05b. Complete configured training or explicitly skip stages before registration. "
            "Optional no-RL selections and additional trusted checkpoints can be named in research_workflow. "
            "Self-play is included only when its configured league has completed.",
            """import torch
torch.set_num_threads(1)
from chess_rl.candidate_registry import load_candidate_registry
registry = load_candidate_registry(PROJECT_ROOT, cfg)
candidates = registry["candidates"]
for candidate in candidates:
    print(candidate["id"], candidate["kind"], candidate["checkpoint"])""",
        ),
        (
            "Play Common Development Opponents",
            "All candidates use the same clocks, opening pairs and "
            "greedy/minimax/original-classical opponents. The development results support selection; "
            "held-out games wait until 06a freezes one winner.",
            """from chess_rl.research_workflow import general_evaluation
general = general_evaluation(PROJECT_ROOT, cfg, candidates)
for name, opponents in general["results"].items():
    print(name, {opponent: result["score"] for opponent, result in opponents.items()})
print("Saved:", PROJECT_ROOT / "results" / cfg["run_id"] / "general_evaluation.json")""",
        ),
    ],
)

books["05b_evaluate_strategy_suites.ipynb"] = notebook(
    "05b - Evaluate Strategy Suites",
    "CPU only. Evaluate all eleven strategy themes with explicit label coverage. "
    "Validation suites support selection; the frozen winner's test suites, including held-out annotated games, run in 06a.",
    [
        (
            "Load the Same Candidates",
            "Load the common registry independently of 05a; no match results are required. It includes the independent classical "
            "control and any configured no-RL runs. For neural models, top-k compares legal policy rankings; "
            "move accuracy compares the search-selected move. Classical top-k is unavailable and value is a heuristic.",
            """import torch
torch.set_num_threads(1)
from chess_rl.candidate_registry import load_candidate_registry
registry = load_candidate_registry(PROJECT_ROOT, cfg)
candidates = registry["candidates"]
print("Evaluation split:", cfg["strategy_evaluation"]["split"])""",
        ),
        (
            "Evaluate Themed Suites",
            "Report move accuracy, top-k agreement, value MAE, legal move rate, "
            "prediction failures and per-position/per-theme pass/fail/not_assessable counts. "
            "Neural policy top-1 and top-3 are separate from search agreement. History-dependent cases are reported separately. "
            "Acceptable move sets are validated; missing labels are not failures. Heuristics are not gold labels.",
            """from chess_rl.strategy_evaluation import evaluate_candidates
strategy = evaluate_candidates(PROJECT_ROOT, cfg, candidates)
from chess_rl.strategy_plots import evaluation_summary
display(evaluation_summary(PROJECT_ROOT, cfg["run_id"], strategy))""",
        ),
        (
            "Persistent Results",
            "Per-position observations, summary JSON/CSV and a plot are saved together.",
            """print(PROJECT_ROOT / "results/strategy_evaluation" / cfg["run_id"] / cfg["strategy_evaluation"]["split"])
print("Next: 06a_select_and_freeze_winner.ipynb.")""",
        ),
    ],
)

books["06a_select_and_freeze_winner.ipynb"] = notebook(
    "06a - Select and Freeze Winner",
    "CPU only. Select from completed candidates using recorded development and strategy-validation evidence, "
    "freeze the choice, then measure its held-out general and strategy performance.",
    [
        (
            "Selection Criteria",
            "Reject candidates with runtime/model errors or unusable development results. "
            "Apply configured minimum general score and optional strategy-pass requirement. Rank by mean opponent "
            "score, then candidate id for deterministic ties; strategy scores are diagnostic by default. "
            "Only the leading challenger faces the fixed classical incumbent on the separate 256-game confirmation suite. "
            "Replacement requires at least 55% score, paired-bootstrap lower bound above 50%, and no runtime failures. "
            "Failure retains the incumbent; never try another challenger on the same suite. "
            "The frozen winner cannot be replaced using its test results.",
            """print(cfg["research_workflow"]["selection"])""",
        ),
        (
            "Freeze and Measure Held-Out Performance",
            "Lock source hashes, configuration, checkpoint hash, "
            "opening hashes, evaluation summaries and ranking before held-out games. "
            "The chosen model alone runs test suites. The final selection works without a self-play checkpoint, "
            "including when a classical candidate wins.",
            """from chess_rl.research_workflow import freeze_winner
selection_path = freeze_winner(PROJECT_ROOT, cfg)
selection = read_json(selection_path)
print("Selected:", selection["selected_id"], selection["kind"])
print("Frozen manifest:", selection_path)
print("Next: final export notebook 06b.")""",
        ),
    ],
)

books["06b_export_submission.ipynb"] = notebook(
    "06b - Export Submission",
    "CPU only, final stage. Export the frozen winner as agent.py and submission.zip, "
    "then run the final submission checks. Nothing is uploaded automatically.",
    [
        (
            "Load the Frozen Winner",
            "Require successful completion of 06a and verify its frozen sources, "
            "configuration, checkpoint and evaluation evidence. The API remains get_move(fen: str, time_left_ms: int) -> str.",
            """import torch
torch.set_num_threads(1)
from chess_rl.export import freeze_requirement
selection, checkpoint = freeze_requirement(PROJECT_ROOT, cfg["run_id"])
print("Winner:", selection["selected_id"], "Checkpoint:", checkpoint)""",
        ),
        (
            "Optional ONNX Dependencies",
            "Only needed for an explicitly configured neural ONNX export.",
            """if checkpoint is not None and selection["config"]["export"]["format"] == "onnx":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])""",
        ),
        (
            "Export and Final Checks",
            "The selected neural or classical export runs import/API checks, legal "
            "moves, low-clock timing, CPU-only inference, package integrity/size and agent.py-at-ZIP-root checks. "
            "It also plays final local games. A failed report requires investigation; this stage does not retrain.",
            """from chess_rl.export import final_selected_checks
report = final_selected_checks(PROJECT_ROOT, cfg["run_id"], live_games=16)
print("Status:", report["status"])
print("Submission:", report.get("archive", report.get("submission")))
print("Checks:", report["checks"])
print("Report:", PROJECT_ROOT / "results" / cfg["run_id"] / "final_compliance.json")""",
        ),
        (
            "Submission",
            "Use the submission.zip reported above for manual platform submission. "
            "Local timing is not a measurement of the platform host, and local success is not platform acceptance.",
            None,
        ),
    ],
)

destination = ROOT / "notebooks"
destination.mkdir(exist_ok=True)
LEGACY_NAMES = (
    "01_chess_environment_and_encoding.ipynb",
    "02_supervised_policy_value_training.ipynb",
    "03_self_play_league_training.ipynb",
    "04_evaluation_and_comparison.ipynb",
    "05_final_export_and_compliance.ipynb",
)
# Recover exact Git blobs. Exclusive creation prevents replacing earlier archives.
git_root = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
archive = destination / "archive" / ("legacy-" + revision[:12])
archive.mkdir(parents=True, exist_ok=True)
provenance = {}
for name in LEGACY_NAMES:
    relative = (destination / name).relative_to(git_root).as_posix()
    content = subprocess.check_output(["git", "show", revision + ":" + relative], cwd=ROOT)
    archived = archive / name
    if not archived.exists():
        with archived.open("xb") as stream:
            stream.write(content)
    if archived.read_bytes() != content:
        raise ValueError("Existing archive differs; it will not be overwritten")
    provenance[name] = dict(
        commit=revision, git_path=relative, sha256=hashlib.sha256(content).hexdigest()
    )
manifest_path = archive / "provenance.json"
if not manifest_path.exists():
    with manifest_path.open("x") as stream:
        json.dump(provenance, stream, indent=2)
dependencies = {
    "01": [],
    "02": ["01"],
    "03a": ["01"],
    "03b": ["02", "03a"],
    "04a": ["02", "03b_if_required", "previous_04b"],
    "04b": ["current_04a"],
    "05a": ["completed_or_explicitly_skipped_training", "candidate_registry"],
    "05b": ["completed_or_explicitly_skipped_training", "candidate_registry", "03a_if_sources"],
    "06a": ["05a", "05b"],
    "06b": ["06a"],
}
for name, book in books.items():
    stage = name.split("_")[0]
    book.metadata.research_workflow = dict(stage=stage, depends_on=dependencies[stage])
    for index, cell in enumerate(book.cells):
        cell.id = f"cell-{index:03d}"
    nbf.write(book, destination / name)
    print(name, len(book.cells), "cells")
for name in LEGACY_NAMES:
    path = destination / name
    if path.exists():
        if path.read_bytes() != (archive / name).read_bytes():
            raise ValueError(
                "Active legacy notebook has edits; preserve it separately before migration"
            )
        path.unlink()
