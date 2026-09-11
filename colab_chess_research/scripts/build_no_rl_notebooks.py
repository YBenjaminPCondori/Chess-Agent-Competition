"""Build the independent no-RL Colab notebooks without rewriting the RL notebooks."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MOUNT = 'from google.colab import drive\ndrive.mount("/content/drive")'
SETUP = """from pathlib import Path
import sys
import subprocess
from IPython.display import display
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt

def show_figure(fig):
    display(fig)
    plt.close(fig)

PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess")
if not (PROJECT_ROOT / "chess_rl/non_rl.py").is_file():
    raise FileNotFoundError(f"Place the updated project contents directly in {PROJECT_ROOT}")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
                       str(PROJECT_ROOT / "requirements_colab.txt")])
sys.path.insert(0, str(PROJECT_ROOT))
from chess_rl.non_rl import load_no_rl_config
from chess_rl.reproducibility import read_json, sha256
cfg = load_no_rl_config(PROJECT_ROOT)
print("Run:", cfg["run_id"], "| Variant:", cfg["non_rl"]["variant"])
print("Root:", PROJECT_ROOT)
"""


def notebook(title, intro, sections):
    cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}\n\n## Mount Google Drive"),
        nbf.v4.new_code_cell(MOUNT),
        nbf.v4.new_markdown_cell(
            "## Project Setup\n\nUses `configs/no_rl.yaml`. No league, self-play, PPO, or DQN is run. Colab's installed PyTorch is preserved."
        ),
        nbf.v4.new_code_cell(SETUP),
    ]
    for heading, prose, code in sections:
        cells.append(nbf.v4.new_markdown_cell(f"## {heading}\n\n{prose}"))
        if code:
            cells.append(nbf.v4.new_code_cell(code))
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
    )


books = {}
books["01_classical_baseline.ipynb"] = notebook(
    "No RL 01 - Classical Baseline",
    "Use the existing alpha-beta engine without a neural model. Evaluation coefficients can be tuned through games in notebook 03. "
    "Default path: **01 -> 03 -> 04 -> 05**. "
    "Optional supervised path: **01 -> 02 -> 03 -> 04 -> 05**. Use a CPU runtime here.",
    [
        (
            "Choose the Variant",
            "For no neural training, leave `non_rl.variant: classical` in `configs/no_rl.yaml`. "
            "For supervised learning without RL, set it to `supervised` and change `run_id` to a new name such as `no_rl_supervised_v1`. "
            "Keep the same configuration throughout one run. Do not change the original `default.yaml` to select this workflow.",
            'print("Settings file:", PROJECT_ROOT / "configs/no_rl.yaml")\nprint(cfg["search"])',
        ),
        (
            "What the Baseline Knows",
            "The current evaluator uses material, piece-square tables, attacked-square activity, bishop pairs, "
            "doubled/isolated pawns, and passed pawns. Iterative deepening, alpha-beta, quiescence, and a transposition table provide calculation. "
            "This notebook does not add opening books or implement every strategy from a chess textbook.",
            None,
        ),
        (
            "Inspect a Position",
            "The board is standard chess, not a reduced grid environment. The classical evaluator returns a score "
            "from the side-to-move perspective. The displayed position is an illustrative input, not a benchmark.",
            """import chess
from chess_rl.classical_evaluation import evaluate_cp, evaluation_breakdown
from chess_rl.search import SearchEngine
from chess_rl.non_rl import variant_config

board = chess.Board()
for uci in ("e2e4", "e7e5", "g1f3"):
    board.push_uci(uci)
display(board)
print("Legal moves:", board.legal_moves.count())
print("Static evaluation (centipawn units):", evaluate_cp(board, cfg["search"]["evaluation_weights"]))
for row in evaluation_breakdown(board, cfg["search"]["evaluation_weights"]):
    print(f'{row["feature"]:22s} {row["value"]:7g} x {row["weight"]:7g} = {row["contribution_cp"]:8.2f} cp')""",
        ),
        (
            "Search for a Move",
            "This runs the classical search directly. A legal fallback is available before deeper search. "
            "The board must be unchanged when search returns. This is ordinary correctness work, not submission validation.",
            """classical_cfg = variant_config(cfg, "classical")
engine = SearchEngine(classical_cfg["search"])
before = board.fen()
result = engine.choose(board, time_left_ms=2000)
assert result.move in board.legal_moves
assert board.fen() == before
assert engine.neural is None
print("Move:", result.move.uci(), "| Completed depth:", result.depth,
      "| Nodes:", result.nodes, "| Time (ms):", round(result.elapsed_ms, 2))
after = board.copy()
after.push(result.move)
display(after)""",
        ),
        (
            "Reserve Opening Suites",
            "The same immutable development/held-out fixtures are shared with the research project. "
            "Their legal random playout provenance is recorded; they are not balanced or the platform's hidden opening set.",
            """from chess_rl.dataset import prepare_openings
opening_manifest = prepare_openings(PROJECT_ROOT, seed=cfg["seed"])
print(opening_manifest)""",
        ),
        (
            "Check Workflow Correctness",
            "Bounded tests verify the independent classical path and its stage dependencies. "
            "They do not start full training, full tournaments, or submission checks.",
            """subprocess.check_call([sys.executable, "-m", "pytest", "-q", "tests/test_non_rl.py",
                       "tests/test_evaluation_tuning.py"],
                      cwd=PROJECT_ROOT)""",
        ),
        (
            "Next Notebook",
            "Classical: open **03_evaluate_no_rl.ipynb**. Supervised: open **02_optional_supervised.ipynb**. "
            "Do not run the original RL notebook 03 for this workflow.",
            None,
        ),
    ],
)
books["02_optional_supervised.ipynb"] = notebook(
    "No RL 02 - Optional Supervised Learning",
    "Optional: train the existing policy/value architecture against labelled positions, without self-play. "
    "This is supervised deep learning, not a requirement for the classical baseline. Set variant to `supervised` before running. "
    "An A100/CUDA runtime accelerates this optional notebook only; inference remains CPU.",
    [
        (
            "Training Choice",
            "Every expensive cell is disabled for the default classical variant. "
            "The default supervised model has six 128-channel residual blocks, a 4,672-action policy head, and one value output. "
            "Defaults: up to 20 epochs, AdamW, learning rate 0.0003, cosine scheduling, and validation early stopping.",
            """TRAIN_SUPERVISED = cfg["non_rl"]["variant"] == "supervised"
print("Supervised training enabled:", TRAIN_SUPERVISED)
if TRAIN_SUPERVISED:
    from chess_rl.reproducibility import seed_all, resolve_device
    seed_all(cfg["seed"], cfg["deterministic"])
    print("Training device:", resolve_device(cfg["device"]))
    print("Architecture:", cfg["model"])
else:
    print("Skip this notebook. Open no-RL notebook 03.")""",
        ),
        (
            "Offline Teacher",
            "Stockfish is used only to label offline data. It is not shipped in the candidate. "
            "The configured classical teacher is also supported. This setup runs only for the supervised variant.",
            """if TRAIN_SUPERVISED:
    teacher_path = Path(cfg["dataset"]["engine_path"])
    if cfg["dataset"]["teacher"] == "stockfish" and not teacher_path.is_file():
        subprocess.check_call(["apt-get", "update", "-qq"])
        subprocess.check_call(["apt-get", "install", "-y", "-qq", "stockfish"])
    if cfg["dataset"]["teacher"] == "stockfish":
        if not teacher_path.is_file():
            raise FileNotFoundError(f"Configure dataset.engine_path: {teacher_path}")
        print("Teacher hash:", sha256(teacher_path))""",
        ),
        (
            "Build or Resume the Dataset",
            "Provide PGNs or JSONL through a `dataset` override in `configs/no_rl.yaml`. "
            "Empty broad sources now stop with a clear error instead of generating synthetic training positions. "
            "This is not a curated strategy curriculum. Source-game splits and held-out exclusion use the shared dataset implementation.",
            """if TRAIN_SUPERVISED:
    from chess_rl.dataset import prepare_dataset, prepare_openings
    prepare_openings(PROJECT_ROOT, seed=cfg["seed"])
    dataset_manifest = prepare_dataset(PROJECT_ROOT, cfg)
    print("Actual split counts:", dataset_manifest["counts"])
    print("Target reached:", dataset_manifest["target_reached"])""",
        ),
        (
            "Train and Save",
            "Policy targets are legal teacher moves; value targets are teacher evaluations or completed-game results, "
            "with their sources distinguished. Training uses legal-only cross-entropy plus value Huber loss. "
            "Latest/best checkpoints, optimizer state, RNG, config, and CSV logs are saved. Interrupted epochs resume from the last completed epoch.",
            """if TRAIN_SUPERVISED:
    from chess_rl.training import fit_supervised
    checkpoint = fit_supervised(PROJECT_ROOT, cfg, dataset_manifest)
    print("Best supervised checkpoint:", checkpoint)
    print("SHA256:", sha256(checkpoint))""",
        ),
        (
            "Inspect Learning Curves",
            "These plots use actual completed-epoch logs only. Lower label loss does not establish stronger play. "
            "Move on to no-RL notebook 03 to measure games; no self-play stage follows this notebook.",
            """if TRAIN_SUPERVISED:
    from chess_rl.plots import plot_supervised
    show_figure(plot_supervised(PROJECT_ROOT, cfg["run_id"]))""",
        ),
    ],
)
books["03_evaluate_no_rl.ipynb"] = notebook(
    "No RL 03 - Evaluation-Weight Tuning and Development Games",
    "Play the selected classical or supervised variant through the preserved process harness on CPU. "
    "The classical path requires no checkpoint. It tunes numeric evaluation coefficients through matches, not RL or neural training. "
    "Actual PGNs, move clocks, logs, and CSV scores are saved.",
    [
        (
            "Inspect the Match Settings",
            "Defaults are 256 games per opponent, with colour-swapped opening pairs, at 120 seconds + 0.5 seconds. "
            "Changing parameters is an experiment: use a new run_id. Game results depend on the local CPU, not just the GPU allocation.",
            'print(cfg["evaluation"])',
        ),
        (
            "Tune Classical Evaluation Weights",
            "Default: compare eight seeded coefficient sets against the fixed starting engine, with 32 games per trial. "
            "Confirm only the best eligible trial over 128 games on unused development openings. Promote only with score >= 55%, "
            "an opening-family bootstrap lower bound above 50%, and no recorded runtime failures on either side. Otherwise keep the baseline. "
            "These defaults are experiment settings, not established optimal values. Held-out openings are never used to select weights. "
            "Edit `non_rl.tuning.ranges` or `search.evaluation_weights` in the config before starting a new run. "
            "Set `non_rl.tuning.enabled: false` to evaluate your manually chosen weights directly. Supervised runs skip this classical tuner.",
            """from chess_rl.evaluation_tuning import tune_classical_weights
tuning = tune_classical_weights(PROJECT_ROOT, cfg)
if tuning is not None:
    print("Decision:", tuning["decision"])
    print("Selected coefficients:", tuning["selected_weights"])
    print("Trial CSV:", PROJECT_ROOT / "results" / cfg["run_id"] / "no_rl/weight_tuning/trials.csv")
else:
    print("Classical weight tuning disabled or supervised variant selected.")""",
        ),
        (
            "Inspect Screening Results",
            "Screening scores select a candidate; they are not independent evidence of improvement. The separate confirmation "
            "match decides whether its weights are kept. Notebook 04 subsequently measures the frozen choice on held-out games.",
            """if tuning is not None:
    from chess_rl.plots import plot_matches
    trial_summaries = {f'Trial {row["trial"]}': row["summary"] for row in tuning["trials"]}
    if any(row["score"] is not None for row in trial_summaries.values()):
        show_figure(plot_matches(PROJECT_ROOT, cfg["run_id"], trial_summaries, "no_rl_weight_screening"))
    else:
        print("No scored screening games. Inspect the saved failure logs.")
    print("Confirmation:", tuning["confirmation"] or "Not run: no usable screening score above 50%.")""",
        ),
        (
            "Run or Resume Development Games",
            "Opponents: greedy, minimax, and the preserved original classical agent. "
            "For a supervised run, also evaluate a matched no-network control and play the supervised agent against that control. "
            "Completed games are reused only when their inputs match.",
            """from chess_rl.non_rl import evaluate_no_rl
development = evaluate_no_rl(PROJECT_ROOT, cfg)
print("Saved results:", PROJECT_ROOT / "results" / cfg["run_id"] / "no_rl/development.json")""",
        ),
        (
            "Compare Scores",
            "Draws contribute half a point. Intervals use opening-family paired bootstrap; they are unavailable "
            "when too few independent pairs were played. Do not claim an improvement from an uncertain interval.",
            """from chess_rl.plots import plot_matches
show_figure(plot_matches(PROJECT_ROOT, cfg["run_id"], development["summaries"], "no_rl_development"))
if development["classical_control"]:
    show_figure(plot_matches(PROJECT_ROOT, cfg["run_id"], development["classical_control"], "no_rl_classical_control"))""",
        ),
        (
            "Find the Games",
            "Open PGNs using a chess viewer to replay full games. The saved files below are from this run only.",
            """match_root = PROJECT_ROOT / "results" / cfg["run_id"] / "matches"
for pgn in sorted(match_root.glob("no-rl-dev-*/game-*.pgn"))[:5]:
    print(pgn)
print("Next: no-RL notebook 04 freezes this variant before held-out evaluation.")""",
        ),
    ],
)
books["04_freeze_no_rl.ipynb"] = notebook(
    "No RL 04 - Held-Out Evaluation and Selection",
    "Freeze the selected variant, then measure it on the held-out suite. No RL completion files are read. "
    "This records a selection, not an assertion that the agent is strong or accepted by the platform.",
    [
        (
            "Confirm the Candidate",
            "Complete development experiments before this step. The lock covers source hashes, search settings, "
            "and any supervised checkpoint. Do not reuse this held-out suite as fresh evidence after tuning to its results.",
            """from chess_rl.non_rl import candidate_spec
spec = candidate_spec(PROJECT_ROOT, cfg)
print("Variant:", spec["variant"])
print("Checkpoint:", spec["checkpoint"] or "None: classical, no training")
print("Held-out games per opponent:", cfg["evaluation"]["heldout_games"])""",
        ),
        (
            "Run or Resume Held-Out Games",
            "The candidate is locked before games begin. All opponents use the same reserved fixtures and clocks. "
            "The selected run may be exported even if its playing strength is poor, but final runtime checks can still fail.",
            """from chess_rl.non_rl import evaluate_and_freeze_no_rl
selection = evaluate_and_freeze_no_rl(PROJECT_ROOT, cfg)
print("Selection:", PROJECT_ROOT / "results" / cfg["run_id"] / "no_rl/final_selection.json")
print("Training status:", selection["training_status"])""",
        ),
        (
            "Read the Results",
            "These are observed scores, not guaranteed future performance. Colab does not reproduce the platform's "
            "CPU speed, isolation, or hidden openings. Final packaging/compliance is exclusively in no-RL notebook 05.",
            """from chess_rl.plots import plot_matches
show_figure(plot_matches(PROJECT_ROOT, cfg["run_id"], selection["heldout"], "no_rl_heldout"))""",
        ),
    ],
)
books["05_export_no_rl.ipynb"] = notebook(
    "No RL 05 - Final Export and Local Checks",
    "Run only after the selected no-RL evaluation is complete. Classical export contains no trained model and requires no training. "
    "Supervised export requires only supervised training, not self-play. No upload happens automatically.",
    [
        (
            "Verify the Selected Run",
            "Classical runtime uses only standard library and python-chess. Supervised runtime adds CPU PyTorch "
            "with one thread. It uses the original encoder/model/search code, not a separately rewritten implementation.",
            """from chess_rl.non_rl import selected_no_rl
selection = selected_no_rl(PROJECT_ROOT, cfg["run_id"])
print("Variant:", selection["variant"])
print("Training:", selection["training_status"])
print("Evaluation complete:", selection["evaluation_complete"])""",
        ),
        (
            "Export and Check",
            "This is the only no-RL notebook that builds/audits the submission archive or performs fresh-process "
            "deployment checks. It checks legal moves, low clocks, import dependencies, archive root/size, and final live games. "
            "Supervised weights also undergo a round trip and missing-weight fallback check. Refer to the current official rules before uploading.",
            """from chess_rl.non_rl_export import final_no_rl_checks
report = final_no_rl_checks(PROJECT_ROOT, cfg["run_id"])
print("Local status:", report["status"])
print("Submission:", report["submission"])
print("Uncompressed bytes:", report["uncompressed_bytes"])
print("Platform acceptance:", report["platform_acceptance"])""",
        ),
        (
            "Submission Location",
            "The ZIP has agent.py at its root. Classical coefficients are in config.json, with no neural weight file. For supervised, it additionally "
            "contains team-trained CPU weights. Do not upload the portable project ZIP as your agent. Actual server acceptance remains authoritative.",
            """print(PROJECT_ROOT / "exports/no_rl" / cfg["run_id"] / cfg["non_rl"]["variant"] / "submission.zip")
print("Official documentation: https://aichessathon.com/docs")""",
        ),
    ],
)


if __name__ == "__main__":
    output = ROOT / "notebooks/no_rl"
    output.mkdir(parents=True, exist_ok=True)
    for name, book in books.items():
        nbf.validate(book)
        nbf.write(book, output / name)
        print(name)
