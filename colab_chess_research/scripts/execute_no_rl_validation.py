"""Bounded LOCAL adaptations of no-RL 01-04; never trains or executes notebook 05."""

from pathlib import Path
import shutil
import tempfile
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    from chess_rl.dataset import write_jsonl

    output = ROOT / "docs/validation/no_rl"
    output.mkdir(parents=True, exist_ok=True)
    work = ROOT / "results/no_rl_notebook_validation"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fixture-", dir=work) as temporary:
        project = Path(temporary)
        for directory in ("chess_rl", "configs", "templates", "reference", "tests", "notebooks"):
            shutil.copytree(
                ROOT / directory, project / directory, ignore=shutil.ignore_patterns("__pycache__")
            )
        for name, pawns in (("development", ("6PP", "5P1P")), ("heldout", ("4P2P",))):
            write_jsonl(
                project / f"datasets/openings/{name}.jsonl",
                [
                    dict(
                        fen=f"6k1/8/8/8/8/8/{pieces}/6K1 b - - 0 300",
                        opening_id=f"local-fixture-{name}-{index}",
                        family_id=f"local-fixture-{name}-{index}",
                    )
                    for index, pieces in enumerate(pawns)
                ],
            )
        setup = (
            "from pathlib import Path\nimport sys, subprocess\n"
            "from IPython.display import display\n"
            'get_ipython().run_line_magic("matplotlib", "inline")\n'
            "import matplotlib.pyplot as plt\n"
            "def show_figure(fig):\n    display(fig)\n    plt.close(fig)\n"
            f"PROJECT_ROOT = Path({str(project)!r})\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "from chess_rl.non_rl import load_no_rl_config\n"
            "from chess_rl.reproducibility import read_json, sha256\n"
            "cfg = load_no_rl_config(PROJECT_ROOT)\n"
            'cfg["non_rl"]["variant"] = "classical"\n'
            'cfg["run_id"] = "no_rl_local_fixture"\n'
            'cfg["evaluation"].update(development_games=2, heldout_games=2, bootstrap_samples=100)\n'
            'cfg["non_rl"]["tuning"].update(trials=1, games_per_trial=2, confirmation_games=2)\n'
            'print("LOCAL FIXTURE: CPU, one ply before game cap, two games per opponent.")\n'
            'print("No Drive mount, no training, no export/compliance execution.")\n'
            "from chess_rl import plots\n"
            "original_plot_matches = plots.plot_matches\n"
            "def fixture_plot(root, run_id, summaries, name):\n"
            "    fig = original_plot_matches(root, run_id, summaries, name)\n"
            "    fig.axes[0].set_title('LOCAL FIXTURE ONLY: one ply before the game cap')\n"
            "    fig.axes[0].set_xlabel('Score (%); intervals unavailable with one opening pair')\n"
            "    plots.save(fig, root, run_id, name)\n"
            "    return fig\n"
            "plots.plot_matches = fixture_plot\n"
        )
        for source in sorted((ROOT / "notebooks/no_rl").glob("0[1-4]_*.ipynb")):
            book = nbformat.read(source, as_version=4)
            book.cells[0].source = (
                "# LOCAL BOUNDED VALIDATION COPY\n\n"
                "Drive and install cells are replaced. Evaluation uses deliberately near-cap fixtures, "
                "not complete-game strength tests. Notebook 02 skips training. Temporary paths in outputs "
                "are not the production Colab paths.\n\n" + book.cells[0].source
            )
            book.cells[1].source = 'print("Local validation only: no Drive mount.")'
            book.cells[3].source = setup
            for cell in book.cells:
                if (
                    cell.cell_type == "code"
                    and "opening_manifest = prepare_openings" in cell.source
                ):
                    cell.source = (
                        'print("Local validation substitutes one-ply-to-cap opening fixtures.")'
                    )
            book.metadata.kernelspec = dict(
                name="chess-research-validation",
                display_name="Local Chess Validation",
                language="python",
            )
            print("Executing local adaptation:", source.name, flush=True)
            try:
                NotebookClient(
                    book,
                    timeout=300,
                    kernel_name="chess-research-validation",
                    resources={"metadata": {"path": str(project)}},
                ).execute()
            finally:
                nbformat.write(book, output / source.name)
            html, _ = HTMLExporter().from_notebook_node(book)
            (output / source.with_suffix(".html").name).write_text(html)
        plots = project / "results/no_rl_local_fixture/plots"
        if plots.exists():
            shutil.copytree(plots, output / "fixture_plots", dirs_exist_ok=True)
    print("Saved labelled local notebook/HTML outputs and fixture plots:", output, flush=True)


if __name__ == "__main__":
    main()
