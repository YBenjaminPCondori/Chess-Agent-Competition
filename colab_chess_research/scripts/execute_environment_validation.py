"""Execute notebook 01 locally with explicitly replaced Drive/setup cells."""

from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "notebooks/01_environment_and_encoding.ipynb"
notebook = nbformat.read(source, as_version=4)
notebook.cells[0].source += (
    "\n\nLOCAL VALIDATION COPY: Drive mount and pip setup were replaced. "
    "Execution used the workstation project and an isolated CPU Python environment, not Colab."
)
notebook.cells[1].source = 'print("Local validation: no Drive mount performed.")'
notebook.cells[3].source = (
    "from pathlib import Path\nimport sys\nimport subprocess\n"
    f"PROJECT_ROOT = Path({str(ROOT)!r})\n"
    "sys.path.insert(0, str(PROJECT_ROOT))\n"
)
notebook.metadata.kernelspec = dict(
    name="chess-research-validation", language="python", display_name="Local Chess Validation"
)
destination = ROOT / "docs/validation"
destination.mkdir(parents=True, exist_ok=True)
print("Executing the local adaptation of notebook 01.", flush=True)
try:
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="chess-research-validation",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
finally:
    nbformat.write(notebook, destination / "01_environment_local.ipynb")
html, _ = HTMLExporter().from_notebook_node(notebook)
(destination / "01_environment_local.html").write_text(html)
print("Saved executed notebook and HTML under docs/validation.", flush=True)
