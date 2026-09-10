# Validation Record

Recorded on 2026-09-10. These checks validate research code, not competition submission acceptance or playing strength.

## Checks Completed

| Check | Actual result |
| --- | --- |
| Correctness and bounded integration suite | 39 tests passed in 15.72 seconds |
| Ruff static checks | Passed |
| Five production notebooks | Valid notebook structures; all code cells parsed; first code cell mounts Drive |
| Notebook 01 execution | A clearly labelled local adaptation completed, with Drive mounting and package installation replaced by local setup |
| Reference preservation | All 18 recorded snapshot hashes matched; corresponding original starter source files were unchanged |

The test suite covers legal action encoding, promotions, castling, en passant, pins, terminal positions, draw and clock rules, search board restoration, transposition contexts, legal-only losses, finite gradients, frozen inference, checkpoint recovery, dataset separation and import, and referee agreement.

Bounded workflow checks include one epoch on temporary fixture data, checkpoint resume, a short self-play episode, and two short near-cap games through the original process harness. These are implementation checks, not a trained result, a strength benchmark, or a tournament. Temporary fixture weights are not supplied as trained models.

The saved notebook 01 execution ran when the suite contained 37 tests. Two integration tests were subsequently added and the complete 39-test suite passed separately. Its inspectable outputs are:

- [Executed local notebook](validation/01_environment_local.ipynb)
- [Local execution HTML](validation/01_environment_local.html)

This execution did not mount Google Drive, use a Colab runtime, or establish that notebooks 02-05 have executed successfully. Notebook structure and code-cell parsing do not replace execution of the full workflow. Training plots have not been rendered against real training data because no full training run was launched.

## Commands Run

From the project directory, using an isolated local Python environment:

```bash
/tmp/chess-research-venv/bin/python -m pytest -q
/tmp/chess-research-venv/bin/ruff check chess_rl tests scripts templates
/tmp/chess-research-venv/bin/python scripts/validate_notebooks.py
/tmp/chess-research-venv/bin/python scripts/execute_environment_validation.py
```

The notebook execution also used the installed `chess-research-validation` IPython kernel. That kernel and the `/tmp` environment are workstation validation utilities, not Colab prerequisites. Follow the main README for Colab setup.

Local checks used Python 3.12, PyTorch 2.14.0+cpu, chess 1.11.2, NumPy 2.5.3, and pytest 9.1.1. This is not the documented platform package set. In particular, a local CPU check with PyTorch 2.14 does not establish export compatibility with the recorded platform PyTorch 2.13 runtime. Colab preserves its installed CUDA PyTorch instead of replacing it with a CPU wheel.

## Not Executed

- Full engine labelling, supervised training, self-play training, or hyperparameter studies.
- CUDA execution, A100 training, mixed-precision overflow handling, or hardware OOM recovery.
- Tournament-scale comparisons, promotion decisions, ablations, or final held-out evaluation.
- Notebooks 02-05 as end-to-end Colab runs.
- CPU deployment export, ONNX conversion, INT8 conversion, standalone deployment benchmarks, or submission ZIP checks.
- Competition server upload, acceptance, or platform resource-isolation checks.

Final export and submission-related checks are intentionally implemented only in notebook 05 and require completed model selection. No competition submission or trained champion is claimed here. The portable `colab_chess_research.zip` contains research source and notebooks, not an agent submission.

## Remaining Research Limits

Generated legal-playout opening suites are reproducible fixtures, not a claim of balanced openings or a reconstruction of the hidden platform distribution. A single FEN cannot recover earlier repetitions. Colab CPU speed and memory behavior cannot certify performance inside the platform container. Consult the environment contract and research design for these boundaries.
