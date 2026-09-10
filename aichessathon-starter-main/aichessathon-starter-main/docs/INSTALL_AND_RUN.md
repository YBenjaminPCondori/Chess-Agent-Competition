# Install And Run Guide

This document explains how to set up the current AI Chessathon starter repo, run the integrated `agent.py`, test it against baselines, and build `submission.zip`.

It assumes this project location:

```text
D:\Benjamin Data\Introduction to AI\Chess Agent Competition\aichessathon-starter-main\aichessathon-starter-main
```

In WSL or Git Bash, the same path is usually:

```text
/mnt/d/Benjamin Data/Introduction to AI/Chess Agent Competition/aichessathon-starter-main/aichessathon-starter-main
```

## What Needs Installing

Required:

- Git, if the repo still needs to be cloned or updated.
- `uv`, the Python project/dependency manager used by this repo.
- Python 3.12. The repo is pinned to Python `==3.12.*`; `uv` can install/manage this if it is missing.

Optional:

- `make`, if you want to use Makefile shortcuts like `make play`.
- A normal terminal: PowerShell, Windows Terminal, WSL, or Git Bash.

GPU/CUDA/HPC is not needed for this current classical-search agent. The competition runtime is CPU-only, so local setup should prioritize matching the Python/package environment.

## Step 1: Open The Repo

### WSL Or Git Bash

```bash
cd "/mnt/d/Benjamin Data/Introduction to AI/Chess Agent Competition/aichessathon-starter-main/aichessathon-starter-main"
```

### Windows PowerShell

```powershell
cd "D:\Benjamin Data\Introduction to AI\Chess Agent Competition\aichessathon-starter-main\aichessathon-starter-main"
```

Confirm you are in the right directory:

```bash
ls
```

You should see files such as:

```text
agent.py
pyproject.toml
uv.lock
Makefile
harness/
baselines/
docs/
```

## Step 2: Install uv

Skip this step if `uv --version` already works.

### WSL, Linux, macOS, Or Git Bash

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal, then check:

```bash
uv --version
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell, then check:

```powershell
uv --version
```

Alternative Windows install methods are also supported by uv, including WinGet:

```powershell
winget install --id=astral-sh.uv -e
```

Reference: https://docs.astral.sh/uv/getting-started/installation/

## Step 3: Install Python 3.12 If Needed

The project requires Python 3.12 exactly:

```text
requires-python = "==3.12.*"
```

Usually, `uv sync` will resolve this automatically. If Python 3.12 is missing, run:

```bash
uv python install 3.12
```

Then verify:

```bash
uv run python --version
```

Expected output should start with:

```text
Python 3.12
```

Reference: https://docs.astral.sh/uv/guides/install-python/

## Step 4: Sync Project Dependencies

From the repo root:

```bash
uv sync
```

This installs the dependencies pinned in `pyproject.toml` and `uv.lock`.

Current project dependencies include:

- `chess==1.11.2`
- `numpy==2.5.2`
- `numba==0.67.0`
- `torch==2.13.0`
- `onnxruntime==1.29.0`

The current `agent.py` only uses `python-chess` plus the Python standard library, but the repo keeps the wider allowed competition stack available for future experiments.

## Step 5: Confirm The Agent Imports

Run:

```bash
uv run python -c "import agent; print(agent.get_move('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 1000))"
```

Expected result:

- A UCI move string such as `e2e4`, `g1f3`, or another legal move.
- The exact move can vary after code changes.

## Step 6: Run One Local Game

Against the default greedy baseline:

```bash
uv run python -m harness.play --white . --black baselines/greedy
```

Against minimax:

```bash
uv run python -m harness.play --white . --black baselines/minimax
```

Using the Makefile shortcut:

```bash
make play
```

With a different opponent:

```bash
make play OPPONENT=baselines/minimax
```

If `make` is not installed, use the `uv run python -m ...` commands directly.

## Step 7: Run Multi-Game Tests

Default arena run:

```bash
uv run python -m harness.arena --opponent baselines/greedy --games 16 --base-ms 10000 --increment-ms 100
```

Equivalent Makefile shortcut:

```bash
make arena
```

Run against minimax:

```bash
make arena OPPONENT=baselines/minimax GAMES=16 BASE_MS=10000 INCREMENT_MS=100
```

Longer measurement run:

```bash
uv run python -m harness.arena --opponent baselines/minimax --games 64 --base-ms 10000 --increment-ms 100
```

Interpretation:

- A small number of games is only a quick signal.
- Use the same opponent, clock, and game count when comparing two agent versions.
- Do not tune conclusions from one or two games.

## Step 8: Run Static Checks

Syntax/import sanity:

```bash
uv run python -m py_compile agent.py
```

Lint:

```bash
uv run ruff check .
```

Type check:

```bash
uv run mypy
```

Combined Makefile target:

```bash
make gate
```

`make gate` runs static checks and an arena run. If local dev tools are missing, run the explicit `uv run ...` commands above.

## Step 9: Build The Submission Zip

Build:

```bash
uv run python -m harness.package
```

Or:

```bash
make zip
```

Expected output:

```text
submission.zip (... bytes, ... unzipped)
Package built. Run harness.play or harness.arena for game testing before upload.
```

The generated file should be:

```text
submission.zip
```

The zip must contain `agent.py` at the zip root.

## Step 10: Verify Zip Contents

Run:

```bash
uv run python -c "import zipfile; z=zipfile.ZipFile('submission.zip'); print(z.namelist())"
```

Expected:

```text
['agent.py']
```

If additional allowed project files are intentionally included later, they may also appear. For the current integrated baseline, `agent.py` at root is the key requirement.

## Step 11: Upload

Upload this file to the competition dashboard:

```text
submission.zip
```

The platform validation log is the authority. Local runs are for catching obvious errors and comparing candidate strength before spending uploads.

## Normal Team Workflow

For each candidate change:

```bash
uv run python -m py_compile agent.py
uv run ruff check .
uv run mypy
uv run python -m harness.arena --opponent baselines/greedy --games 16 --base-ms 10000 --increment-ms 100
uv run python -m harness.arena --opponent baselines/minimax --games 16 --base-ms 10000 --increment-ms 100
uv run python -m harness.package
```

For a cleaner experiment record, write down:

- Git commit or file version.
- Opponent baseline.
- Game count.
- Clock settings.
- Score.
- Any crashes, flags, or illegal moves.

## Troubleshooting

### `uv: command not found`

Install uv, restart the terminal, and run:

```bash
uv --version
```

If the command still fails, the uv install directory is probably not on `PATH`.

### Wrong Python Version

Run:

```bash
uv python install 3.12
uv sync
uv run python --version
```

The version should be Python 3.12.

### `ModuleNotFoundError: No module named 'chess'`

Run commands through uv:

```bash
uv sync
uv run python -c "import chess; print(chess.__version__)"
```

Avoid using plain `python` unless you know it points to the uv environment.

### `make: command not found`

Use the direct commands:

```bash
uv run python -m harness.play --white . --black baselines/greedy
uv run python -m harness.arena --opponent baselines/greedy --games 16 --base-ms 10000 --increment-ms 100
uv run python -m harness.package
```

### Path Problems On Windows

Use quotes around paths because the project directory contains spaces:

```powershell
cd "D:\Benjamin Data\Introduction to AI\Chess Agent Competition\aichessathon-starter-main\aichessathon-starter-main"
```

In WSL:

```bash
cd "/mnt/d/Benjamin Data/Introduction to AI/Chess Agent Competition/aichessathon-starter-main/aichessathon-starter-main"
```

### Zip Does Not Contain `agent.py` At Root

Rebuild from the starter repo root:

```bash
pwd
uv run python -m harness.package
```

Do not run the package command from the outer folder.
