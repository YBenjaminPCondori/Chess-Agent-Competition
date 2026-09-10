# No-RL Validation Record

## Executed Checks

- Full ordinary test suite: **65 passed in 36.09 seconds**, including 26 no-RL workflow/evaluation-tuning checks.
- Ruff static checks: passed for the shared code, templates, tests and scripts.
- All five no-RL notebooks: validated with nbformat; code cells parsed as Python; first executable cells mount Drive.
- Dependency checks: no no-RL notebook calls self-play/league training; only notebook 05 calls final export/compliance.
- Original five RL notebook files and preserved starter source hashes: unchanged.
- Default evaluator scores compared with the prior implementation over 240 seeded legal-playout positions: zero mismatches. Eight frozen scores are covered by persistent regression tests.
- Coefficient validation, search-leaf/continuation integration, deterministic search ranges, opening partition separation, saved selection propagation and resume behavior: passed.

An earlier test run exposed test isolation trouble: a temporary harness import was left in the interpreter and shadowed the real reference in later tests. The fixture was corrected to reuse the canonical harness. The harness and its import guard were not modified; the combined test run above passed.

## Local Notebook Execution

Labelled local adaptations of notebooks **01, 02, 03 and 04** executed using the isolated CPU Python environment. Drive mounting and pip installation were replaced with local setup. A temporary project copy was used; it has been removed after execution.

- Notebook 01 ran the classical search demonstration, feature-contribution display and 26 bounded no-RL tests.
- Notebook 02 exercised the default classical branch: all supervised training cells were skipped.
- Notebook 03 first ran one tuning trial with two actual process-harness games. Both were draws, so the baseline was retained and no confirmation match was warranted. Distinct screening, confirmation and held-out positions were supplied.
- Notebooks 03 and 04 each then played two games against each of three opponents through the original process harness, using positions exactly one ply before the event cap. They exercised record writing, resume/dependency logic, selected-coefficient propagation, freezing, and plotting.
- These fourteen deliberately short notebook games are **not strength measurements**. Their draws are the expected consequence of the fixture's cap, not evidence that the engines are equal in strength. The bounded tests also use separate temporary game fixtures.
- The positive-promotion branch uses a clearly marked synthetic summary in a unit test to verify confirmation selection and runtime configuration propagation. It is not a measured win rate or a trained/tuned candidate for use in competition.
- No checkpoint or league was needed for the classical path. A separate supervised-selection test used temporary fixture weights only, not a trained model.

Executed notebooks, HTML previews and explicitly labelled fixture plots are in `docs/validation/no_rl/`. The validation script replaces input paths, game counts and fixtures transparently; these are not claimed to be end-to-end Colab runs. Production notebooks remain unexecuted templates with the real Drive path and configured game counts.

## Not Executed

- Actual Google Drive mounting and package installation in a live Colab runtime.
- Supervised dataset generation/training, CUDA/A100 execution, or supervised full-game comparisons.
- Full-size coefficient-tuning, confirmation, classical development or held-out tournaments. No empirically improved coefficient set is claimed.
- No-RL notebook 05, final ZIP creation, deployment benchmarks, resource-isolation checks, or platform upload/acceptance.

Final submission checks remain exclusively in notebook 05 after the selected workflow finishes. A pure classical selection marks training as `not_applicable`; it does not fabricate a training-completion file. The portable project archive contains source and notebooks, not a validated agent submission.

## Reproduce the Local Checks

From the project root, in the already prepared isolated workstation environment:

```bash
/tmp/chess-research-venv/bin/python -m pytest -q tests
/tmp/chess-research-venv/bin/ruff check chess_rl tests scripts templates
/tmp/chess-research-venv/bin/python scripts/execute_no_rl_validation.py
```

The last command requires nbclient, nbconvert, matplotlib and the installed `chess-research-validation` IPython kernel. It runs bounded local adaptations only, never notebook 05. These workstation paths are not prerequisites for Colab; use `NO_RL_README.md` for Colab installation.

Visual inspection covers the rendered fixture comparison charts. The complete HTML previews have not been checked in a browser; open `docs/validation/no_rl/03_evaluate_no_rl.html` and `04_freeze_no_rl.html` to inspect the full saved presentation. The figures are produced from the executed fixtures, not invented training curves.
