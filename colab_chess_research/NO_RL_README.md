# No-RL Colab Notebooks

This is a separate notebook sequence. It does not require the RL notebooks, a league, or self-play training.

## Exact Local Location

```text
D:\Benjamin Data\Introduction to AI\Chess Agent Competition\colab_chess_research\notebooks\no_rl
```

Start with `01_classical_baseline.ipynb` in that folder.

The shared code and configuration are in the outer `colab_chess_research` folder, beside `notebooks`. An existing nested `colab_chess_research\Chess` copy is not the updated source location; it has been left untouched.

## Exact Google Drive Location

Extract the supplied `colab_chess_research_no_rl.zip`. Put its `Chess` folder under:

```text
MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/
```

The resulting notebook folder must be:

```text
MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess/notebooks/no_rl/
```

The Colab runtime root is:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess")
```

Upload the project contents, not only the `.ipynb` files. The notebooks import the shared `chess_rl` helpers. Check that `Chess/chess_rl/non_rl.py` and `Chess/configs/no_rl.yaml` exist. Do not create `Chess/Chess/chess_rl`.

These files were created locally. They have not automatically been uploaded to your Drive. Do not overwrite newer Drive work without reviewing the differences.

## Version A: Classical, No Neural Training

Leave `configs/no_rl.yaml` at its defaults:

```yaml
run_id: no_rl_classical_v1
device: auto
non_rl:
  variant: classical
  final_games: 16
```

Use a Colab **CPU** runtime. Open and run these notebooks in order:

1. `01_classical_baseline.ipynb`: environment, existing evaluator, search, correctness, opening fixtures.
2. Skip `02_optional_supervised.ipynb`.
3. `03_evaluate_no_rl.ipynb`: tune evaluation coefficients through games, then play greedy, minimax, and the original classical agent.
4. `04_freeze_no_rl.ipynb`: freeze this candidate and run held-out games.
5. `05_export_no_rl.ipynb`: final packaging and local submission checks.

There is no model to train or checkpoint to wait for. The deployed classical agent imports only standard library and python-chess. Notebook tooling uses additional installed libraries, but those are not shipped in the classical submission.

## The Four-Step Classical Workflow

1. Tree search: iterative-deepening negamax with alpha-beta pruning, quiescence and a transposition table in `chess_rl/search.py`.
2. Fundamentals: material, piece-square placement, attacked-square activity, bishop pair, doubled/isolated pawns and passed-pawn advancement in `chess_rl/classical_evaluation.py`.
3. Compare continuations: sum each feature multiplied by its coefficient, score from the side-to-move perspective, and use `tanh(score / 600)` at search leaves. This bounded score is not a calibrated win probability. Search handles terminal outcomes separately.
4. Test and adjust: notebook 03 runs the seeded, match-based parameter search in `chess_rl/evaluation_tuning.py`. This is numerical coefficient tuning, not reinforcement learning.

Notebook 01 displays each feature's contribution for an example position. Defaults preserve the previous evaluator's scores. Activity counts attacked squares, not legal-move mobility; material and piece-square coefficients multiply the existing piece values/tables, rather than learning every piece value or table entry independently.

### Automatic Weight Tuning

The supplied `configs/no_rl.yaml` enables tuning for the classical variant:

- Eight candidate coefficient sets, sampled reproducibly from `non_rl.tuning.ranges`.
- 32 games per candidate against the fixed starting engine, with colours swapped for each opening.
- Only the best usable screening candidate above 50% proceeds to a 128-game confirmation match on unused development openings.
- Accept only if confirmation score is at least 55%, the paired-bootstrap lower 95% bound exceeds 50%, and neither side has recorded runtime failures. Otherwise retain the original coefficients.
- Material scale stays fixed by default. The seven other coefficient ranges are configurable.

This uses 16 screening and 64 confirmation openings, all from the development suite. Held-out openings are excluded from tuning and reserved for notebook 04. Subsequent development comparisons are not independent confirmation of a tuned candidate; held-out results provide the separate final measurement. The defaults and ranges are starting experiment settings, not established optimal values, and the procedure does not guarantee improvement.

### Manual Changes and Saved Results

Edit `search.evaluation_weights` in `configs/no_rl.yaml` before starting a new run. Set `non_rl.tuning.enabled: false` to evaluate those coefficients directly without automatic tuning. Always choose a new `no_rl_` run ID after changing code, settings or inputs; do not overwrite a completed selection.

Notebook 03 saves under `results/<run_id>/no_rl/weight_tuning/`:

- `manifest.json`: fixed configuration, seed, source hashes, opening partitions and planned candidates.
- `trials.csv` and `trial-*.json`: coefficients, observed game scores and eligibility.
- `selected_weights.json`: the selected evaluation coefficients, including the unchanged baseline if no improvement is confirmed.
- `selection.json`: the screening/confirmation decision and supporting results.

Full PGNs and logs are under `results/<run_id>/matches/weight-*/`. Interrupted matches resume completed games only with matching inputs. Notebooks 03, 04 and 05 automatically use the selected coefficients; do not copy them back into the configuration during that run. The exported runtime's `config.json` carries them. No `.pt` model is required.

## Version B: Supervised, Still No RL

This is optional. It uses the existing neural policy/value model with alpha-beta search, trained on labelled positions without self-play. It is not a linear-regression tuner.

Change `configs/no_rl.yaml` before beginning a separate run:

```yaml
run_id: no_rl_supervised_v1
device: auto
non_rl:
  variant: supervised
  final_games: 16
```

Run **01 -> 02 -> 03 -> 04 -> 05**. Select GPU/A100 for notebook 02 when available. Other notebooks evaluate the agent on CPU. CUDA is not required for the classical version.

Notebook 02 reuses the established dataset, model, training, checkpoint and plotting implementations. It saves under `checkpoints/supervised/no_rl_supervised_v1/`. It does not read `checkpoints/self_play` or call the RL league. Notebook 03 also evaluates a no-network control for the supervised variant.

The classical coefficient tuner is skipped for the supervised variant. Its policy/value training remains a separate optional path.

The architecture and training defaults are inherited from `configs/default.yaml`; add experiment overrides to `configs/no_rl.yaml`. For example, a `training` section can override epochs or batch size. Choose a new `no_rl_` run ID whenever settings, source code, or selected weights change.

## What Is and Is Not Added

Both variants use iterative-deepening alpha-beta, quiescence, legal fallbacks and a transposition table. The classical evaluator now exposes configurable coefficients with the same default scores as before.

This notebook set does not implement every chapter of a chess strategy book, add opening books/tablebases, add a regression-based tuner, or claim that the optional neural model has learned any particular named strategy. The original starter agent, harness and five RL notebooks are preserved. The shared search/evaluation helpers gain configurable coefficients; original RL configurations still use the preserved defaults.

## Results and Submission

- Match PGNs, logs and CSVs: `Chess/results/<run_id>/matches/`.
- Development/selection records: `Chess/results/<run_id>/no_rl/`.
- Charts: `Chess/results/<run_id>/plots/`.
- Final ZIP: `Chess/exports/no_rl/<run_id>/<variant>/submission.zip`.

Classical example:

```text
Chess/exports/no_rl/no_rl_classical_v1/classical/submission.zip
```

Only notebook 05 executes submission-related checks. Earlier checks are ordinary research correctness and game evaluation. The original RL final-selection gate remains unchanged; the no-RL notebooks have an independent gate that records classical training as `not_applicable`.

The portable project ZIP is not the competition submission. Actual platform acceptance remains authoritative.

## Parallel Runs

Within one run, follow the sequence above. Notebooks 03 and 04 must not run against files still being trained or edited. Notebook 05 follows completed selection. Do not run two copies of the same writing notebook against the same run ID.

Independent runs can run in parallel only with separate run IDs and configuration files. The supplied notebooks all load `no_rl.yaml`, so use separate project copies or deliberately change their setup to load separate config files. Do not edit a shared `no_rl.yaml` while another run depends on it. Create the shared opening fixtures once in notebook 01 before starting parallel jobs.

## Validation Status

See `docs/NO_RL_VALIDATION.md` for executed checks and limitations. No full supervised training, full-size tournament, or final submission export has been run just by creating these notebooks.
