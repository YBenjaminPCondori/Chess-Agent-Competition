# Colab Chess Research

A runnable research project for a policy/value network used by alpha-beta search. The existing starter agent and harness are preserved in the original repository. This directory is the portable Colab project, not a trained competition submission.

**No-RL version:** start in [`notebooks/no_rl/`](notebooks/no_rl/README.md). That independent sequence supports classical search with no training, plus an optional supervised-only variant. Exact paths and run order are in [NO_RL_README.md](NO_RL_README.md). The sequence below is the original self-play research workflow.

## Start in Colab

1. Place this project's contents in the Drive folder named **Chess**, at:
   `MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess`.
2. Open `notebooks/01_chess_environment_and_encoding.ipynb` using Google Colab.
3. For notebooks 02 and 03, select **Runtime > Change runtime type > GPU**. Choose A100 when available.
4. Edit `configs/default.yaml` for the run. The defaults implement the supplied research specification.
5. Run notebooks in order: **01, 02, 03, 04, 05**.

Detailed beginner instructions and parallel-run rules are in [NOTEBOOK_RUN_ORDER_README.md](NOTEBOOK_RUN_ORDER_README.md).

Every notebook's first executable cell mounts Drive. Every later input is loaded from disk; an earlier notebook does not need to stay open. The runtime root is exactly:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess")
```

If using the supplied portable project ZIP, extract its `Chess/` directory under `MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/`. A double folder such as `Chess/Chess/chess_rl` is incorrect. The correct code path ends in `Chess/chess_rl/model.py`.

## What Runs

| Notebook | Main work | Persistent result |
| --- | --- | --- |
| 01 | Rules, encodings, correctness tests, reserved opening suites | Opening fixtures and their hashes |
| 02 | Dataset import or generation, offline teacher labelling, supervised training | Dataset manifest, best validation checkpoint, initialization choice |
| 03 | Frozen-teacher self-play, replay updates, CPU promotion games | Candidate checkpoints and retained league champion |
| 04 | CPU opponent comparisons, ablations, frozen held-out evaluation | Match records and final selection manifest |
| 05 | CPU export, isolated candidate, artifact checks, final games | Exported weights, submission ZIP and local report |

Notebook 02's default 100,000 positions are actual requested data generation, not a sample download hidden behind a label. With no supplied PGN/JSONL, legal random playouts provide positions and a training-only Stockfish labels them. These positions are not claimed to be a high-quality or balanced human game corpus. Provide better PGNs through the config when conducting corpus-quality experiments.

The default model has **2,335,370 parameters**: six 128-channel residual blocks, a 4,672-logit policy head, and one scalar value head. Values are from the side to move; the policy only assigns probability to legal actions.

## Project Layout

- `notebooks/`: five executable workflow notebooks.
- `chess_rl/`: shared rules, encoders, model, search, datasets, training, league, evaluation, export and plotting.
- `configs/`: defaults, optional overrides and search spaces. Notebooks use `default.yaml` unless deliberately changed to load an override.
- `tests/`: ordinary correctness and bounded integration tests.
- `reference/`: byte-preserved harness/baselines and current team agent snapshots.
- `templates/`: research and final agent wrappers, materialized by the workflow.
- `docs/`: research decisions, encoding contract, source hashes and validation status.
- `datasets/`, `checkpoints/`, `logs/`, `results/`: real run artifacts created by notebooks.
- `exports/`, `submission_candidate/`: final artifacts produced by notebook 05 only.

## Configuration and Resume

Use one unique `run_id` for each independent experiment. Reusing an existing run with different training settings raises an error instead of silently mixing results. The default supervised run is 20 epochs maximum, with early stopping. CUDA uses BF16 when supported, otherwise FP16 with scaling; CPU uses FP32.

Supervised checkpoints resume at the end of the last completed epoch. A partial epoch may be repeated after a disconnect. Self-play saves completed games separately and optimizer state every 250 updates. Interrupted games are restarted, not labelled as draws. An OOM before the optimizer step can reduce microbatch size while keeping the effective batch; this changes BatchNorm's batch statistics and is logged. An OOM during the optimizer step requires checkpoint recovery.

The checkpoint roles differ: `latest.json` is for resuming, `best_validation.json` selects supervised validation loss, and `league.json` records the playing champion. CPU loading of the project's full training checkpoint is supported. Only load trusted project checkpoints: complete RNG/optimizer snapshots use Python serialization. Final deployment uses its separate minimal weight format.

## Evaluation and Compliance

Ordinary tests, validation curves and CPU development matches run during research. Submission checks run only in notebook 05, after the final selected run is complete. The default final candidate uses CPU PyTorch and one thread. Colab CPU timings are not measurements of the platform machine.

The candidate ZIP is generated under `exports/<model_version>/submission.zip`. It contains the adapted `agent.py` at its root, `chess_runtime/`, `weights/`, `config.json`, and `model_manifest.json`. The portable project ZIP is a different archive and must not be uploaded as the agent submission.

No training, tournament-scale evaluation, submission export, or server upload has been performed just by generating this project. See [VALIDATION.md](docs/VALIDATION.md) for the checks actually executed.

## Sources

The supplied specification is preserved in `docs/REQUEST.md`. The project was built from the local starter and the [official competition documentation](https://aichessathon.com/docs). The source snapshot and version record is `docs/source_manifest.json`. A FEN does not supply repetition history, and the hidden rated opening set is unavailable; both limits are documented in [ENVIRONMENT_CONTRACT.md](docs/ENVIRONMENT_CONTRACT.md).
