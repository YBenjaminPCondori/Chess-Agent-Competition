# No-RL Notebook Entry Point

Start with **01_classical_baseline.ipynb** in this folder.

- Classical only: **01 -> 03 -> 04 -> 05**. No neural training or GPU is needed. Notebook 03 tunes evaluation weights through games, then compares the selected engine against baselines.
- Optional supervised model, no RL: **01 -> 02 -> 03 -> 04 -> 05**.

Settings: `../../configs/no_rl.yaml`.

Change `search.evaluation_weights` for manual coefficients, or `non_rl.tuning.ranges` for the automatic search. Selected weights and trial scores are saved in `results/<run_id>/no_rl/weight_tuning/` under the project root. Use a new run ID for changed inputs.

See [NO_RL_README.md](../../NO_RL_README.md) for exact Windows/Drive paths, installation, outputs, and parallel-run rules.

The original RL notebooks are one folder above. They are a different sequence; do not mix their notebook 03 or final export with this sequence.
