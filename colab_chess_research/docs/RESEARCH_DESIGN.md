# Research Design

## Objective

Train a policy/value network from scratch and use it inside the team's alpha-beta search. The policy orders every legal candidate; it does not discard candidates. The value estimates a side-to-move outcome-like score. This is approximate policy iteration with search-generated targets, not PPO, DQN, MCTS, or a claim of AlphaZero replication.

## Architecture and Evaluation

The default network uses a 21-plane input, 128-channel stem, six residual blocks, and two heads. Policy logits have shape [73,8,8]; the value head uses 32 feature channels, a 256-unit hidden layer and tanh. The default has 2,335,370 parameters. Exact layers and variant construction are in model.py; board/action conventions are fixed by ENCODING_SPEC.md.

The team's original piece values and piece-square constants are extracted without changing the original agent. Classical evaluation includes material, those tables, activity, pawn structure and the bishop pair. Its centipawn score is transformed by tanh(cp/600). By default the searched leaf is 0.5 * classical_value + 0.5 * neural_value. Mate scores use a separate magnitude of 10,000 with distance preference.

The original pawn/piece-square definitions are retained as the baseline's evaluation choices, not declared chess truth or optimally tuned parameters. Their empirical quality is a future evaluation question.

Search uses iterative deepening, full legal candidate ordering, alpha-beta pruning, quiescence, conservative time allocation, and a bounded transposition table. Search history, counters and model version enter cache identity. Every search push has a finally-pop. Regular search keeps the first strict best improvement: a fail-low bound equal to the current best is not mistaken for an equally good exact score.

## Data and Supervised Learning

Data can be imported from PGN or compatible JSONL. The default fallback corpus consists of generated legal playouts, labelled by a separately installed training-only Stockfish. Teacher settings are 50,000 nodes, MultiPV 1, one thread and 128 MB hash. Binary provenance and analysis depth/node information are recorded. An explicitly selected team-classical teacher is labelled separately.

Source games are assigned deterministically to approximately 80/10/10 train/validation/test groups before sampling. Position duplicates across groups are removed. Reserved development and held-out positions are excluded. Hash-based grouping gives approximate rather than exact percentages, so the manifest reports actual resulting counts. Validation data never enters self-play replay.

The baseline objective is legal policy cross-entropy plus Huber value loss, with equal weights. AdamW starts at 3e-4, weight decay 1e-4, gradient norm limit 1, and cosine scheduling. Training runs up to 20 epochs with early stopping. The selected checkpoint is the actual minimum validation loss; minimum-delta early stopping uses a separate progress threshold. Validation CE is unsmoothed so HPO smoothing choices cannot alter the comparison target.

Outcome labels are +1 for a win by the state's side to move, 0 for a draw, and -1 for a loss. Discount is 1. Engine centipawn targets are transformed heuristics and are identified separately. Expected win-minus-loss is not identical to win probability, and an engine annotation is not a calibrated result probability. Reward shaping is off.

## Self-Play Improvement

Each iteration freezes the champion for game collection. A full-window search for each legal root move produces a common completed-depth score set. Its normalized-score softmax, with temperature 0.25, becomes the policy target. A separate temperature and epsilon determine the played move. Proven positive mates prefer the shortest mate. Incomplete analyses have no policy target.

The league samples champion mirrors, previous/older champions, the preserved classical bot, and greedy/minimax. Missing historical buckets are renormalized and logged. Completed normal games supply outcome labels. Crashes, flags, init failures and interrupted games are logged but excluded from gradient data by default.

The replay buffer retains at most 500,000 positions. Each update samples 25% original supervised training data and 75% self-play records. Each iteration trains for 2,000 updates with freshly initialized AdamW at 1e-4. Update checkpoints retain sampler RNG, optimizer, model, and replay/league references for resumption.

An inference call itself cannot be interrupted by a Python deadline. The search checks every node and before/after inference; if predicted call latency exceeds available time it falls back to classical evaluation. Budget/model fallbacks are logged. Using a deeper network can reduce completed search depth and must be measured in matches.

## Selection and Uncertainty

Development matches use colour-swapped opening pairs, 256 games per matchup at the official clock. Promotion requires score >= 55%, lower paired 95% bootstrap bound > 50%, no void games, no candidate operational failures, and no recorded model errors. Bootstrap uses 10,000 resamples of source groups, preserving pairs.

Repeated development selection can overfit that suite. The held-out suite is locked to a selected checkpoint and search configuration, then used for 512 games per selected matchup. It is not fed back into tuning. These research fixtures are legal playouts and do not certify strength on the hidden platform distribution.

## Colab and Reproducibility

Training uses Colab CUDA if available, BF16 on supported GPUs or scaled FP16 otherwise. CPU uses FP32. The runtime is CPU-only with one PyTorch thread. No claim is made that a Colab CPU benchmark reproduces the platform EPYC machine.

Seeds cover Python, NumPy, Torch, workers, game sampling and HPO. Checkpoints record resolved configs, versions, encoder IDs and data hashes. Immutable checkpoint files have checksums; role pointers are updated separately. Atomic filename replacement reduces partial-read risk but is not a claim that Drive provides transactional storage.

A partial supervised epoch is replayed from the last complete epoch. During OOM recovery, microbatching can alter BatchNorm statistics even when the effective gradient batch stays constant. Reproducibility across devices or dependency versions is not promised.

## References and Status

The inspected presentation references were Part_C_PPO_Environment.ipynb and Part_C_PPO_Training.ipynb in the user's Education notebook directory. Their section/plot workflow informed presentation. Their hospital environment and PPO reward/transition logic were not reused.

The official [competition docs](https://aichessathon.com/docs), local README/AGENTS/IDEAS, agent, baselines and process harness were inspected. The Markdown contract endpoints were unavailable during the earlier rules review; the main docs page was used. Exact source hashes and versions are recorded in source_manifest.json.

Read VALIDATION.md for checks actually executed. Final artifact compliance is implemented but is not run until notebook 05 has genuine completed training/evaluation artifacts. Generated project materials are not evidence that a trained model exists.
