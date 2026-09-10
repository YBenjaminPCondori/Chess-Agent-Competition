# Final Colab Chess Research Prompt

Act as a reinforcement learning researcher, senior software engineer, and chess-engine architect.

Build a runnable Google Colab research project for AI Chessathon. Implement the project, including shared Python modules, notebooks, configuration files, documentation, and meaningful correctness tests. Do not stop at a design outline, empty functions, pseudocode, or notebooks that merely refer to helpers that do not exist.

The research method is a hybrid of classical alpha-beta search and a jointly trained policy/value network. Use supervised learning on engine-labelled positions, followed by search-guided self-play and league training. Do not introduce PPO, DQN, or MCTS into the default implementation. Explain the actual learning procedure without implying that self-play itself specifies an algorithm or guarantees improvement.

Ignore competition deadline pressure and financial constraints when designing the research. Treat the architecture and numerical defaults below as explicit starting hypotheses, not official recommendations or proven optimal settings.

## 1. Workspace, References, and Boundaries

The Colab runtime project root must be exactly:

```python
from pathlib import Path

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/Education/Deep Reinforcement Learning/Chess"
)
```

All project-owned source, notebooks, configs, datasets, checkpoints, exports, logs, and results must persist under this root. Installed system packages and Colab's own runtime files are outside this requirement.

Existing Windows workspace:

```text
D:\Benjamin Data\Introduction to AI\Chess Agent Competition
```

Existing starter repository:

```text
D:\Benjamin Data\Introduction to AI\Chess Agent Competition\aichessathon-starter-main\aichessathon-starter-main
```

Notebook presentation references:

```text
D:\Benjamin Data\GitHub\Education\Deep_Reinforcement_Learning\notebooks
```

For WSL, these drives map to `/mnt/d/`. Inspect accessible reference notebooks and reuse their sectioning, markdown/code balance, explanation style, plots, and training workflow presentation. Their environments, actions, rewards, and PPO implementations are not scientific specifications for this chess project. If the reference files are unavailable, record that limitation and proceed with a clear notebook structure.

Read the starter repository's `AGENTS.md`, `README.md`, `docs/IDEAS.md`, current `agent.py`, relevant baselines, and harness before designing the adapter. Preserve the existing competition `agent.py`, harness, baselines, rules, Makefile, and packaging scripts. Preserve unrelated work.

Use `chess_rl/` as the shared importable package, directly under the project root. Do not add a second competing `src/chess_rl/` layout. Notebooks should orchestrate reusable modules rather than duplicate their implementations.

If authoring on the local Windows/WSL machine, create the portable project in a new `colab_chess_research/` directory under the existing workspace. Its contents correspond to the contents of `PROJECT_ROOT`. Supply a portable project archive and precise Drive placement instructions. Do not create a fake local `/content/drive` mount or claim that local files are already on Google Drive. Training must be designed for Colab; do not launch a full local training run.

## 2. Source Authority and Final-Only Compliance

Read the current official documentation when defining the environment:

- https://aichessathon.com/docs
- https://aichessathon.com/docs/agent-contract.md
- https://aichessathon.com/docs/rules.md
- https://github.com/advitrocks9/aichessathon-starter

If a Markdown endpoint is unavailable, use the main documentation page. Record source URLs, retrieval date, package versions, and the starter commit or source-file hashes in the research design. Surface disagreements between the official wording and the local harness instead of silently changing the harness. The official platform remains authoritative.

The currently documented submission contract includes:

| Item | Requirement |
| --- | --- |
| API | `get_move(fen: str, time_left_ms: int) -> str` |
| Output | Legal UCI move |
| Runtime | Python 3.12; one CPU core; 2 GB RAM; no GPU or network |
| Clock | 120,000 ms plus 500 ms after each legal move |
| Initialization | 90 seconds before the game clock |
| Submission | `agent.py` at archive root; at most 50,000,000 uncompressed bytes |
| Filesystem | Read-only except `/tmp`, with a 256 MB allowance |
| Dependencies | Standard library and the officially preinstalled packages only |

Record the exact current runtime package versions separately from compatible Colab training versions. Do not blindly replace Colab's CUDA PyTorch with the platform's CPU wheel.

The default submission uses only standard library, `chess`, and CPU PyTorch. Training utilities may have additional dependencies. Team-trained weights and offline engine-labelled data are permitted; shipped third-party engines, published chess weights or derivatives, and middlegame answer tables are excluded. Keep books and tablebases disabled in this implementation; document them only as optional future work subject to the current rules and size limit.

Submission compliance execution belongs exclusively in `05_final_export_and_compliance.ipynb`, after training and model selection finish. Do not add submission gates, ZIP audits, export compatibility checks, or standalone deployment benchmarks to every epoch, experiment, or earlier notebook.

Reading the environment specification, testing the chess implementation, monitoring losses, and evaluating games are ordinary research work and should happen before final export. Do not defer basic correctness until after training. Do not describe these tests as submission acceptance.

## 3. Exact Chess Environment

Use full standard chess, `chess.Board`, legal UCI moves, and legal `board.push(move)` transitions. Do not build a grid-world, restrict the pieces, or invent movement constraints. The rules engine handles rooks, bishops, knights, pawns, promotions, castling, checks, pins, and en passant.

Separate the referee's complete game state from the agent's observation:

- Referee state: board, move history since episode initialization, clocks, termination status, and initial FEN.
- Agent observation: FEN plus that agent's remaining milliseconds, exactly as the public API supplies.
- The opponent's clock, referee history, engine annotations, and future outcomes are not policy inputs.

Initialize episodes from valid FENs with a fresh move stack. Do not invent pre-FEN repetition history. Preserve the supplied FEN counters as the inspected harness does. If the official fifty-move initialization wording conflicts with this behavior, document the discrepancy and preserve the harness reference rather than silently normalizing arbitrary FENs.

Mirror the inspected referee's termination order before asking for a move:

```python
outcome = board.outcome()
if outcome is not None:
    # Use the returned winner and termination reason.
    ...
elif board.is_repetition(3):
    # Draw by current-position threefold repetition.
    ...
elif board.is_fifty_moves():
    # Draw at the current fifty-move boundary.
    ...
elif board.ply() >= 600:
    # Draw by the event cap.
    ...
```

This is illustrative ordering, not permission to leave placeholder implementation. Do not substitute `outcome(claim_draw=True)`: it can include a draw claim enabled by a possible next move. The 600-ply boundary is `board.ply()`, including the opening's FEN move number, not 600 additional moves after reset.

Deduct measured move wall time before adding increment. Match the reference handling of illegal output, initialization failure, crashes, and flag falls, including its insufficient-material condition for a draw on time. Record failed episodes separately from normal chess outcomes. Do not label infrastructure failures as checkmates.

Use the unmodified harness for CPU match evaluation. A GPU self-play collector may use a research adapter with the same transition, clock, and termination logic. Explicitly distinguish its training hardware from the tournament hardware. Do not claim that Colab reproduces platform CPU speed, resource isolation, or the unpublished opening distribution.

Include the starter's eight openings as visible examples. Create separately versioned development and held-out opening suites from valid games or legal playouts. Record their provenance and keep held-out suites out of training and teacher data. Do not describe the public openings as the complete rated environment.

## 4. Board Encoding: Version `board_v1`

Use `float32` tensors of shape `[21, 8, 8]`, batched as `[B, 21, 8, 8]`.

Use absolute White-oriented coordinates throughout: `tensor[channel, rank, file]`, with rank 0 corresponding to rank 1 and file 0 to file a. Do not rotate Black positions in this version.

| Plane | Meaning |
| --- | --- |
| 0-5 | White pawn, knight, bishop, rook, queen, king |
| 6-11 | Black pawn, knight, bishop, rook, queen, king |
| 12 | Constant 1 for White to move, 0 for Black |
| 13-16 | Constant White kingside, White queenside, Black kingside, Black queenside castling rights |
| 17 | One at the legal en-passant target square, otherwise zero |
| 18 | Constant `min(board.halfmove_clock, 100) / 100.0` |
| 19 | Constant `min(board.ply(), 600) / 600.0` |
| 20 | Reserved repetition feature, always zero in version 1 |

Castling-right planes describe rights, not whether castling is currently legal through occupied or attacked squares. Use `board.has_legal_en_passant()` to normalize plane 17 consistently with the harness's FEN serialization.

A single FEN cannot reconstruct repetition history. Keep plane 20 identically zero in imported data, self-play, validation, and runtime. The referee must still track repetition correctly. The baseline search may detect repetitions within its hypothetical search line, but does not claim knowledge of the actual pre-root history. Document this information limitation. Any future history-aware model requires a new encoder version and matching training/runtime history reconstruction.

## 5. Action Encoding: Version `action_v1`

Use 4,672 outputs with layout `[73, 8, 8]`. Define:

```python
from_square = rank * 8 + file
action_index = move_plane * 64 + from_square
```

This matches a contiguous flatten of the model's `[B, 73, 8, 8]` output. Provide reversible board-dependent move encoding and decoding. Fix the plane order:

- Planes 0-55: direction-major queen-like displacements, with direction order N, NE, E, SE, S, SW, W, NW and distances 1-7. `plane = direction_index * 7 + distance - 1`. These displacements also encode ordinary pawn and king moves when legal.
- Planes 56-63: knight `(delta_file, delta_rank)` offsets in this order: `(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)`.
- Planes 64-72: underpromotion piece order N, B, R; within each piece, file displacement -1, 0, +1. `plane = 64 + piece_index * 3 + displacement_index`. Promotion rank displacement is +1 for White and -1 for Black. File displacement is absolute, not ambiguously player-relative.

Queen promotion uses the normal displacement plane; decoding must add `promotion=chess.QUEEN` for a pawn reaching its final rank. Encode standard castling as the legal king's two-square displacement and en passant as the pawn displacement.

Build the legal mask solely by encoding `board.legal_moves`. An encoded illegal move must never be selected. Invalid indices or off-board displacements must not silently decode into legal moves. Handle terminal boards without applying an all-masked softmax.

Compute policy loss and sampling over legal entries only. If label smoothing is enabled, distribute the smoothing mass over legal actions only. Do not use stock smoothing across all 4,672 outputs with illegal logits set to negative infinity. Cross-entropy consumes logits or log probabilities as appropriate; do not apply a redundant softmax before a logits-based loss.

## 6. Exact Default Network

Implement `ChessPolicyValueNetSmall`, initialized from scratch, with a shared trunk and two heads.

```text
Input: [B, 21, 8, 8]

Stem:
  Conv2d(21, 128, kernel_size=3, padding=1, bias=False)
  BatchNorm2d(128, eps=1e-5, momentum=0.1)
  ReLU

Six residual blocks, each:
  Conv2d(128, 128, 3, padding=1, bias=False)
  BatchNorm2d(128, eps=1e-5, momentum=0.1)
  ReLU
  Conv2d(128, 128, 3, padding=1, bias=False)
  BatchNorm2d(128, eps=1e-5, momentum=0.1)
  Add input skip connection
  ReLU

Policy head:
  Conv2d(128, 73, kernel_size=1, bias=True)
  Flatten to [B, 4672] using action_v1
  Return raw logits; apply legal masking outside the model

Value head:
  Conv2d(128, 32, kernel_size=1, bias=False)
  BatchNorm2d(32, eps=1e-5, momentum=0.1)
  ReLU
  Flatten to [B, 2048]
  Linear(2048, 256)
  ReLU
  Linear(256, 1)
  Tanh

Default dropout: 0
```

Specify deterministic initialization under the run seed: Kaiming initialization for convolution weights, Xavier for linear weights, unit BatchNorm scale, zero biases. Record the parameter count and resolved architecture in every checkpoint. Constructor parameters must reconstruct every supported tuning variant without changing encoder conventions.

Inference uses `model.eval()` and `torch.inference_mode()`. BatchNorm statistics must not update during search or game collection.

## 7. Rewards, Values, and Search Scores

Use sparse chess outcomes with discount `gamma = 1.0`. For a state whose side to move is `c`, the completed-game target is +1 if `c` wins, 0 for a draw, and -1 if `c` loses. Nonterminal rewards are zero. Define the reward recipient explicitly; do not accidentally assign the winning mover's reward to the opponent after `board.push()` flips the turn.

This value estimates expected win-minus-loss outcome under the data-generating play, not a direct win probability or guaranteed perfect-play minimax value.

For engine annotations, request the score from the current side-to-move perspective and transform non-mate centipawns using `tanh(cp / 600.0)`. Map mate annotations to signed endpoints for training, while preserving their raw mate distances in metadata. Engine values are heuristic targets, not calibrated outcome probabilities. Keep source-specific metrics and preserve the distinction from completed-game results.

Reward shaping is disabled by default. Any later shaping experiment needs an explicit formula and a separate config; it must not replace true result labels or alter the referee's rules.

Make classical and neural evaluations comparable inside search:

```python
classical_value = math.tanh(classical_cp / 600.0)
leaf_value = (1.0 - value_eval_mix) * classical_value + value_eval_mix * neural_value
```

Use `value_eval_mix = 0.5` initially. All values are from the side to move. Negamax negates the child's returned score. Reserve a separate mate scale, such as `MATE_SCORE = 10_000`, with ply distance preference. Exact terminal results override learned and blended evaluations. Do not average mate scores with ordinary values or interpret alpha-beta bounds as exact teacher scores.

## 8. Search and Research Agent

Adapt the existing team-owned classical search into shared research modules while preserving the submitted repository agent. Implement:

- Piece values, piece-square tables, and explainable classical evaluation.
- Iterative-deepening negamax with alpha-beta pruning.
- Quiescence search for captures and promotions; all legal evasions when in check. Never use stand-pat while in check.
- Deterministic legal move ordering using the transposition-table move, tactical ordering, policy ranking, and a stable final tie-break.
- A bounded transposition table with depth, exact/lower/upper flags, mate-distance handling, and legal best-move validation.
- Rule-sensitive cache handling: do not reuse scores across incompatible halfmove clocks, ply-cap states, model versions, or repetition histories. Use sufficiently complete keys or disable score reuse when equivalence is not established.
- `try/finally` around every search push/pop.

Policy affects order, not the set of legal candidates. Defaults are `policy_top_k_root = "all"` and `policy_top_k_internal = "all"`. Ordinary alpha-beta cutoffs remain enabled. Candidate exclusion is a separately labelled future ablation, disabled in the default and tuning configurations supplied here.

Search defaults: maximum nominal depth 6, quiescence depth 4, and at most 100,000 transposition entries. If a quiescence limit is reached in check, process evasions under the overall deadline rather than treating check as a quiet position.

Choose a legal fallback before model inference or search. For `time_left_ms <= 100`, return it immediately. Use monotonic time, check deadlines at every node and before/after inference, and return the best move from the last completed iteration on timeout.

Initial time allocation in milliseconds:

```python
reserve_ms = max(10.0, min(50.0, 0.10 * time_left_ms))
usable_ms = max(0.0, time_left_ms - reserve_ms)
budget_ms = min(usable_ms, 0.05 * time_left_ms + 250.0, 1000.0)
```

Account for work since `get_move` entry rather than starting the clock after encoding. The increment is future time, not additional current time. A neural call is not preemptible: skip it when measured latency plus margin exceeds the remaining budget, and use classical evaluation. A deadline check cannot guarantee interrupting an already running PyTorch call.

Keep a research adapter exposing the exact public API for notebooks 3 and 4. It may load selected research checkpoints; notebook 5 produces the isolated shipping artifact. Both must use the same encoder and model definitions. Do not silently replace a requested neural candidate with the classical bot in reported neural evaluation results; record all fallback events.

## 9. Dataset Acquisition and Format

Provide real PGN/JSONL ingestion and an optional offline engine-labelling utility in the training code. Stockfish may be used as a training-only teacher, with recorded source, version, binary hash, and analysis settings. Its executable, wrappers, caches, and datasets never enter the submission candidate.

Default supervised corpus target: 100,000 unique nonterminal positions, configurable. Obtain positions from configured PGN sources or legal games generated by the project when no external corpus is supplied. Provide an executable Colab path for acquiring or locating the training-only teacher; record the resolved version. Do not assume a dataset or executable already exists, invent download URLs, or claim a tiny demonstration corpus is the full dataset.

Default engine analysis: 50,000 nodes per position, MultiPV 1, one thread per engine worker, 128 MB hash. Record whether the node limit completed and preserve the final reported depth. If the external teacher is unavailable, support an explicitly labelled classical-teacher data mode; do not mislabel it as Stockfish data.

Use JSONL records plus optional tensor shards. Include:

```text
schema_version
fen
legal_moves_uci
best_move_uci
played_move_uci                 # nullable for static annotations
policy_target_index
policy_target_distribution      # sparse legal index/probability entries; nullable
value_target
value_target_kind               # engine_cp, engine_mate, or game_result
game_result                     # nullable if unknown
termination_reason              # nullable for static annotations
source_label
source_uri_or_path
source_version
game_id
opening_id
ply
split                           # train, val, test
board_encoder_version
action_encoder_version
model_version                   # nullable for engine annotations
teacher_search_depth
teacher_node_count
raw_engine_cp                   # nullable
raw_engine_mate                 # nullable
```

For completed self-play, store both the teacher's preferred move/distribution and the move actually sampled. Do not pretend an exploratory move was the teacher's best move.

Split 80/10/10 at source-game group level with seed 42 before making training samples. Remove cross-split duplicates and transpositions using an explicit position key; document handling of common opening positions. Exclude reserved evaluation positions. Compute preprocessing statistics on training data only. Use validation for selection and tuning; reserve test data for the frozen final model.

Record actual post-filter counts, target coverage, source mixing, and dataset hashes. Reject illegal target moves and inconsistent target masks. An unavailable result is null, not an invented draw. Terminal positions may be used for value-only training; their policy loss must be skipped.

## 10. Supervised Training and CUDA

Use these concrete defaults:

```yaml
seed: 42
optimizer: adamw
learning_rate: 0.0003
adam_betas: [0.9, 0.999]
adam_eps: 0.00000001
weight_decay: 0.0001
max_epochs: 20
early_stopping_patience: 4
early_stopping_min_delta: 0.0001
batch_size_cuda: 512
batch_size_cpu: 64
gradient_accumulation_steps: 1
gradient_clip_norm: 1.0
scheduler: cosine
minimum_learning_rate: 0.000003
policy_loss: legal_cross_entropy
policy_label_smoothing: 0.0
value_loss: huber
huber_delta: 1.0
value_weight: 1.0
num_workers: 2
```

Total loss is mean legal policy cross-entropy plus `value_weight` times mean value loss. Handle records with only one valid target head using explicit masks and per-head normalization.

Training device selection:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Use BF16 autocast on supported CUDA devices such as A100; otherwise use FP16 with gradient scaling where appropriate. Use full precision on CPU. Pin DataLoader memory only for CUDA. Record GPU type, resolved precision, effective batch size, and all relevant versions. Do not enable mixed precision blindly on unsupported hardware.

Support explicit CPU-only training for portability, but all production training instructions target Colab. On out-of-memory, provide resumable microbatch/gradient-accumulation adjustment and log the change; do not silently change architecture, data, or loss settings.

Track total and component losses, legal top-1/top-3 accuracy, value MAE, source-specific value metrics, epoch time, learning rate, and data counts. Keep the lowest validation-loss checkpoint as `best_validation`; it is not automatically the strongest playing model.

## 11. Search-Guided Self-Play and League Learning

Implement approximate policy iteration using a frozen search teacher and a trainable policy/value network. The default sequence begins with notebook 2's selected supervised checkpoint.

Concrete defaults:

```yaml
iterations: 10
games_per_iteration: 512
teacher_max_depth: 3
teacher_target_temperature: 0.25
temperature_early: 1.0
temperature_late: 0.1
temperature_cutoff_episode_ply: 16
epsilon_exploration_early: 0.05
epsilon_exploration_late: 0.0
replay_capacity_positions: 500000
gradient_updates_per_iteration: 2000
self_play_learning_rate: 0.0001
supervised_replay_fraction: 0.25
base_ms: 120000
increment_ms: 500
clock_mode: competition
```

Implement the following complete iteration:

1. Freeze the current champion's weights and its BatchNorm statistics for collection. Use explicit immutable checkpoint paths and seeds. Do not update collector weights midway through a game.
2. Choose the opponent from this league mixture: 40% champion mirror match, 25% previous champion, 15% uniformly sampled older champions, 10% the preserved classical agent, and 10% starter baselines split equally between greedy and minimax. Renormalize unavailable historical buckets deterministically and log the resulting mixture. Alternate learner colour.
3. On learner turns, run alpha-beta teacher analysis. Obtain root scores at a common fully completed nominal depth by searching each legal root child with a full window. Iteratively increase that common depth up to 3 under the allotted move budget. Preserve the last complete root-score set; never softmax a mixture of incomplete-depth scores or fail-high/fail-low bounds.
4. For non-mate root scores, build a stable legal policy target `pi_teacher = softmax(root_scores / 0.25)` in the normalized search units defined above. If a winning mate is proven within the completed search, target the shortest proven mate, with deterministic tie-breaking. Preserve mate metadata. Use a separate behaviour distribution at the configured early/late temperature; temperature zero means deterministic argmax. Apply epsilon exploration only to the behaviour distribution, not to the stored teacher target.
5. If no full root-score iteration completes, play the legal fallback and mark the policy target missing. Do not invent teacher scores. Record actual elapsed time for the move, update clocks, and let the referee terminate the game. A move whose computation itself exceeded the clock remains a flag fall.
6. Record learner states with teacher targets, actions, game provenance, and clock data. Keep opponent records separate; do not label a baseline's move as if produced by the neural teacher. In champion mirror games both sides may provide learner data.
7. Once a game finishes normally, assign each recorded state's value target from the final result and its own side-to-move colour. By default exclude crashes, flags, initialization failures, and malformed-output episodes from gradient data, but retain them in operational results. Do not turn interrupted or unfinished Colab games into draws.
8. Append eligible records to the bounded replay buffer. Sample 25% of each update batch from the original supervised training split and 75% uniformly from the self-play replay buffer. Store each record's teacher version. Prevent any validation/test records from entering replay.
9. Train a candidate for 2,000 optimizer updates using the same legal policy and Huber value losses. Initialize from the champion; reset AdamW for each iteration with learning rate 1e-4 and constant learning rate for these updates. Save optimizer/update state so an interrupted iteration resumes rather than restarts. Save a candidate even if it is not promoted.
10. Evaluate the frozen candidate against the champion using the promotion protocol below. Promote only when the criterion is met; otherwise retain the champion and archive the candidate and results. The next iteration starts from the retained champion.

Changing the training actor to CUDA does not change the legal action space. It does change measured thinking speed. Identify GPU collection as research data generation, and use CPU agents for promotion and final match evaluation. An optional untimed collection mode must be named as an ablation and must not be reported as tournament-clock performance.

## 12. Evaluation, Promotion, and Ablations

Use the unmodified harness with CPU inference, one PyTorch thread, fixed checkpoint hashes, explicit opening suites, and reproducible baseline seeds. Preserve the official API and process lifecycle. Do not depend on harness-only seed variables in the submitted agent.

Development comparisons: 256 games per matchup, made from 128 distinct development FENs played once with each colour. Use 120,000 ms plus 500 ms. Compare the neural/search candidate against greedy, minimax, the original classical agent, and the previous champion. Record the exact classical snapshot hash.

Promotion: 256 games against the current champion using the paired development suite. Promote only if score is at least 55%, the lower bound of a two-sided 95% bootstrap interval exceeds 50%, and the candidate has zero crashes, illegal moves, initialization failures, or flags. Use 10,000 bootstrap resamples of opening pairs, preserving each colour pair as one resampling unit. Group by source family where the suite contains related openings. Use seed 42. Label this a development selection rule, not proof against arbitrary opponents.

After all tuning and training finish, evaluate the frozen selected model on a separate held-out suite: 512 games per chosen final matchup, using 256 FENs with colours swapped. Do not tune from these held-out results.

Report W/D/L, `(wins + 0.5 * draws) / games`, uncertainty, void games, failure reasons, mean/p50/p95/max move time, remaining clocks, completed search depths, node counts, inference-call counts, inference times observed during games, and fallback counts. Keep pairing and denominators explicit. Do not estimate a confident Elo advantage from a small sample or pretend deterministic repeats are independent openings.

Include controlled ablations: classical search only; policy ordering only; value evaluation only; policy plus value; supervised checkpoint versus self-play checkpoint. Hold clocks, opening pairs, seeds, and unrelated search settings fixed. These measure research performance, not submission compliance.

## 13. Hyperparameter Search

Provide an optional Colab training dependency on Optuna, using a seeded TPE sampler. Never run the Cartesian product of all settings. Default to 20 model/training trials and 10 self-play/search trials. Trials run sequentially within one Colab GPU runtime. Separate GPU sessions may run independent studies with unique study/output directories and immutable input data.

Model/training trial objective: best validation total loss with default fixed `value_weight = 1.0`. Search these spaces:

```yaml
residual_blocks: [4, 6, 8, 10]
channels: [64, 96, 128, 192]
value_head_hidden: [128, 256, 512]
activation: [relu, gelu, silu]
use_batch_norm: [true, false]
dropout: [0.0, 0.05, 0.1]
learning_rate: {low: 0.0001, high: 0.001, log: true}
weight_decay: {low: 0.00001, high: 0.0003, log: true}
batch_size_cuda: [512, 1024, 2048]
max_epochs: [10, 20, 30]
policy_label_smoothing: [0.0, 0.02, 0.05]
scheduler: [cosine, onecycle]
```

Make alternate architectures precise: the activation choice replaces every hidden activation; disabling BN removes BN and enables biases in those convolutions; nonzero dropout means `Dropout2d(p)` after the first residual activation only. For OneCycle, use the sampled learning rate as `max_lr`, `pct_start = 0.1`, `div_factor = 10`, `final_div_factor = 100`, and count actual optimizer steps. For cosine, use an end rate of 1% of the initial rate. Record all resolved settings.

Shortlist three model trials by validation loss for development matches. Use development playing results to select the research champion. Do not add CPU export benchmarks or package gates to these trials.

Self-play/search trial objective: paired development score against a fixed reference champion. Search a separate space:

```yaml
games_per_iteration: [128, 256, 512, 1024]
temperature_early: [0.75, 1.0, 1.25]
temperature_late: [0.0, 0.05, 0.1, 0.25]
temperature_cutoff_episode_ply: [8, 12, 16, 20]
value_weight: [0.5, 1.0, 2.0]
value_eval_mix: [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_beta_max_depth: [3, 4, 5, 6]
quiescence_depth: [2, 4, 6]
time_budget_fraction: [0.02, 0.04, 0.05, 0.06]
transposition_table_size: [50000, 100000, 200000]
policy_top_k_root: [all]
policy_top_k_internal: [all]
```

Use a fixed three-iteration training allowance per self-play tuning trial, then evaluate the final trial candidate on 128 paired development games before the full promotion evaluation. Settings that require training changes must actually retrain from the fixed starting checkpoint. Search-only settings reuse frozen weights. Do not compare differently weighted losses as if they were the same HPO objective.

Persist all trials, failures, seeds, configs, checkpoint paths, and results. Optional export-format/quantization comparison occurs only in notebook 5 after weights are frozen.

## 14. Checkpoints and Reproducibility

Use seed 42 by default, with replication seeds 43 and 44 configurable. Seed Python, NumPy, PyTorch, CUDA, data workers, opening sampling, exploration, and HPO. Offer a documented deterministic mode and report unsupported nondeterministic operations rather than promising bit-for-bit reproducibility across hardware.

Save a complete training checkpoint after every supervised epoch and every 250 self-play training updates, plus on orderly interruption when possible. Save `latest`, `best_validation`, and `champion` as distinct concepts. Use immutable versioned files and manifests rather than allowing parallel runs to overwrite a shared `best.pt`.

Training checkpoints include model, optimizer, scheduler, optional scaler, epoch/update/iteration counters, resolved config, metrics, Python/NumPy/Torch/CUDA RNG states, data-sampler state, replay manifest, league state, encoder versions, dataset hashes, model version, and software/hardware metadata. Preserve resume semantics through early stopping and scheduler progress. Explain any unavoidable limitation on resuming halfway through a multiprocess DataLoader epoch.

Write a temporary file, complete the save, publish the final filename, and verify its checksum. Do not assume a Colab disconnect always executes cleanup handlers or that Drive provides database transactions. Recover from the last complete checkpoint; ignore incomplete temporary files.

Use versioned dataset shards and checkpoint manifests to avoid excessive small-file Drive traffic. Keep durable outputs under `PROJECT_ROOT`. Log to CSV/JSONL, with readable plots saved under `results/`. Record code hashes or Git commit when available. Never fabricate completed epochs, match results, or trained weights.

## 15. Notebook Responsibilities and Parallel Execution

Create exactly these main workflow notebooks:

1. `01_chess_environment_and_encoding.ipynb`: mount Drive, dependency setup, paths/config, source snapshot, chess environment, encodings, ordinary correctness tests, and dataset/evaluation split preparation.
2. `02_supervised_policy_value_training.ipynb`: dataset import/generation and teacher labelling, model, supervised training, optional model HPO, validation plots, checkpoints, and resume.
3. `03_self_play_league_training.ipynb`: load the selected supervised checkpoint, collect games, create replay targets, train candidates, run development promotion matches, and persist league state.
4. `04_evaluation_and_comparison.ipynb`: frozen candidate comparisons, ablations, final selection record, and a single held-out evaluation after selection. No ZIP or submission compliance gates.
5. `05_final_export_and_compliance.ipynb`: CPU export, candidate packaging, artifact checks, final CPU benchmarks and live harness verification after the training/evaluation workflow is complete.

The first executable cell in every notebook mounts Drive. Subsequent setup resolves `PROJECT_ROOT`, verifies the shared code exists, and imports the package. Do not require local Windows paths at Colab runtime. Record dependency versions and explain any required runtime restart. A fresh Colab session must be able to resume a notebook from persisted inputs without another notebook's in-memory variables.

The default dependency chain is `01 -> 02 -> 03 -> 04 -> 05`. Notebook 03 needs a completed selected checkpoint from notebook 02. An explicit from-scratch self-play ablation is a different run configuration, not the default parallel workflow.

Independent notebook 02 experiments may run concurrently on separate Colab sessions. Independent notebook 03 branches may run concurrently after receiving immutable initialization checkpoints. Notebook 04 may evaluate already frozen candidates while separate experiments continue, provided results and inputs are isolated. Notebook 05 starts only after all experiments selected for this final run have finished and the final model is frozen.

Create a separate `NOTEBOOK_RUN_ORDER_README.md` for a teammate without a computer-science background. Explain opening Colab, enabling GPU, mounting Drive, locating files, changing the one run configuration, running cells, recognizing completion/failure, locating saved models, resuming after a disconnect, and which notebooks depend on which outputs. Include a small dependency diagram and a clear parallel/not-parallel table. Explain that opening multiple notebooks does not automatically allocate multiple GPUs or accelerate one shared job.

## 16. Meaningful Correctness Tests

Put focused tests in `tests/`. These are ordinary implementation tests, not iterative submission compliance. Run relevant tests during development and notebook 01 before expensive training. Do not remove them because packaging checks are deferred.

Cover:

- Known legal positions and rule transitions; promotion to all four pieces for both colours, castling through check, en passant exposing the king, pins, checks, checkmate, stalemate, and insufficient material.
- Per-position bijection of legal UCI moves to action indices and back; no index collisions; correct policy flatten order; Black promotion direction; legal-mask cardinality.
- Board plane orientation, piece colours, castling rights, counter normalization, en-passant convention, and reserved repetition plane equal to zero for both FEN and history-bearing boards.
- Threefold current-position detection versus a claimable next move; fifty-move boundary; supplied FEN counters; opening-inclusive ply cap; exact terminal precedence against the unmodified referee.
- Correct outcome sign for White and Black, negamax sign flips, separation of mate and leaf scales, and terminal positions with no policy target.
- Finite legal-only cross-entropy, including nonzero label smoothing, and valid gradients. Catch illegal teacher labels and missing targets.
- Board restoration on normal search return and timeout exceptions; TT score flags and rule-sensitive reuse; legal fallback when time is almost exhausted or the model is unavailable.
- No cross-split game/position leakage, immutable evaluation fixtures, resume state restoration, frozen teacher weights, and no accidental optimizer updates during inference.

Keep final export round-trip, isolated submission imports, ZIP validation, platform dependency checks, and the standalone CPU inference benchmark exclusively in notebook 05.

## 17. Final Export and Separate Submission Candidate

Notebook 05 must require a completed run manifest with final checkpoint hash and evaluation record. It does not retrain models or silently restart earlier notebooks when a final check fails.

Default export: a CPU `state_dict` of tensors in `best_model_cpu.pt`, reconstructable from a JSON architecture config and included Python model source. Exclude optimizer, replay, datasets, training logs, and teacher engines. Optional ONNX export is disabled by default; choose an opset supported by the actually documented runtime and record it.

The runtime must include:

```python
device = "cpu"
torch.set_num_threads(1)
```

Load the model once at import under the initialization budget, use `torch.load(path, map_location="cpu", weights_only=True)` for the tensor-only export, then `model.eval()` and `torch.inference_mode()`. Configure CPU thread limits before inference. A GPU-trained checkpoint is not automatically a packaged CPU agent: export, reconstruction, and integration must be implemented.

Place the final adapted wrapper only in `submission_candidate/agent.py`, preserving the existing starter repository's `agent.py`. Its API remains exactly:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    ...
```

Provide a complete implementation, not that placeholder. Share versioned model/encoding/search source through an explicit minimal runtime bundle. Runtime imports must not pull in training-only libraries, notebooks, plotting, data acquisition, or league utilities. All required runtime files use paths relative to the candidate's own directory.

Create the candidate archive using the unmodified packaging code, with explicit includes for JSON configuration and the runtime package. Use `weights/` for model files. Inspect the archive contents rather than assuming a `.pt` or `.json` file is automatically included. The ZIP root must contain the actual candidate `agent.py`, never the unchanged classical snapshot by mistake.

Final-only checks:

- Reload the exported weights and compare outputs on fixed positions against the selected model in inference mode, recording tolerances and precision.
- Import and run the extracted candidate in a fresh CPU process, with runtime dependencies and paths isolated from the training project.
- Exercise legal moves and low-clock/model-failure fallback through the public API; run real harness games from selected edge-case FENs as well as normal openings.
- Verify there is no required CUDA, network, Drive mount, external engine executable, training-only dependency, or writable path outside allowed scratch space.
- Verify complete runtime source, encoder versions, model config, file hashes, and model provenance records. Distinguish automated checks from provenance claims requiring human review.
- Check the ZIP root, every included file, and total uncompressed size against the current limit. Do not ship native engine binaries or precomputed middlegame answers.
- Benchmark batch-one CPU encoding plus inference after warm-up on fixed nonterminal positions: at least 20 warm-up calls and 200 measured calls; report median, p95, max, initialization time, and measured memory. Use 5 ms p95 as an initial research target, not an official rule or a promised result.
- Run the candidate through the unmodified harness at the official clock and record results. Perform resource-limit checks only where the execution environment can enforce them; report unavailable isolation checks as not verified. Colab timing does not certify the tournament's hardware.
- Write a final report with pass/fail/not-verified status, exact selected checkpoint and ZIP paths, commands, results, remaining failures, and evidence. Never claim platform acceptance before upload validation.

Quantization default is `none`. An optional final-only dynamic INT8 experiment may target supported linear layers; do not claim it quantizes or accelerates the convolutional trunk. Quantized/ONNX alternatives require their own output comparison and CPU match evaluation before being selected. For ONNX, separately configure its session to one inference thread; `torch.set_num_threads(1)` does not control ONNX Runtime.

If final checks fail, retain the artifacts and report the failure precisely. Do not automatically cycle compliance checks back through training notebooks. Do not upload anything or replace the original starter agent as part of this project-generation request.

## 18. Required Project Tree

Create the following organization. Generated artifacts appear only when their producing notebook actually succeeds; do not create fake weights or result CSVs with fabricated runs.

```text
Chess/
|-- README.md
|-- NOTEBOOK_RUN_ORDER_README.md
|-- requirements_colab.txt
|-- notebooks/
|   |-- 01_chess_environment_and_encoding.ipynb
|   |-- 02_supervised_policy_value_training.ipynb
|   |-- 03_self_play_league_training.ipynb
|   |-- 04_evaluation_and_comparison.ipynb
|   `-- 05_final_export_and_compliance.ipynb
|-- chess_rl/
|   |-- __init__.py
|   |-- config.py
|   |-- environment.py
|   |-- board_encoding.py
|   |-- action_encoding.py
|   |-- model.py
|   |-- classical_evaluation.py
|   |-- search.py
|   |-- time_management.py
|   |-- transposition.py
|   |-- inference.py
|   |-- dataset.py
|   |-- teacher_labelling.py
|   |-- training.py
|   |-- self_play.py
|   |-- league.py
|   |-- evaluation.py
|   |-- tuning.py
|   |-- checkpoints.py
|   |-- export.py
|   `-- reproducibility.py
|-- configs/
|   |-- default.yaml
|   |-- supervised.yaml
|   |-- self_play.yaml
|   |-- evaluation.yaml
|   `-- search_spaces.yaml
|-- docs/
|   |-- RESEARCH_DESIGN.md
|   |-- ENVIRONMENT_CONTRACT.md
|   `-- ENCODING_SPEC.md
|-- tests/
|-- reference/
|   |-- starter/                  # Unmodified reference source snapshot
|   `-- classical_agent/          # Preserved current team baseline
|-- datasets/
|   |-- raw/
|   |-- processed/
|   |-- openings/
|   `-- manifests/
|-- checkpoints/
|   |-- supervised/<run_id>/
|   `-- self_play/<run_id>/
|-- logs/<run_id>/
|-- results/<run_id>/
|   |-- games/
|   |-- plots/
|   `-- tables/
|-- exports/<model_version>/      # Generated only in notebook 05
|   |-- best_model_cpu.pt
|   |-- config.json
|   `-- model_manifest.json
`-- submission_candidate/         # Populated only in notebook 05
    |-- agent.py
    |-- chess_runtime/            # Minimal shared runtime source bundle
    |-- weights/
    |   `-- best_model_cpu.pt
    |-- config.json
    `-- model_manifest.json
```

Keep the runtime bundle behavior identical to its selected research implementation through explicit source selection and recorded hashes. Do not maintain a second independently edited encoder. Use a JSON runtime config so YAML is not a submission dependency. Exclude the reference repository and training package from the ZIP unless a minimal required source subset has been deliberately selected.

## 19. Completion Requirements

Deliver the runnable files and explain:

1. Where the project was created and how it maps to the exact Colab root.
2. Which reference files and official sources were actually inspected.
3. The environment contract and any source discrepancy or unreplicated platform behavior.
4. The exact encodings, network, rewards, search scores, and self-play learning targets.
5. How to run the notebooks, resume training, locate models, and coordinate independent runs.
6. Which ordinary correctness tests were run and their results.
7. Which training, evaluations, and final checks were not run, without inventing results.
8. How notebook 05 exports the selected model and builds a separate integrated submission candidate while leaving the starter agent unchanged.

Make the implementation readable enough for a physics and computer-science team to inspect and explain. Use concise comments for non-obvious logic, clear mathematical definitions, and useful plots. Finish the authorized code and documentation work; missing GPU access in the authoring environment should not prevent producing runnable Colab materials. Never represent project generation as completed model training.
