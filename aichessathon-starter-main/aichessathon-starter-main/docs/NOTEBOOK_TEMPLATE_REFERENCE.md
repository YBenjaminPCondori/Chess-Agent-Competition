# Notebook Template Reference

Use these notebooks as style and structure references for future chess/RL notebooks:

```text
D:\Benjamin Data\GitHub\Education\Deep_Reinforcement_Learning\notebooks
```

WSL path:

```text
/mnt/d/Benjamin Data/GitHub/Education/Deep_Reinforcement_Learning/notebooks
```

Reference notebooks found:

- `Part_C_PPO_Environment.ipynb`
- `Part_C_PPO_Training.ipynb`
- `Part_C_PPO_Training_OldEnv.ipynb`
- `DRL-CW-1b-AC.ipynb`
- `DRL-CW-1c-AC-PPO.ipynb`
- `DRL-CW-1.ipynb`
- `DRL-CW-1a.ipynb`
- `DRL-CW-1-Heatmap.ipynb`
- `DRL-CW-1-CapacityIncrease.ipynb`

## What To Reuse

- Notebook sectioning style.
- Markdown/code balance.
- Setup/import/reproducibility sections.
- Environment-definition sections.
- Training-loop sections.
- Evaluation sections.
- Result-table and plotting style.
- Saved-output/checkpoint conventions.
- Comparison between random, heuristic, and trained agents.

## What Not To Reuse Directly

The reference notebooks are from a different environment. Do not copy their environment dynamics, state variables, action semantics, reward functions, value targets, or policy targets into the chess competition work.

For AI Chessathon notebooks, the real environment remains:

```text
State: FEN / python-chess Board
Actions: legal chess moves in UCI
Runtime API: get_move(fen: str, time_left_ms: int) -> str
Episode: standard chess from curated opening FENs
Rewards: chess result and/or chess-specific shaped objectives
```

Use the old notebooks as presentation and workflow templates, not as the scientific definition of the chess task.

