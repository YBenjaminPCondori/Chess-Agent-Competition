# Current Agent Technical Documentation

This document describes the current `agent.py` implementation in engineering terms. It is written for a physics and computer science team that wants to understand the present baseline, reason about its behavior, and choose future experiments.

## Scope

The current submission agent is a classical chess-search program. It does not use a trained model, an opening book, Syzygy tablebases, external engines, GPU inference, network access, or non-standard dependencies.

The implementation is intentionally contained in the starter repository's root `agent.py`, because the competition platform imports that file directly.

## Competition Interface

The platform calls exactly one public function:

```python
def get_move(fen: str, time_left_ms: int) -> str
```

Inputs:

- `fen`: the full board state, including side to move, castling rights, en-passant state, clocks, and move number.
- `time_left_ms`: wall-clock time remaining for the agent before making this move.

Output:

- A legal move in UCI format, for example `e2e4`, `g7g8q`, or `e1g1`.
- If the position is terminal and no legal move exists, the function returns the UCI null move `0000`.

The implementation validates the selected move before returning it. If search crashes for any reason, `get_move` falls back to the first legal move rather than failing the game.

## Runtime Model

The agent creates one module-level `Engine` instance:

```python
ENGINE = Engine()
```

This means in-process state, especially the transposition table, can survive between moves during one game. The platform starts a fresh process for another game, so this state should not be treated as persistent training or long-term memory.

Only `python-chess` and Python standard library modules are imported.

## High-Level Move Selection

The move-selection path is:

```text
get_move(fen, time_left_ms)
  -> parse FEN into chess.Board
  -> Engine.choose_move(board, time_left_ms)
     -> list legal moves
     -> choose a legal fallback before search
     -> allocate conservative search time
     -> iterative deepening
        -> root search
           -> negamax alpha-beta
              -> quiescence at horizon
              -> static evaluation
     -> return best fully searched move, or fallback on timeout/error
  -> verify returned move is legal
  -> return UCI string
```

The most important safety property is that a legal fallback is selected before expensive search starts. If the deadline is hit or an exception occurs, the agent can still return a legal move.

## Search Algorithm

The core game-tree search is negamax with alpha-beta pruning.

Negamax is used because chess is a zero-sum game: the score for the current player is the negative of the score for the opponent after a move. Alpha-beta pruning reduces the number of nodes searched by skipping branches that cannot affect the final decision under perfect-play assumptions.

The search uses iterative deepening:

- Search depth 1 first.
- Then depth 2.
- Then depth 3, and so on until the soft deadline is reached.

This gives the agent an anytime property: even if a deeper search times out, the best move from the last completed depth remains available.

The implementation uses safe board mutation:

```python
board.push(move)
try:
    ...
finally:
    board.pop()
```

This guarantees that temporary search moves are undone even if a timeout exception is raised inside the subtree.

## Time Management

The agent uses a soft per-move deadline based on `time_left_ms`.

Important constants:

- `CHECK_INTERVAL = 64`: the search checks the deadline every 64 visited nodes.
- `MAX_Q_DEPTH = 4`: quiescence search is capped to avoid unbounded capture sequences.

Low-clock behavior:

- If `time_left_ms <= 100`, the agent returns the fallback move immediately.
- For other clocks, `allocate_time_ms()` computes a conservative search budget.
- Larger clocks allocate more time, but the cap is still conservative to avoid flagging.

This is not a hard real-time guarantee. Python execution, `python-chess` overhead, and the platform watchdog are still relevant. The design goal is to return safely under normal short-clock and live-game conditions.

## Static Evaluation

`evaluate(board)` returns an integer score from the side-to-move perspective. Positive means favorable for the player to move; negative means favorable for the opponent.

The score is roughly centipawn-like, but it is not calibrated to a professional engine scale.

Evaluation terms:

- Material:
  - Pawn: 100
  - Knight: 320
  - Bishop: 330
  - Rook: 500
  - Queen: 900
  - King: 0
- Piece-square tables:
  - Reward or penalize pieces based on square location.
  - Tables are defined from White's perspective.
  - Black pieces use mirrored square indices.
- Activity:
  - Counts attacked squares for knights, bishops, rooks, and queens.
  - More active pieces receive a small bonus.
- Pawn structure:
  - Penalizes doubled pawns.
  - Penalizes isolated pawns.
  - Rewards passed pawns increasingly as they advance.
- Bishop pair:
  - Adds a bonus for owning two bishops.
- Terminal states:
  - Checkmate returns a large negative score for the side to move.
  - Stalemate, insufficient material, and fifty-move-rule positions return zero.

The current king table is midgame-oriented. There is no separate endgame king activity model yet.

## Quiescence Search

At the normal depth limit, the agent does not immediately evaluate every position. Instead, it runs quiescence search.

Purpose:

- Reduce horizon effects from unstable tactical positions.
- Continue looking through forcing captures and promotions before applying static evaluation.

Behavior:

- If the side to move is in check, quiescence searches all legal evasions.
- Otherwise, it evaluates the quiet position as `stand_pat`.
- It then searches only noisy moves:
  - captures
  - promotions
- Search stops at `MAX_Q_DEPTH`.

This is a tactical stabilizer, not a full tactical solver. It does not search every check extension or all quiet threats.

## Move Ordering

Move ordering is important because alpha-beta pruning is strongest when good moves are searched first.

The current ordering prioritizes:

- The transposition-table best move, when available.
- The previous iteration's best move at the root.
- Promotions.
- Captures, scored with a simple MVV-LVA-style heuristic.
- Checking moves.
- Moves toward central squares.

This does not change the mathematical result at a fixed full-width depth, but it strongly changes how much useful depth the agent reaches before the deadline.

## Transposition Table

The transposition table stores previous search results for board positions.

Each entry contains:

- Search depth.
- Score.
- Bound flag:
  - exact value
  - lower bound
  - upper bound
- Best move in UCI form.

The table is bounded:

```python
MAX_TT_ENTRIES = 100_000
```

When the table reaches that size, it is cleared. This is simple and predictable, but not optimal. A future version could use aging, replacement priority, or per-game cleanup.

The position key uses `python-chess` transposition support when available, otherwise it falls back to board layout, side to move, castling rights, and en-passant square.

## Fallback Move Logic

The fallback move is selected before search begins.

It scores legal moves using cheap static rules:

- Prefer promotions.
- Prefer captures.
- Prefer high-value victim and low-value attacker captures.
- Prefer moves toward central squares.

The fallback is not meant to be strategically strong. Its purpose is to guarantee a legal return under low clock, timeout, or unexpected exceptions.

## Current Repository Workflow

Relevant files:

- `agent.py`: current competition candidate.
- `harness/play.py`: one live game against a baseline or another agent directory.
- `harness/arena.py`: multiple games with aggregate result statistics.
- `harness/package.py`: builds `submission.zip`.
- `baselines/`: reference opponents.
- `docs/IDEAS.md`: search and improvement ideas from the starter repo.

Current Makefile targets:

```text
make play      # one game
make arena     # multiple games
make live      # alias for arena
make zip       # build submission.zip
make gate      # static checks plus arena run, if local tools exist
```

The package step now only builds the zip. Game testing is done through `play`, `arena`, or `live`.

## Current Strength Profile

This agent is stronger than the original starter random bot and is intended to be the main classical-search baseline for experimentation.

Expected strengths:

- Always produces legal moves for valid non-terminal FENs under normal operation.
- Searches beyond one-ply greedy tactics.
- Handles captures, promotions, checks, low-clock calls, and common endgame states.
- Has a clear architecture for tuning evaluation and search behavior.

Expected weaknesses:

- Evaluation is hand-written and not tuned against data.
- No opening book or learned policy prior.
- No tablebases.
- No neural evaluation.
- No advanced pruning such as null-move pruning, late-move reductions, aspiration windows, or principal variation search.
- No specialized endgame evaluation.
- No explicit threefold-repetition claim logic in evaluation.
- Time control is conservative and may leave search strength unused on large clocks.

## Experiment Surfaces

The safest future improvements are isolated around specific functions:

- Evaluation weights:
  - `PIECE_VALUES`
  - piece-square tables
  - activity scaling
  - pawn-structure coefficients
  - bishop-pair bonus
- Search control:
  - maximum depth strategy
  - time allocation
  - quiescence depth
  - check extensions
  - move ordering
- Transposition table:
  - replacement policy
  - stored principal variation
  - memory-aware sizing
- Engine features:
  - aspiration windows
  - null-move pruning
  - late-move reductions
  - killer move heuristic
  - history heuristic
- Data-driven work:
  - tune weights by self-play or labelled positions
  - train a small evaluation model separately
  - add a model only after the classical baseline is measured and stable

For controlled research, modify one category at a time and measure against the same baseline set, time control, opening seed set, and game count.

## Practical Interpretation

The current code is competition-compatible baseline code, not an educational toy. It is still not a finished high-strength engine.

The right way to treat it is:

- Use it as the reference implementation.
- Run live games against starter baselines before and after each change.
- Keep `get_move(fen, time_left_ms) -> str` unchanged.
- Keep all dependencies within the allowed competition environment.
- Avoid modifying the harness unless the harness itself has a clear local bug.

