# Environment Contract

Reference: [AI Chessathon documentation](https://aichessathon.com/docs), retrieved 2026-09-10; local source hashes are in source_manifest.json.

The game is full standard chess. The agent sees FEN and its own remaining milliseconds, and returns one legal UCI move. The referee maintains history and clocks. The actor does not receive hidden history, the other player's clock, engine labels, or later results.

## End Conditions

The reference first calls board.outcome(), then checks current-position threefold repetition, current fifty-move status, and board.ply() >= 600. Checkmate therefore takes priority over a move-clock or ply-cap draw. The cap includes the initial FEN's fullmove number. A draw claim made possible by a future move is not an immediate draw in this adapter.

Each game starts with a fresh move stack from its given FEN. The supplied halfmove counter is preserved. The public documentation says counts begin at the first FEN; the local implementation and IDEAS.md preserve its halfmove counter. This wording difference is recorded rather than resolved by editing the harness.

The actor can search hypothetical repetitions starting at its root. This version does not reconstruct the prior actual game history. The neural repetition plane is always zero in both training and runtime, while the referee's actual repetition detection remains enabled. FEN is an observation of the complete game state, not a complete history-aware Markov state for every draw condition.

## Clocks and Failures

Each side begins with 120,000 ms and receives 500 ms after a legal move. The reference subtracts measured wall time and checks for a negative remaining clock before applying the returned move. An exact zero is not yet negative. A flag is a loss except for the reference's insufficient-material condition. Illegal UCI, illegal moves, crashes and init failures are separate failure reasons.

The research collection adapter preserves these rule decisions, but does not reproduce process-level memory limits, OS scheduling, or the platform's physical CPU. GPU collection changes thinking speed and is identified in its records. CPU evaluation uses the unchanged process harness.

## Initial Positions

The public eight openings are examples, not the hidden rated set. Self-play uses these public openings. Development and held-out suites are separately seeded legal playouts. They are not certified balanced or drawn from the unpublished platform distribution. Corpus and opening quality can change research results without changing the legal action environment.

## Submission Boundary

The final candidate is a separate folder built only in notebook 05. The original submitted agent is not changed. Platform package versions and numerical limits are recorded in source_manifest.json. CPU export, ZIP checks, initialization benchmarking and final live checks are deferred to notebook 05.
