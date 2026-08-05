# CIPP-RL PPO Suite v4 — corrected benchmark semantics

- Corrected the previous v3 interpretation: `14S` is 14 visitable states, not a count including Idle.
- `D_14S_30P` now loads 14 reward/cost rows after the zero row and creates 15 actions total.
- Updated loader validation, metadata, tests, and documentation accordingly.
- Retained vectorized rollouts, periodic logs, fixed validation, full early stopping, and benchmark-only mode.

# PPO Suite v4 — Correctness, Speed, Logging, and Early Stopping

## Correctness
- `D_14S_30P` means 14 visitable states; Idle is one additional internal action, for 15 actions total.
- Source row zero must be Idle/Rest with zero reward and zero cost.
- Idle is represented exactly once as environment action 0.
- Processed CSV fallback allows budget-disabled tests/runs without importing xlrd.

## Training
- Full construction early stopping on fixed validation trajectories.
- Full improvement early stopping on fixed construction starts.
- Best checkpoint uses the current frozen policy validation score, never archive best.
- Configurable validation metric: deterministic, stochastic mean, or Best-of-K.
- Training summaries include actual completed updates, stop reason, best update, and elapsed time.

## Speed
- Batched rollout collection: one policy forward per day for all synchronized episodes.
- Batched Best-of-K evaluation.
- Batched policy beam search and batched simulation completions.
- Partial history is no longer rewritten after every update.
- `torch.inference_mode()` is used for frozen-policy batched inference.

## Benchmarking
- Strict `--benchmark-only`: missing checkpoints cause an error instead of unexpected training.
- RL improvement runtime now includes start generation plus improvement.
- Runtime breakdown is included in leaderboard rows.

## Validation
- 75 tests passed; one optional Gurobi test skipped when Gurobi is unavailable.
- Full five-method D_14S_30P smoke run completed.
- Strict benchmark-only smoke run completed from saved checkpoints.
