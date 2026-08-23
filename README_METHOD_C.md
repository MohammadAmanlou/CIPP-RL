# Method C — Context-Conditioned Adaptive Attention

## Files to copy into the CIPP-RL repository

Copy exactly:

- `src/adaptive/__init__.py`
- `src/adaptive/context.py`
- `src/adaptive/training.py`
- `src/adaptive/gurobi.py`
- `experiments/run_adaptive_method_c.py`

No existing static file needs to be overwritten.

## Pilot

Use `D_14S_30P`.

The held-out pilot shock is at 50% of the horizon. The lowest-reward third of
locations is boosted and the highest-reward third is suppressed. Training uses
random disjoint shock subsets and several switch times, so it does not see the
exact held-out rank-flip scenario.

Three RL methods are compared from one identical realized prefix:

1. B — Frozen Attention greedy, no reward adaptation.
2. D — Frozen Attention + residual Best-of-K, candidates selected under new reward.
3. C — Context-conditioned Attention + residual Best-of-K.

Adaptive Gurobi fixes the exact same prefix and reoptimizes the remaining horizon
with the exact same time-varying reward objective and constraints.

## Important scientific point

If adaptive Gurobi proves optimality, a feasible RL solution cannot validly beat
that optimum under the identical objective and constraints. The paper target on
small instances should be: near-zero objective gap plus much lower adaptation
latency, and a clear gain over B/D. To show a higher feasible incumbent than
Gurobi, use a hard time-limited instance or a matched wall-clock Gurobi budget.

## Recommended strong pilot command

```bash
python experiments/run_adaptive_method_c.py \
  --instance D_14S_30P \
  --data-directory . \
  --device cuda \
  --updates 300 \
  --episodes-per-update 64 \
  --validation-interval 10 \
  --validation-scenarios 8 \
  --validation-rollouts 16 \
  --early-stopping-warmup 80 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 2.0 \
  --final-rollouts 512 \
  --gurobi-time-limit 3600
```

For a fast smoke test first:

```bash
python experiments/run_adaptive_method_c.py \
  --instance D_14S_30P \
  --data-directory . \
  --device cuda \
  --updates 3 \
  --episodes-per-update 8 \
  --validation-interval 1 \
  --validation-scenarios 2 \
  --validation-rollouts 4 \
  --early-stopping-warmup 0 \
  --early-stopping-patience 0 \
  --final-rollouts 16 \
  --skip-gurobi
```
