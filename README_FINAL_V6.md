# FINAL V6 — Rich Shock Curriculum + Anticipatory RL + Shock-Time RL Retraining

This package supersedes V5 for new experiments.

## Research questions

V6 separates three questions that were entangled before:

1. **Adaptation:** given the same realized pre-shock history, can Method C react
   well after a reward shock?
2. **Anticipation:** can an RL policy trained over future uncertainty preserve
   better optionality before the shock than a myopic static optimizer?
3. **Shock-time online RL optimization:** if we follow a static RL plan before
   the shock, then freeze those actions and retrain RL only on the remaining
   realized suffix, how competitive is that with Gurobi replanning?

## Methods

### B — frozen static RL, greedy
No adaptive observation.

### D — frozen static RL, post-shock best-of-K
Policy remains shock-unaware; candidate suffixes are scored under the realized
objective.

### C — rich-curriculum anticipatory Method C
Offline warm-start from the static Attention checkpoint. V6 trains over:
- no-shock episodes;
- mixed boost/suppress shocks;
- boost-only shocks;
- suppress-only shocks;
- concentrated 1–2-location extreme shocks;
- distributed mild shocks;
- multi-shock episodes (training only);
- a wide range of shock times and magnitudes.

The future shock identity is never observable before it occurs.

### R-static-init — user's proposed shock-time RL retraining
1. Follow the static RL route until the shock.
2. Freeze every pre-shock action.
3. Initialize a new adaptive RL training phase from the static RL weights.
4. Train only the remaining suffix on the one realized scenario.

### R-scratch
Same as R-static-init, except the post-shock residual policy starts from random
weights. This distinguishes "retrain again" from warm-start fine-tuning.

Online training time is reported because Method R is an online optimization
procedure, not a pre-trained generalist policy.

## Held-out suite

V6 freezes a deterministic 100-scenario single-shock test suite:
- scenario 0 is the original day-15 rank-flip pilot;
- ~70% are unseen random single shocks from broad families;
- the rest are structured OOD cases such as rank flips, extreme single-city
  boosts, high-reward collapse, high-cost boosts, and budget-reallocation
  shocks.

The exact suite is saved to `heldout_suite.json`. Local Gurobi reads this file,
so RL and Gurobi see exactly the same realized scenarios.

Multi-shock scenarios are used in training only in V6. They are excluded from
the exact two-stage Gurobi suite so the comparison has a single clean shock
boundary.

## Exact decomposition with Gurobi

For every held-out scenario, the local Gurobi script computes:

- static day-0 Gurobi plan (solved once globally);
- exact post-shock continuation from the static-Gurobi prefix;
- exact post-shock continuation from Method C's own realized prefix;
- clairvoyant perfect-information Gurobi.

This gives:

`prefix advantage = oracle(Method C prefix) - two-stage Gurobi`

`adaptation gap = oracle(Method C prefix) - Method C`

`final advantage = Method C - two-stage Gurobi`

The final suite report contains:
- Method C win rate vs two-stage Gurobi;
- mean / median / std final gap;
- Method C-prefix win rate;
- mean prefix advantage;
- mean post-shock adaptation gap;
- clairvoyant gap;
- family-wise breakdown;
- Method R comparison on the representative retraining subset.

## Files to overwrite / add

Overwrite:
- `src/adaptive/context.py`
- `src/adaptive/training.py`
- `src/adaptive/__init__.py`
- `experiments/run_adaptive_method_c.py`
- `experiments/check_static_baseline_no_leakage.py`

Keep / overwrite with package copy:
- `src/adaptive/gurobi.py`

Add:
- `experiments/run_v6_gurobi_suite.py`
- `README_FINAL_V6.md`

## Recommended Kaggle full run

```bash
python -u experiments/run_adaptive_method_c.py \
  --instance D_14S_30P \
  --data-directory . \
  --output-directory /kaggle/working/AdaptiveV6 \
  --device cuda \
  --seed 42 \
  --updates 400 \
  --episodes-per-update 96 \
  --validation-interval 10 \
  --validation-scenarios 16 \
  --validation-rollouts 16 \
  --early-stopping-warmup 120 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 2.0 \
  --adaptive-lr-scale 0.10 \
  --adaptive-update-epochs 2 \
  --main-rollouts 512 \
  --suite-count 100 \
  --suite-rollouts 64 \
  --heldout-seed 260824 \
  --residual-updates 120 \
  --residual-episodes 64 \
  --residual-validation-interval 5 \
  --residual-validation-rollouts 32 \
  --residual-warmup 20 \
  --residual-patience 10 \
  --residual-min-delta 1.0 \
  --residual-lr-scale 0.20 \
  --residual-update-epochs 2 \
  --retrain-suite-count 12 \
  --suite-retrain-updates 50 \
  --suite-retrain-episodes 32 \
  --suite-retrain-rollouts 64
```

## Local exact Gurobi suite

After copying the Kaggle `AdaptiveV6/D_14S_30P/seed_42` folder to
`results/AdaptiveV6/D_14S_30P/seed_42`:

```powershell
python .\experiments\run_v6_gurobi_suite.py `
  --instance D_14S_30P `
  --data-directory . `
  --rl-output-directory results\AdaptiveV6 `
  --seed 42 `
  --time-limit 3600 `
  --mip-gap 0 `
  --artifact-scenarios 1
```

Important output:
- `gurobi_suite_results.csv`
- `gurobi_suite_summary.json`
- `main_complete_comparison.json`

## Interpretation rules

- Method C may legitimately beat two-stage Gurobi because the latter chooses
  its pre-shock plan under the static objective.
- Method C should not beat a proven-optimal clairvoyant Gurobi under identical
  objective and constraints.
- A positive Method-C prefix advantage with a negative final advantage means
  anticipation worked but post-shock adaptation/search lost the benefit.
- Method R can be compared in objective, but its online training time must be
  reported beside Gurobi reoptimization time.
