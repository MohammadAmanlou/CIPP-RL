# FINAL V7 — All Adaptive CIPP Ideas in One Experimental Framework

V7 is the union of all ideas developed in V3–V6 plus the new recourse-aware
prefix and hybrid RL/Gurobi ideas. Nothing scientifically distinct from earlier
versions is removed.

## Core information pattern

For every anticipatory RL method:

1. The policy chooses the pre-shock prefix without observing the realized
   future shock identity.
2. The prefix is committed and immutable.
3. The shock is revealed.
4. A second-stage solver/policy handles the suffix.

No best-of-K prefix hindsight is allowed.

---

## Methods retained from earlier versions

### B — Frozen static RL, greedy
Static RL prefix/policy, no shock-aware representation, greedy suffix.

### D — Frozen static RL + best-of-K
Static RL remains shock-unaware. Multiple suffixes are sampled after the shock
and scored under the realized adaptive objective.

### C-S — Generalist Method C on a fixed static-RL prefix
Pure post-shock adaptation ablation.

### C-C — Generalist Method C end-to-end
Rich shock-trained RL chooses its own non-clairvoyant prefix, then adapts after
the shock. V7 trains multiple C restarts and selects the best on a fixed
validation suite, never on the held-out test suite.

### S-R — Static RL prefix -> shock-specific residual RL
The V6 residual-retraining idea is preserved.

### Scratch residual RL
The main pilot still includes scratch initialization as an ablation.

---

## New V7 ideas

### P — Recourse-aware prefix policy

P is initialized from the best C checkpoint.

During training:

1. P chooses only the pre-shock prefix.
2. The future shock is hidden while the prefix is being chosen.
3. After the prefix is committed, the shock is revealed.
4. A frozen strong Method-C teacher solves the suffix with best-of-K search.
5. The final downstream objective is propagated back as the terminal learning
   signal for the prefix.

So P is explicitly trained for:

`current reward + downstream recourse value`

rather than merely for completing the entire trajectory itself.

Two deployment variants are evaluated:

- **P-P**: P prefix -> P suffix
- **P-C**: P prefix -> frozen strong C suffix

Local Gurobi later adds:

- **P-G**: P prefix -> exact Gurobi suffix

### I — Intensive instance-specific recourse-aware prefix

I is a heavier same-instance refinement initialized from P. It is trained on a
fresh, disjoint random shock stream for the target instance. The held-out test
shock identities are never used during training.

This implements the user's instance-specific optimization idea while still
keeping shock realizations held out.

Deployment:

- **I-I**
- **I-C**
- **I-R**
- **I-G**

### C-R / P-R / I-R

After each respective prefix is committed and the shock is observed, a new RL
optimization phase is trained only on the residual instance.

The main pilot tries several initializations and keeps the best achieved
solution while reporting the total online training cost.

### C-G / P-G / I-G

Exact Gurobi takes over after the RL prefix.

These are the cleanest tests of prefix quality because the suffix is solved
exactly. In particular:

`P-G > G-G`

means the recourse-aware RL prefix created a better post-shock state than the
myopic static-Gurobi prefix.

### SG-G — Stochastic Gurobi -> exact Gurobi recourse

V7 adds the strong fairness baseline that knows the shock *distribution* but
not the future realization.

For each shock day, a scenario-based stochastic MILP optimizes one common
nonanticipative prefix with scenario-specific recourse. The realized test shock
is still unknown while that prefix is chosen.

This addresses the reviewer concern that anticipatory RL should not be compared
only against a myopic optimizer that has no uncertainty model.

### Clairvoyant Gurobi

Perfect-information upper bound. It knows the realized shock from day 0 and is
not an operational baseline.

---

## Generalist across compatible instances

`run_adaptive_v7_all_methods.py` accepts:

`--training-instances ID1,ID2,...`

If the instances have the same number of locations/action space, a single
shared Attention network can be trained over all of them with
`train_multi_instance_method_c`.

If omitted, V7 is general over the rich shock distribution on the target
instance only.

---

## Rich shock curriculum retained from V6

- no shock
- mixed boost/suppress
- boost only
- suppress only
- concentrated extreme shocks
- distributed shocks
- multi-shock training episodes
- broad shock timing

The 100-scenario held-out single-shock suite is still frozen and saved before
evaluation.

---

## High-compute quality controls

V7 uses:

- multiple C training restarts;
- multiple P restarts;
- multiple I restarts;
- validation-only checkpoint/restart selection;
- high best-of-K search on the main pilot;
- separate greedy results where useful;
- multiple residual-RL initializations on the main pilot;
- exact Gurobi suffix oracles for every prefix;
- stochastic Gurobi as a stronger anticipatory optimization baseline;
- clairvoyant Gurobi as the perfect-information bound.

No test scenario is used to choose C/P/I training checkpoints.

---

## Main files

New:
- `experiments/run_adaptive_v7_all_methods.py`
- `experiments/run_v7_gurobi_all_methods.py`
- `README_FINAL_V7.md`

Updated:
- `src/adaptive/training.py`
- `src/adaptive/gurobi.py`
- `src/adaptive/__init__.py`

Retained:
- V6 runner
- V6 Gurobi suite runner
- V6 leakage regression
- V6 rich shock context implementation

---

## Kaggle

Use the supplied `Kaggle_CIPP_RL_FINAL_V7_ALL_METHODS.ipynb`.

The notebook:
1. clones the pushed V7 commit;
2. validates required files/imports;
3. runs leakage checks;
4. runs a tiny all-method smoke test;
5. runs the high-compute V7 RL experiment;
6. packages results/checkpoints/logs.

---

## Local Gurobi

After copying the Kaggle folder to:

`results/AdaptiveV7/D_14S_30P/seed_42`

run:

```powershell
python .\experiments\run_v7_gurobi_all_methods.py `
  --instance D_14S_30P `
  --data-directory . `
  --rl-output-directory results\AdaptiveV7 `
  --seed 42 `
  --time-limit 3600 `
  --mip-gap 0 `
  --stochastic-scenarios 8 `
  --stochastic-time-limit 900 `
  --stochastic-mip-gap 0.01 `
  --artifact-scenarios 1
```

Final outputs:

- `v7_all_methods_complete_results.csv`
- `v7_all_methods_complete_results.json`
- `v7_all_methods_complete_summary.json`
- `v7_main_pilot_complete.json`

---

## Key final comparisons

### Pure operational performance

- C-C vs G-G
- P-P / P-C vs G-G
- I-I / I-C vs G-G
- S-R / C-R / P-R / I-R vs G-G

### Clean prefix quality

- S-G vs G-G
- C-G vs G-G
- P-G vs G-G
- I-G vs G-G
- SG-G vs G-G

### Diagnostic decomposition

For a prefix X:

`prefix advantage = X-G - G-G`

`adaptation gap = X-G - X-RL`

### Upper bound

Every feasible nonclairvoyant method should remain at or below clairvoyant
Gurobi under the same formulation.

---

## Important interpretation

Training and testing an *online residual optimizer* on the same realized
residual instance is legitimate because R is an optimization procedure, not a
generalization claim. Its online training time must be reported.

For C/P/I, checkpoint selection is performed on separate random validation
shock streams. The fixed held-out suite remains untouched until evaluation.
