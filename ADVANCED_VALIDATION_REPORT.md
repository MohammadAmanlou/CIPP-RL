# Advanced PPO validation report

Validated locally on 2026-08-05.

## Static validation

```text
python -m compileall -q src experiments tests
```

Result: passed.

## Automated tests

```text
python -m pytest -q
```

Result:

```text
71 passed, 1 skipped
```

The skipped test is the optional licensed Gurobi integration test. The test
suite covers the canonical objective, constraints, viability masking, legacy
DQN/PPO behavior, instance-owned dimensions and explicit Idle/action semantics, all four advanced
construction networks, masked PPO updates, count planning, POMO batches,
policy-only SGBS, vectorized neighborhood evaluation, and the non-degradation
guarantee of RL improvement.

## End-to-end smoke pipeline

The following was run successfully on CPU:

```text
python -m experiments.run_ppo_suite \
  --instances D_14S_30P \
  --profile smoke \
  --methods stable_mlp attention hierarchical hacipp hacipp_rl_improve \
  --device cpu
```

Result:

```text
completed=14 rows
```

This verified training, checkpoint round-trips, deterministic decoding,
Best-of-K, SGBS, RL improvement, resumable output layout, and leaderboard
generation with `n=14`, `H=30`, and 15 actions including Idle. A second smoke
run using `--instance-files data/instances/week1_seed42.json` also passed,
confirming that JSON `n` and `H` values control the network and environment.
Smoke objectives are intentionally not research results because each
construction model receives only two PPO updates.
