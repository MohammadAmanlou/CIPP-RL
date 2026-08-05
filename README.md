# CIPP-RL

## Advanced PPO suite

The repository now includes a paper-ready, RL-only advanced suite:

- stable separate-actor/critic MLP PPO;
- location-token Attention PPO;
- hierarchical Idle/Visit/location PPO;
- HACIPP with exact-marginal residual logits, an auxiliary count planner,
  POMO/group-relative training, and elite self-imitation;
- simulation-guided beam search driven only by the frozen RL policy;
- a second PPO agent that performs up to 30 feasible neural-neighborhood
  improvement moves while retaining the best incumbent.

Run the end-to-end smoke test:

```bash
python -m experiments.run_ppo_suite \
  --instances D_14S_30P \
  --profile smoke \
  --methods stable_mlp attention hierarchical hacipp hacipp_rl_improve
```

See `KAGGLE_RUN_FA.md` for exact Kaggle commands and
`METHODS_AND_ABLATIONS_FA.md` for the scientific method map.

Reproducible feasibility-masked reinforcement-learning code for deterministic
Campaign Itinerary Planning Problem (CIPP) experiments.

## Implemented through Stage 4

- exact CIPP data model, objective evaluator and constraint checker;
- reproducible synthetic and professor-calibrated instance generation;
- exact viability masking and objective-consistent incremental rewards;
- Random Feasible and exact-increment Greedy baselines;
- masked Double DQN with replay buffer, target network and checkpointing;
- DQN-Greedy inference without rollout;
- Q-guided DQN backtracking with a configurable complete-rollout budget;
- canonical Gurobi MILP benchmark;
- CSV, Markdown, LaTeX and Excel table templates;
- automated tests and standalone 1,000-episode validation.

See `AUDIT_AND_FIXES.md` for the detailed review.

## Important dataset interpretation

The supplied professor script uses `Cities=16`, with index 0 equal to `Rest`.
That is 15 real visit locations.  In the published instance ID `14S`, the 14
means 14 visitable states. Idle is an additional internal action, so the action space has 15 actions. The legacy DQN
pipeline supports both source shapes explicitly:

```text
--instance-mode supplied-code   # 15 real locations; exact supplied-script shape
--instance-mode paper-14        # 14 real locations; revised-paper shape
```

Do not compare a `15S_CODE` result directly with a published `14S` table value.
The advanced PPO pipeline does not use either shape flag: `--instances
D_14S_30P` directly creates 14 visit actions plus one Idle action over 30 periods, while
JSON instance files directly supply their own `n` and `H` values.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Gurobi additionally requires a valid local license.

## Validate the environment

```bash
python -m scripts.run_week2_validation --episodes 1000
```

Expected: 1,000 completed, zero violations, zero dead ends and zero reward
mismatches.

## Smoke test

```bash
python -m experiments.train_dqn_real_matched \
  --episodes 100 \
  --seeds 0 \
  --validation-instances 10 \
  --normalizer-instances 10 \
  --batch-size 32 \
  --warmup-steps 100 \
  --output-dir checkpoints/smoke

python -m experiments.benchmark_smallest_real_instance \
  --party D \
  --checkpoints checkpoints/smoke/seed_0/best.pt \
  --rollouts 5 \
  --random-runs 5 \
  --skip-gurobi \
  --output-dir results/smoke
```

Smoke results are only for pipeline verification.

## Full DQN training

Supplied-code shape:

```bash
python -m experiments.train_dqn_real_matched \
  --instance-mode supplied-code \
  --episodes 20000 \
  --seeds 0 1 2 3 4 \
  --output-dir checkpoints/dqn_real_matched_code
```

Paper-14 shape:

```bash
python -m experiments.train_dqn_real_matched \
  --instance-mode paper-14 \
  --episodes 20000 \
  --seeds 0 1 2 3 4 \
  --output-dir checkpoints/dqn_real_matched_paper14
```

Every seed directory contains `best.pt`, `last.pt`, metrics, metadata and three
learning-curve figures.

## Benchmark DQN with and without rollout

```bash
python -m experiments.benchmark_smallest_real_instance \
  --party D \
  --instance-mode supplied-code \
  --checkpoints \
    checkpoints/dqn_real_matched_code/seed_0/best.pt \
    checkpoints/dqn_real_matched_code/seed_1/best.pt \
    checkpoints/dqn_real_matched_code/seed_2/best.pt \
    checkpoints/dqn_real_matched_code/seed_3/best.pt \
    checkpoints/dqn_real_matched_code/seed_4/best.pt \
  --rollouts 30 \
  --alternatives-per-state 3 \
  --random-runs 30 \
  --gurobi-time-limit 3600 \
  --output-dir results/smallest_real_D
```

Methods reported:

- Random Feasible distribution;
- Random Best-of-30;
- Greedy Exact Increment;
- DQN-Greedy (single frozen trajectory);
- DQN-Backtracking-30 (same frozen checkpoint, at most 30 complete paths);
- Gurobi.

If Gurobi is not certified optimal, the output says `Gurobi incumbent` and does
not call the reference an optimum.

## Run the whole pipeline for D and R

```bash
python -m scripts.run_stage4_pipeline \
  --episodes 20000 \
  --seeds 0 1 2 3 4 \
  --party both \
  --instance-mode supplied-code \
  --rollouts 30 \
  --random-runs 30 \
  --gurobi-time-limit 3600
```

## Main outputs

```text
checkpoints/.../seed_k/
  best.pt
  last.pt
  training_metrics.csv
  training_summary.json
  figures/*.png

results/.../
  benchmark_details.json
  comparison_table.csv
  comparison_table.md
  comparison_table.tex
  gurobi.log
  gurobi_model.lp
  gurobi_model.sol
```

The reusable Excel template is `results/comparison_table_template.xlsx`.

## v4 benchmark semantics correction

`D_14S_30P` contains 14 visitable states and one additional Idle action. The source row with zero reward and zero cost is validated as Idle and is not counted among the 14 visitable states. Therefore `instance.n = 14` and `instance.num_actions = 15`. Do not use the v3 package, which interpreted `14S` incorrectly.
