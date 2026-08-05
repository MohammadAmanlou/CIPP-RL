# CIPP-RL

Reproducible reinforcement-learning research code for the
deterministic Campaign Itinerary Planning Problem and its
robust extension.

## Week 1-4 status

The deterministic foundation includes:

- validated `CIPPInstance` data model;
- exact campaign-wide deterministic objective;
- full-itinerary feasibility evaluation;
- reproducible synthetic instance generation;
- JSON save/load utilities;
- location-permutation invariance tests;
- deterministic automated tests;
- Week 1 validation script.

Week 4 additionally includes instance-specific masked DQN and PPO,
on-policy rollouts with GAE, a Gurobi teacher, cross-entropy policy
pre-training, imitation-initialized PPO, and paper-style
CSV/Markdown/LaTeX/JSON outputs.

See `INSTANCE_SPECIFIC_RUN_FA.md` for the current Persian execution
guide. The older synthetic zero-shot workflow remains available for
ablation experiments, but it is not the primary report pipeline.

Action encoding:

- `0`: idle period;
- `1, ..., n`: visit a location.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For imitation learning, install `gurobipy` in the same environment
and activate the academic license. SciPy/HiGHS is included only as
a local verification fallback and is never labelled as Gurobi.

## Run tests

```bash
python -m pytest -q
```

## Week 4 quick commands

The report pipeline treats each real professor instance as a
deterministic optimization target. Train separate models for D and R.
For example, plain PPO on D:

```bash
python -m experiments.train_ppo \
  --professor-excel CIPP-D.xls \
  --party D \
  --output-directory results/week4_instance/D/ppo_seed42 \
  --objective-variant professor_code \
  --updates 100 \
  --episodes-per-update 8 \
  --normalizer-instances 8 \
  --update-epochs 4 \
  --device auto
```

Create a D-specific imitation initialization from Gurobi:

```bash
python -m experiments.train_imitation \
  --professor-excel CIPP-D.xls \
  --party D \
  --output-directory results/week4_instance/D/imitation_seed42 \
  --objective-variant professor_code \
  --solver gurobi \
  --teacher-time-limit 3600 \
  --epochs 300 \
  --hidden-dimension 128
```

Fine-tune it while retaining update zero as a valid best checkpoint:

```bash
python -m experiments.train_ppo \
  --professor-excel CIPP-D.xls \
  --party D \
  --output-directory results/week4_instance/D/imitation_ppo_seed42 \
  --objective-variant professor_code \
  --initial-checkpoint \
    results/week4_instance/D/imitation_seed42/imitation_initialization.pt \
  --updates 50 \
  --episodes-per-update 8 \
  --update-epochs 2 \
  --device auto
```

Train DQN:

```bash
python -m experiments.train_dqn \
  --professor-excel CIPP-D.xls \
  --party D \
  --output-directory results/week4_instance/D/dqn_seed42 \
  --objective-variant professor_code \
  --episodes 300 \
  --device auto
```

Then benchmark D with Greedy-1 and Best-of-30 for each learned model.
Repeat the same four commands with `CIPP-R.xls`, `--party R`, and an
R-specific output directory. Exact commands and the recommended
two-stage smoke/full procedure are in `INSTANCE_SPECIFIC_RUN_FA.md`.

## Run original Stage 1 demo

```bash
python -m scripts.run_stage1_demo
```

## Run complete Week 1 validation

```bash
python -m scripts.run_week1_validation
```

This command:

1. generates a deterministic synthetic instance;
2. saves it as JSON;
3. loads it again;
4. evaluates the all-idle itinerary;
5. saves the evaluation results.

## Minimal example

```python
from src.core import evaluate_itinerary
from src.utils import (
    generate_cipp_instance,
    load_instance,
    save_instance,
)

instance = generate_cipp_instance(
    n=14,
    H=30,
    seed=42,
)

save_instance(
    instance,
    "data/instances/example.json",
)

loaded = load_instance(
    "data/instances/example.json",
)

itinerary = [0] * loaded.H

result = evaluate_itinerary(
    loaded,
    itinerary,
)

print(result.objective)
print(result.total_cost)
print(result.feasible)
print(result.violations)
```

## Project structure

```text
CIPP-RL/
├── data/instances/
├── results/
├── scripts/
│   ├── run_stage1_demo.py
│   └── run_week1_validation.py
├── src/
│   ├── core/
│   │   ├── evaluation.py
│   │   └── instance.py
│   └── utils/
│       ├── instance_generator.py
│       ├── instance_io.py
│       └── permutation.py
├── tests/
│   ├── test_stage1_core.py
│   └── test_stage1_generation.py
├── .gitignore
├── requirements.txt
└── README.md
```

The next milestone after completing the full Week 4 experiments is
the Week 5 robust CIPP environment. Do not begin it until the
deterministic DQN/PPO/Imitation+PPO comparison is complete.
