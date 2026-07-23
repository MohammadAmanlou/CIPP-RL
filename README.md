# CIPP-RL

Reproducible reinforcement-learning research code for the
deterministic Campaign Itinerary Planning Problem and its
robust extension.

## Week 1 status

The deterministic foundation includes:

- validated `CIPPInstance` data model;
- exact campaign-wide deterministic objective;
- full-itinerary feasibility evaluation;
- reproducible synthetic instance generation;
- JSON save/load utilities;
- location-permutation invariance tests;
- deterministic automated tests;
- Week 1 validation script.

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

## Run tests

```bash
python -m pytest -q
```

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

The next milestone is Week 2:

- `CIPPEnv`;
- exact incremental reward;
- viability-based action masking;
- 1,000 feasible masked episodes.
