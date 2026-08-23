"""FINAL information-leakage regression checks for Method C V5."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np

from src.advanced.features import StructuredFeatureBuilder
from src.adaptive import AdaptiveCIPPEnv, AdaptiveFeatureBuilder, ShockScenario
from src.envs import CIPPEnv
from src.utils import load_professor_benchmark, parse_professor_instance_id


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="D_14S_30P")
    p.add_argument("--data-directory", type=Path, default=Path("."))
    p.add_argument("--switch-day", type=int, default=15)
    return p.parse_args()


def _states_equal(a, b) -> bool:
    return (
        np.array_equal(a.locations, b.locations)
        and np.array_equal(a.global_features, b.global_features)
        and np.array_equal(a.action_mask, b.action_mask)
    )


def main() -> None:
    a = _args()
    spec = parse_professor_instance_id(a.instance)
    benchmark = load_professor_benchmark(
        a.data_directory / f"CIPP-{spec.party}.xls",
        instance_id=spec.instance_id,
        objective_variant="professor_code",
        budget_mode="auto",
    )
    instance = benchmark.instance

    group = max(1, instance.n // 3)
    m1 = np.ones(instance.n, dtype=np.float64)
    m2 = np.ones(instance.n, dtype=np.float64)
    m1[:group] = 2.25
    m2[:group] = 0.55
    m2[-group:] = 2.25

    s1 = ShockScenario(a.switch_day, m1, "future_shock_A")
    s2 = ShockScenario(a.switch_day, m2, "future_shock_B")

    static_builder = StructuredFeatureBuilder(instance)
    adaptive_builder = AdaptiveFeatureBuilder(instance)

    # --------------------------------------------------------------
    # Check 1: BEFORE shock, two different future shocks must produce
    # identical Method C observations for the same realized history.
    # --------------------------------------------------------------
    pre1 = AdaptiveCIPPEnv(instance, s1, seed=1)
    pre2 = AdaptiveCIPPEnv(instance, s2, seed=2)
    pre1.reset(seed=1)
    pre2.reset(seed=2)

    # Use a common feasible history and stop one decision before the shock.
    for _ in range(max(a.switch_day - 1, 0)):
        mask = pre1.get_action_mask()
        feasible = np.flatnonzero(mask)
        action = int(feasible[0])
        pre1.step(action)
        pre2.step(action)

    method_c_before_1 = adaptive_builder.build(pre1)
    method_c_before_2 = adaptive_builder.build(pre2)
    pre_shock_hidden = _states_equal(
        method_c_before_1,
        method_c_before_2,
    )

    # --------------------------------------------------------------
    # Cross the shock using identical realized actions.
    # --------------------------------------------------------------
    while pre1.day <= a.switch_day:
        mask = pre1.get_action_mask()
        feasible = np.flatnonzero(mask)
        action = int(feasible[0])
        pre1.step(action)
        pre2.step(action)
        if pre1.done:
            break

    method_c_after_1 = adaptive_builder.build(pre1)
    method_c_after_2 = adaptive_builder.build(pre2)
    post_shock_visible = not _states_equal(
        method_c_after_1,
        method_c_after_2,
    )

    # --------------------------------------------------------------
    # Check 2: frozen B/D observations come from separate static envs,
    # hence are identical regardless of which adaptive shock scores them.
    # --------------------------------------------------------------
    st1 = CIPPEnv(instance, seed=3)
    st2 = CIPPEnv(instance, seed=4)
    st1.reset(seed=3)
    st2.reset(seed=4)

    for _ in range(min(a.switch_day + 1, instance.H)):
        mask = st1.get_action_mask()
        feasible = np.flatnonzero(mask)
        action = int(feasible[0])
        st1.step(action)
        st2.step(action)

    frozen_1 = static_builder.build(st1)
    frozen_2 = static_builder.build(st2)
    frozen_identical = _states_equal(frozen_1, frozen_2)

    print(
        "Method C pre-shock observations identical across different "
        "future shock identities:",
        pre_shock_hidden,
    )
    print(
        "Method C post-shock observations differ after shock is realized:",
        post_shock_visible,
    )
    print(
        "Frozen B/D strict-static observations identical:",
        frozen_identical,
    )

    if not pre_shock_hidden:
        raise RuntimeError(
            "FUTURE-SHOCK LEAKAGE: Method C sees future shock identity pre-shock"
        )
    if not post_shock_visible:
        raise RuntimeError(
            "ADAPTATION FAILURE: Method C does not observe realized shock"
        )
    if not frozen_identical:
        raise RuntimeError(
            "STATIC BASELINE LEAKAGE: B/D observation semantics are not static"
        )

    print("ALL INFORMATION-LEAKAGE CHECKS PASSED")


if __name__ == "__main__":
    main()
