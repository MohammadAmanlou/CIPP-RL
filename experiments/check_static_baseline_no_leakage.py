"""FINAL V6 information-leakage regression checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced.features import StructuredFeatureBuilder
from src.adaptive import (
    AdaptiveCIPPEnv,
    AdaptiveFeatureBuilder,
    MultiShockScenario,
    ShockScenario,
    scenario_from_dict,
    scenario_to_dict,
)
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


def _common_action(env) -> int:
    feasible = np.flatnonzero(env.get_action_mask())
    if feasible.size == 0:
        raise RuntimeError("no feasible action in regression check")
    return int(feasible[0])


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

    s1 = ShockScenario(a.switch_day, m1, "future_A", "check")
    s2 = ShockScenario(a.switch_day, m2, "future_B", "check")

    second_day = min(instance.H - 2, a.switch_day + max(2, instance.H // 6))
    multi1 = MultiShockScenario(
        change_days=(a.switch_day, second_day),
        post_multipliers=(m1, m2),
        scenario_id="multi_A_then_B",
    )
    multi2 = MultiShockScenario(
        change_days=(a.switch_day, second_day),
        post_multipliers=(m2, m1),
        scenario_id="multi_B_then_A",
    )

    adaptive_builder = AdaptiveFeatureBuilder(instance)
    static_builder = StructuredFeatureBuilder(instance)

    # 1) Future single-shock identity hidden before realization.
    e1 = AdaptiveCIPPEnv(instance, s1, seed=1)
    e2 = AdaptiveCIPPEnv(instance, s2, seed=2)
    e1.reset(seed=1)
    e2.reset(seed=2)
    for _ in range(max(a.switch_day - 1, 0)):
        action = _common_action(e1)
        e1.step(action)
        e2.step(action)
    pre_single_equal = _states_equal(
        adaptive_builder.build(e1),
        adaptive_builder.build(e2),
    )

    # 2) Future multi-shock identity also hidden before first realization.
    m_e1 = AdaptiveCIPPEnv(instance, multi1, seed=3)
    m_e2 = AdaptiveCIPPEnv(instance, multi2, seed=4)
    m_e1.reset(seed=3)
    m_e2.reset(seed=4)
    for _ in range(max(a.switch_day - 1, 0)):
        action = _common_action(m_e1)
        m_e1.step(action)
        m_e2.step(action)
    pre_multi_equal = _states_equal(
        adaptive_builder.build(m_e1),
        adaptive_builder.build(m_e2),
    )

    # Cross first shock; observations should now differ.
    while e1.day <= a.switch_day and not e1.done:
        action = _common_action(e1)
        e1.step(action)
        e2.step(action)
    post_single_diff = not _states_equal(
        adaptive_builder.build(e1),
        adaptive_builder.build(e2),
    )

    # 3) Frozen B/D observations remain static regardless of shock.
    st1 = CIPPEnv(instance, seed=5)
    st2 = CIPPEnv(instance, seed=6)
    st1.reset(seed=5)
    st2.reset(seed=6)
    for _ in range(min(a.switch_day + 1, instance.H)):
        action = _common_action(st1)
        st1.step(action)
        st2.step(action)
    frozen_equal = _states_equal(
        static_builder.build(st1),
        static_builder.build(st2),
    )

    # 4) Scenario JSON roundtrip is exact.
    roundtrip_single = scenario_from_dict(scenario_to_dict(s1))
    roundtrip_multi = scenario_from_dict(scenario_to_dict(multi1))
    roundtrip_ok = (
        np.array_equal(
            s1.multiplier_matrix(instance),
            roundtrip_single.multiplier_matrix(instance),
        )
        and np.array_equal(
            multi1.multiplier_matrix(instance),
            roundtrip_multi.multiplier_matrix(instance),
        )
    )

    print(
        "Method C pre-shock observations hide future SINGLE shock identity:",
        pre_single_equal,
    )
    print(
        "Method C pre-shock observations hide future MULTI-shock identity:",
        pre_multi_equal,
    )
    print(
        "Method C post-shock observations expose realized shock:",
        post_single_diff,
    )
    print(
        "Frozen B/D strict-static observations remain shock-independent:",
        frozen_equal,
    )
    print("Scenario serialization roundtrip exact:", roundtrip_ok)

    if not pre_single_equal:
        raise RuntimeError("single-shock future information leakage detected")
    if not pre_multi_equal:
        raise RuntimeError("multi-shock future information leakage detected")
    if not post_single_diff:
        raise RuntimeError("Method C does not observe realized shock")
    if not frozen_equal:
        raise RuntimeError("Frozen B/D observation leakage detected")
    if not roundtrip_ok:
        raise RuntimeError("scenario serialization is not exact")

    print("ALL V6 INFORMATION-LEAKAGE CHECKS PASSED")


if __name__ == "__main__":
    main()
