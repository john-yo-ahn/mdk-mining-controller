"""
Unit tests for the True Efficiency KPI formula after the Level 1
assignment-compliance rewrite.

This script tests the formula in `src/kpi/true_efficiency.py` without
requiring pytest — it uses plain assertions and runs via
`uv run python -m scripts.test_te_formula`. Matches the testing
pattern of `scripts/test_live_feature_parity.py`.

Coverage:
  * baseline case (default voltage, Normal mode) numerically
    reproduces the formula behavior of a reference calculation
  * each of the 4 assignment §3.1.b variables demonstrably changes
    the output when toggled independently
  * Shutdown / Idle operating_mode_factor zeroes out the KPI
  * voltage deviation produces a bounded penalty (never negative)
  * scalar helpers agree with the vectorized path on the same inputs
  * compute_all_te_variants round-trips a synthetic DataFrame cleanly
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from src.config import TEConfig, DEFAULT_MINER_SPECS
from src.kpi.true_efficiency import (
    compute_te_base,
    compute_te_base_scalar,
    compute_te_adjusted,
    compute_te_adjusted_scalar,
    compute_te_health,
    compute_te_health_scalar,
    compute_all_te_variants,
)


# Simple test runner: each function returns None on success, raises
# AssertionError on failure. main() catches and tallies.

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        def wrapper():
            print(f"\n── {name} ──")
            try:
                fn()
                RESULTS.append((name, True, "ok"))
                print("  PASS")
            except AssertionError as e:
                RESULTS.append((name, False, f"ASSERTION: {e}"))
                print(f"  FAIL  {e}")
            except Exception as e:
                RESULTS.append((name, False, f"EXCEPTION: {type(e).__name__}: {e}"))
                print(f"  ERROR {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


# Reference values used across tests — an Antminer S21 Pro at spec.
SPEC_HASHRATE = 234.0
SPEC_POWER = 3510.0
SPEC_VOLTAGE = 0.38
SPEC_AMBIENT = 25.0
SPEC_NAMEPLATE = 234.0
CFG = TEConfig()


@check("1. baseline case (default voltage, Normal mode) matches reference math")
def test_baseline():
    """
    At default voltage and Normal mode, voltage_stability_factor=1 and
    operating_mode_factor=1, so te_base should equal
        hashrate / (power × (1 + α + β))
    which is the pre-Level-1 formula.
    """
    expected = SPEC_HASHRATE / (SPEC_POWER * (1 + CFG.alpha_cooling + CFG.beta_infra))
    got = compute_te_base_scalar(
        hashrate_th=SPEC_HASHRATE,
        power_chip_w=SPEC_POWER,
        voltage_v=SPEC_VOLTAGE,
        voltage_default_v=SPEC_VOLTAGE,
        operating_mode="Normal",
        config=CFG,
    )
    assert math.isclose(got, expected, rel_tol=1e-9), (
        f"te_base at spec should be {expected:.6f}, got {got:.6f}"
    )
    # Sanity: should be ~0.0556 TH/s per W at spec
    assert 0.05 < got < 0.06, f"expected te_base ~0.0556, got {got:.6f}"


@check("2. Shutdown mode zeroes out te_base (variable #4 works)")
def test_shutdown_zeroes_out():
    got = compute_te_base_scalar(
        hashrate_th=SPEC_HASHRATE,
        power_chip_w=SPEC_POWER,
        voltage_v=SPEC_VOLTAGE,
        voltage_default_v=SPEC_VOLTAGE,
        operating_mode="Shutdown",
        config=CFG,
    )
    assert got == 0.0, f"Shutdown should zero te_base, got {got}"

    # And Idle
    got_idle = compute_te_base_scalar(
        hashrate_th=SPEC_HASHRATE,
        power_chip_w=SPEC_POWER,
        voltage_v=SPEC_VOLTAGE,
        voltage_default_v=SPEC_VOLTAGE,
        operating_mode="Idle",
        config=CFG,
    )
    assert got_idle == 0.0, f"Idle should zero te_base, got {got_idle}"

    # And unknown modes fall back to 1.0
    got_unknown = compute_te_base_scalar(
        hashrate_th=SPEC_HASHRATE,
        power_chip_w=SPEC_POWER,
        voltage_v=SPEC_VOLTAGE,
        voltage_default_v=SPEC_VOLTAGE,
        operating_mode="Turbo",  # not in the weights dict
        config=CFG,
    )
    baseline = compute_te_base_scalar(
        hashrate_th=SPEC_HASHRATE, power_chip_w=SPEC_POWER,
        voltage_v=SPEC_VOLTAGE, voltage_default_v=SPEC_VOLTAGE,
        operating_mode="Normal", config=CFG,
    )
    assert math.isclose(got_unknown, baseline, rel_tol=1e-9), (
        f"unknown mode should fall back to 1.0, got {got_unknown} vs baseline {baseline}"
    )


@check("3. voltage deviation penalty behaves correctly (variable #2)")
def test_voltage_deviation():
    """
    At 20% above spec voltage, penalty is 0.5 × 0.2 = 0.10 → factor 0.90.
    At 20% below spec voltage, same penalty → factor 0.90.
    At spec voltage, factor 1.0.
    At 200% deviation, factor would be negative but clip to 0.
    """
    at_spec = compute_te_base_scalar(
        SPEC_HASHRATE, SPEC_POWER, 0.38, 0.38, "Normal", CFG,
    )
    plus_20pct = compute_te_base_scalar(
        SPEC_HASHRATE, SPEC_POWER, 0.456, 0.38, "Normal", CFG,  # 0.456 = 0.38 × 1.2
    )
    minus_20pct = compute_te_base_scalar(
        SPEC_HASHRATE, SPEC_POWER, 0.304, 0.38, "Normal", CFG,  # 0.304 = 0.38 × 0.8
    )

    # +20% and -20% should give symmetric penalty
    assert math.isclose(plus_20pct, minus_20pct, rel_tol=1e-9), (
        f"+20% ({plus_20pct:.6f}) and -20% ({minus_20pct:.6f}) should be symmetric"
    )
    # Penalty should be exactly 10% at ±20% deviation (with coefficient 0.5)
    expected_penalty = 0.90
    assert math.isclose(plus_20pct, at_spec * expected_penalty, rel_tol=1e-9), (
        f"+20% deviation should apply {(1-expected_penalty)*100:.0f}% penalty, "
        f"got {plus_20pct / at_spec:.4f}"
    )

    # Extreme deviation clips to 0 (never goes negative)
    extreme = compute_te_base_scalar(
        SPEC_HASHRATE, SPEC_POWER, 5.0, 0.38, "Normal", CFG,  # 13× spec voltage
    )
    assert extreme >= 0.0, f"extreme voltage should clip to >= 0, got {extreme}"
    assert extreme < at_spec * 0.1, f"extreme voltage should approach 0, got {extreme}"


@check("4. cooling factor in denominator (variable #1)")
def test_cooling_factor():
    """
    Sweeping α_cooling from 0 to 0.30 should monotonically decrease te_base.
    """
    hashrate, power, v, vd, mode = 234.0, 3510.0, 0.38, 0.38, "Normal"
    low = compute_te_base_scalar(
        hashrate, power, v, vd, mode,
        TEConfig(alpha_cooling=0.0, beta_infra=0.0),
    )
    med = compute_te_base_scalar(
        hashrate, power, v, vd, mode,
        TEConfig(alpha_cooling=0.15, beta_infra=0.05),
    )
    high = compute_te_base_scalar(
        hashrate, power, v, vd, mode,
        TEConfig(alpha_cooling=0.30, beta_infra=0.10),
    )
    assert low > med > high, (
        f"Higher cooling overhead should reduce te_base: {low} > {med} > {high}"
    )
    # Specifically: no cooling overhead → 1.20× higher than the default
    assert math.isclose(low / med, 1.20, rel_tol=1e-6), (
        f"Zero-overhead should be 1.20× default, got {low/med:.4f}"
    )


@check("5. environmental (temperature) penalty (variable #3)")
def test_temperature_penalty():
    """
    te_adjusted applies the temperature penalty on top of te_base.
    At ambient = baseline (25°C) the penalty is 1.0.
    At 45°C ambient the penalty is 1 - 0.008 × 20 = 0.84.
    """
    base = 0.0556
    at_baseline = compute_te_adjusted_scalar(base, 25.0, CFG)
    hot = compute_te_adjusted_scalar(base, 45.0, CFG)
    cold = compute_te_adjusted_scalar(base, 10.0, CFG)

    assert math.isclose(at_baseline, base, rel_tol=1e-9), (
        f"at baseline should equal te_base, got {at_baseline}"
    )
    assert math.isclose(hot, base * 0.84, rel_tol=1e-9), (
        f"at 45°C should be 0.84× base, got {hot:.6f} (expected {base*0.84:.6f})"
    )
    # Cold ambient should NOT reward the miner (penalty clipped at ≤1)
    assert math.isclose(cold, base, rel_tol=1e-9), (
        f"at 10°C (below baseline) should equal te_base, got {cold}"
    )


@check("6. all 4 variables independently change te_base output")
def test_all_four_variables_matter():
    """
    Sweep each of the four assignment variables independently with
    everything else fixed, and verify each produces a distinct output.
    """
    base = compute_te_base_scalar(
        234.0, 3510.0, 0.38, 0.38, "Normal", CFG,
    )

    # 1. cooling overhead differs
    cooling_changed = compute_te_base_scalar(
        234.0, 3510.0, 0.38, 0.38, "Normal",
        TEConfig(alpha_cooling=0.30, beta_infra=0.05),
    )
    assert cooling_changed != base, "cooling factor must change te_base"

    # 2. voltage differs
    voltage_changed = compute_te_base_scalar(
        234.0, 3510.0, 0.42, 0.38, "Normal", CFG,
    )
    assert voltage_changed != base, "voltage deviation must change te_base"

    # 3. ambient temp differs (via te_adjusted)
    temp_changed = compute_te_adjusted_scalar(base, 40.0, CFG)
    assert temp_changed != base, "ambient temperature must change te_adjusted"

    # 4. operating mode differs
    mode_changed = compute_te_base_scalar(
        234.0, 3510.0, 0.38, 0.38, "Idle", CFG,
    )
    assert mode_changed != base, "operating mode must change te_base"


@check("7. zero power produces zero te_base (no divide by zero)")
def test_zero_power_is_zero():
    got = compute_te_base_scalar(234.0, 0.0, 0.38, 0.38, "Normal", CFG)
    assert got == 0.0, f"zero power should yield zero te_base, got {got}"

    got_neg = compute_te_base_scalar(234.0, -10.0, 0.38, 0.38, "Normal", CFG)
    assert got_neg == 0.0, f"negative power should yield zero te_base, got {got_neg}"


@check("8. scalar helpers agree with vectorized path on the same inputs")
def test_scalar_matches_vector():
    """
    Build a small DataFrame and run compute_te_base (vectorized) +
    compute_te_base_scalar on each row; the two must agree row-by-row.
    """
    rows = [
        dict(hashrate=234.0, power=3510.0, v=0.38, vd=0.38, mode="Normal"),
        dict(hashrate=220.0, power=3400.0, v=0.35, vd=0.38, mode="Normal"),
        dict(hashrate=180.0, power=2800.0, v=0.32, vd=0.38, mode="Normal"),
        dict(hashrate=0.0,   power=0.0,    v=0.00, vd=0.38, mode="Shutdown"),
        dict(hashrate=234.0, power=3510.0, v=0.40, vd=0.38, mode="Idle"),
    ]
    df = pd.DataFrame(rows)

    vec = compute_te_base(
        df["hashrate"], df["power"], df["v"], df["vd"], df["mode"], CFG,
    )

    for i, row in df.iterrows():
        sca = compute_te_base_scalar(
            row["hashrate"], row["power"], row["v"], row["vd"], row["mode"], CFG,
        )
        v_val = float(vec[i])
        assert math.isclose(sca, v_val, rel_tol=1e-6, abs_tol=1e-9), (
            f"row {i}: scalar {sca} vs vector {v_val}"
        )


@check("9. compute_all_te_variants on synthetic DataFrame works end-to-end")
def test_full_pipeline_on_df():
    """
    A mini DataFrame with all required columns should produce te_base,
    te_adjusted, te_health without errors and with sensible numeric
    properties.
    """
    df = pd.DataFrame([
        # Healthy Pro miner
        dict(miner_id="M1", model="Pro", hashrate_th=234.0, power_w=3510.0,
             voltage_v=0.38, ambient_temperature_c=25.0, operating_mode="Normal",
             failure_type="none"),
        # Same Pro miner but 10°C hotter ambient
        dict(miner_id="M2", model="Pro", hashrate_th=234.0, power_w=3510.0,
             voltage_v=0.38, ambient_temperature_c=35.0, operating_mode="Normal",
             failure_type="none"),
        # Failing Pro miner (degraded hashrate)
        dict(miner_id="M3", model="Pro", hashrate_th=117.0, power_w=3510.0,
             voltage_v=0.38, ambient_temperature_c=25.0, operating_mode="Normal",
             failure_type="connector_corrosion"),
        # Shutdown
        dict(miner_id="M4", model="Pro", hashrate_th=0.0, power_w=0.0,
             voltage_v=0.0, ambient_temperature_c=25.0, operating_mode="Shutdown",
             failure_type="none"),
    ])

    result = compute_all_te_variants(df)

    # Required columns present
    for col in ("te_base", "te_adjusted", "te_health",
                "hashrate_realization", "hashrate_nameplate_th",
                "voltage_default_v"):
        assert col in result.columns, f"missing column {col}"

    # M1: healthy baseline — all three TE layers should be positive and equal
    # (no temp penalty at 25, realization=1)
    assert result.loc[0, "te_base"] > 0
    assert math.isclose(
        result.loc[0, "te_base"], result.loc[0, "te_adjusted"], rel_tol=1e-9,
    ), "at baseline ambient, te_base should equal te_adjusted"
    assert math.isclose(
        result.loc[0, "te_adjusted"], result.loc[0, "te_health"], rel_tol=1e-9,
    ), "at full realization, te_adjusted should equal te_health"

    # M2: hotter ambient → te_adjusted < te_base
    assert result.loc[1, "te_adjusted"] < result.loc[1, "te_base"], (
        "hotter ambient should reduce te_adjusted vs te_base"
    )

    # M3: degraded hashrate → te_health much less than te_adjusted
    assert result.loc[2, "te_health"] < result.loc[2, "te_adjusted"] * 0.6, (
        "50% realization should reduce te_health by ~half"
    )

    # M4: shutdown → all zero
    assert result.loc[3, "te_base"] == 0.0
    assert result.loc[3, "te_adjusted"] == 0.0
    assert result.loc[3, "te_health"] == 0.0


@check("10. all 4 hardware models resolve voltage_default_v correctly")
def test_all_hardware_models_resolve():
    """
    The short-token model keys ("Pro", "M56S", "M63", "XP") must
    all map to a voltage_default_v and a hashrate_nameplate_th
    via DEFAULT_MINER_SPECS.
    """
    df = pd.DataFrame([
        dict(miner_id="A", model="Pro",  hashrate_th=234.0, power_w=3510.0,
             voltage_v=0.38, ambient_temperature_c=25.0, operating_mode="Normal",
             failure_type="none"),
        dict(miner_id="B", model="M56S", hashrate_th=194.0, power_w=3360.0,
             voltage_v=0.40, ambient_temperature_c=25.0, operating_mode="Normal",
             failure_type="none"),
        dict(miner_id="C", model="M63",  hashrate_th=390.0, power_w=7440.0,
             voltage_v=0.36, ambient_temperature_c=25.0, operating_mode="Normal",
             failure_type="none"),
        dict(miner_id="D", model="XP",   hashrate_th=140.0, power_w=3000.0,
             voltage_v=0.42, ambient_temperature_c=25.0, operating_mode="Normal",
             failure_type="none"),
    ])
    out = compute_all_te_variants(df)

    for i, expected_default in enumerate([0.38, 0.40, 0.36, 0.42]):
        got_default = out.loc[i, "voltage_default_v"]
        assert math.isclose(got_default, expected_default, rel_tol=1e-6), (
            f"row {i}: voltage_default_v should be {expected_default}, got {got_default}"
        )
        assert out.loc[i, "hashrate_nameplate_th"] > 0, (
            f"row {i}: nameplate should resolve, got {out.loc[i, 'hashrate_nameplate_th']}"
        )
        assert out.loc[i, "te_base"] > 0, (
            f"row {i}: te_base should be positive at baseline, got {out.loc[i, 'te_base']}"
        )


def main() -> int:
    print("=" * 70)
    print("  True Efficiency KPI — Level 1 unit tests")
    print("=" * 70)

    test_baseline()
    test_shutdown_zeroes_out()
    test_voltage_deviation()
    test_cooling_factor()
    test_temperature_penalty()
    test_all_four_variables_matter()
    test_zero_power_is_zero()
    test_scalar_matches_vector()
    test_full_pipeline_on_df()
    test_all_hardware_models_resolve()

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        if not ok:
            print(f"         {detail}")
    print()
    print(f"  Total: {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
