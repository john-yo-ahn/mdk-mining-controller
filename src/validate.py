"""
Validation Runner — Tests whether the AI controller actually works.

4 tests:
  1. Hold-out: Train on 5 failure types, test on 3 unseen ones
  2. Race:     AI vs simple threshold — who detects failure first?
  3. Blind:    Inject failure into random miner mid-run, can AI find it?
  4. Noise:    Add increasing noise, when does the model break?

Run with: uv run mdk validate
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .config import SimulationConfig, FeatureConfig, MODELS_DIR, DEFAULT_MINER_SPECS
from .synthetic.generator import MiningDataGenerator
from .synthetic.scenarios import SCENARIOS, list_scenarios, get_scenario, apply_scenario_effects
from .pipeline.preprocessing import preprocess_pipeline
from .pipeline.features import build_feature_matrix, get_feature_columns, split_temporal
from .kpi.true_efficiency import compute_all_te_variants
from .models.xgboost_classifier import MinerFailureClassifier
from .models.evaluation import compute_classification_metrics, compute_confusion_matrix


def header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subheader(title: str):
    print(f"\n  {title}")
    print(f"  {'-' * (len(title) + 2)}")


# ─── Test 1: Hold-out ─────────────────────────────────────────────────

def test_holdout(verbose: bool = True) -> Dict:
    """
    Train on 5 failure types, test on 3 the model has never seen.
    Measures generalization — can the model catch novel failure patterns?
    """
    header("TEST 1: HOLD-OUT FAILURE TYPES")

    train_types = ["gradual_degradation", "thermal_runaway", "fan_stall",
                   "psu_degradation", "sudden_chip_failure"]
    holdout_types = ["coolant_restriction", "firmware_oscillation", "connector_corrosion"]

    print(f"  Train on:   {', '.join(train_types)}")
    print(f"  Test on:    {', '.join(holdout_types)} (NEVER SEEN)")

    # Generate training data with only train_types
    config = SimulationConfig(n_miners=20, n_days=5, random_seed=100)
    gen = MiningDataGenerator(config)
    df_all = gen.generate()

    # Split: train data = healthy + train failure types
    train_data = df_all[
        (df_all["failure_type"] == "none") |
        (df_all["failure_type"].isin(train_types))
    ].copy()

    print(f"\n  Training data: {len(train_data):,} rows")
    print(f"  Pre-failure labels: {train_data['is_pre_failure'].sum():,}")

    # Preprocess and featurize training data
    train_clean = preprocess_pipeline(train_data)
    train_te = compute_all_te_variants(train_clean)
    train_features = build_feature_matrix(train_te)
    feature_cols = get_feature_columns(train_features)

    # Temporal split for training
    train_split, val_split = split_temporal(train_features, 0.8)
    X_train = train_split[feature_cols].fillna(0)
    y_train = train_split["is_pre_failure"].astype(int).values
    X_val = val_split[feature_cols].fillna(0)
    y_val = val_split["is_pre_failure"].astype(int).values

    # Train model.
    # Use the legacy recall-target strategy here on purpose: the hold-out
    # test below wants the model to FLAG as much as possible so we can
    # observe whether unseen failure types are caught at all. The
    # precision penalty is acceptable in a generalization probe.
    model = MinerFailureClassifier(n_estimators=150, max_depth=5)
    model.fit(X_train, y_train, X_val, y_val)
    model.optimize_threshold(X_val, y_val, strategy="recall_target", target_recall=0.85)

    # Now test on hold-out types using a SEPARATE dataset
    subheader("Hold-out evaluation")
    config2 = SimulationConfig(n_miners=15, n_days=5, random_seed=200)
    gen2 = MiningDataGenerator(config2)
    df_test_all = gen2.generate()

    results = {}
    for ft in holdout_types:
        # Get miners with this failure type
        test_miners = df_test_all[
            (df_test_all["failure_type"] == ft) |
            (df_test_all["failure_type"] == "none")
        ].copy()

        if len(test_miners) == 0:
            results[ft] = {"detected": False, "reason": "no test data"}
            continue

        test_clean = preprocess_pipeline(test_miners)
        test_te = compute_all_te_variants(test_clean)
        test_features = build_feature_matrix(test_te)

        # Ensure same columns
        for col in feature_cols:
            if col not in test_features.columns:
                test_features[col] = 0
        X_test = test_features[feature_cols].fillna(0)
        y_test = test_features["is_pre_failure"].astype(int).values

        if y_test.sum() == 0:
            results[ft] = {"detected": False, "reason": "no positive labels in test"}
            continue

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Check if any pre-failure rows were detected
        pre_failure_mask = y_test == 1
        detected_any = y_pred[pre_failure_mask].sum() > 0
        detection_rate = y_pred[pre_failure_mask].mean() if pre_failure_mask.sum() > 0 else 0

        # Compute max score on pre-failure rows
        max_score = y_prob[pre_failure_mask].max() if pre_failure_mask.sum() > 0 else 0
        mean_score = y_prob[pre_failure_mask].mean() if pre_failure_mask.sum() > 0 else 0

        results[ft] = {
            "detected": bool(detected_any),
            "detection_rate": float(detection_rate),
            "max_score": float(max_score),
            "mean_score": float(mean_score),
            "n_positive": int(pre_failure_mask.sum()),
            "n_flagged": int(y_pred[pre_failure_mask].sum()),
        }

        status = "[DETECTED]" if detected_any else "[MISSED]"
        print(f"\n    {ft:30s} {status}")
        print(f"      Detection rate: {detection_rate:.0%} of pre-failure windows flagged")
        print(f"      Max score: {max_score:.4f}  Mean score: {mean_score:.4f}")
        print(f"      Positive samples: {pre_failure_mask.sum():,}  Flagged: {y_pred[pre_failure_mask].sum():,}")

    detected_count = sum(1 for r in results.values() if r.get("detected", False))
    print(f"\n  GENERALIZATION SCORE: {detected_count}/{len(holdout_types)} unseen failure types detected")

    return results


# ─── Test 2: AI vs Threshold Race ─────────────────────────────────────

def test_race(verbose: bool = True) -> Dict:
    """
    For each failure: when does AI flag it vs when does a simple threshold flag it?
    Simple threshold = temperature > 85C OR hashrate < 80% nameplate.
    """
    header("TEST 2: AI vs THRESHOLD RACE")

    # Load trained model
    try:
        model = MinerFailureClassifier.load()
    except Exception:
        print("  No trained model found. Run 'uv run mdk train' first.")
        return {}

    config = SimulationConfig(n_miners=20, n_days=7, random_seed=300)
    gen = MiningDataGenerator(config)
    df = gen.generate()

    df_clean = preprocess_pipeline(df)
    df_te = compute_all_te_variants(df_clean)
    df_features = build_feature_matrix(df_te)
    feature_cols = get_feature_columns(df_features)

    X = df_features[feature_cols].fillna(0)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= model.threshold_).astype(int)

    df_features = df_features.copy()
    df_features["ai_score"] = y_prob
    df_features["ai_flag"] = y_pred

    # Simple threshold rule
    nameplate_map = {k: v.hashrate_nameplate_th for k, v in DEFAULT_MINER_SPECS.items()}
    df_features["nameplate"] = df_features["model"].map(nameplate_map)
    df_features["threshold_flag"] = (
        (df_features["temperature_c"] > 85) |
        (df_features["hashrate_th"] < df_features["nameplate"] * 0.80)
    ).astype(int)

    results = {}
    print(f"\n  {'Failure Type':<30s} {'AI Lead':>10s} {'Threshold Lead':>15s} {'Winner':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*15} {'-'*10}")

    for miner_id in df_features["miner_id"].unique():
        miner_df = df_features[df_features["miner_id"] == miner_id].sort_values("timestamp")
        failure_rows = miner_df[miner_df["failure_type"] != "none"]
        if len(failure_rows) == 0:
            continue

        failure_type = failure_rows.iloc[0]["failure_type"]
        onset_time = failure_rows.iloc[0]["timestamp"]

        # When did AI first flag (before onset)?
        ai_flags = miner_df[(miner_df["timestamp"] < onset_time) & (miner_df["ai_flag"] == 1)]
        ai_first = ai_flags.iloc[0]["timestamp"] if len(ai_flags) > 0 else None

        # When did threshold first flag (before onset)?
        thr_flags = miner_df[(miner_df["timestamp"] < onset_time) & (miner_df["threshold_flag"] == 1)]
        thr_first = thr_flags.iloc[0]["timestamp"] if len(thr_flags) > 0 else None

        ai_lead_hrs = (onset_time - ai_first).total_seconds() / 3600 if ai_first else 0
        thr_lead_hrs = (onset_time - thr_first).total_seconds() / 3600 if thr_first else 0

        if ai_lead_hrs > thr_lead_hrs:
            winner = "AI"
        elif thr_lead_hrs > ai_lead_hrs:
            winner = "THRESHOLD"
        else:
            winner = "TIE"

        results[miner_id] = {
            "failure_type": failure_type,
            "ai_lead_hours": ai_lead_hrs,
            "threshold_lead_hours": thr_lead_hrs,
            "winner": winner,
        }

        ai_str = f"+{ai_lead_hrs:.1f}h" if ai_lead_hrs > 0 else "missed"
        thr_str = f"+{thr_lead_hrs:.1f}h" if thr_lead_hrs > 0 else "missed"
        print(f"  {failure_type:<30s} {ai_str:>10s} {thr_str:>15s} {winner:>10s}")

    ai_wins = sum(1 for r in results.values() if r["winner"] == "AI")
    total = len(results)
    avg_ai = np.mean([r["ai_lead_hours"] for r in results.values() if r["ai_lead_hours"] > 0]) if results else 0
    avg_thr = np.mean([r["threshold_lead_hours"] for r in results.values() if r["threshold_lead_hours"] > 0]) if results else 0

    print(f"\n  AI wins: {ai_wins}/{total} scenarios")
    print(f"  Average AI lead time: {avg_ai:.1f} hours")
    print(f"  Average threshold lead time: {avg_thr:.1f} hours")

    return results


# ─── Test 3: Blind Injection ──────────────────────────────────────────

def test_blind(verbose: bool = True) -> Dict:
    """
    Run healthy fleet, inject a failure at random time into random miner.
    Can the AI find it without being told which miner or when?
    """
    header("TEST 3: BLIND INJECTION")

    try:
        model = MinerFailureClassifier.load()
    except Exception:
        print("  No trained model found. Run 'uv run mdk train' first.")
        return {}

    # Generate healthy-only data
    config = SimulationConfig(n_miners=12, n_days=5, random_seed=400)
    config.failure_fraction = 0.0  # NO failures
    gen = MiningDataGenerator(config)
    df_healthy = gen.generate()

    # Pick a random miner and inject a failure manually
    rng = np.random.default_rng(42)
    target_miner = f"MNR-{rng.integers(1, 13):03d}"
    injection_step = config.n_steps // 2  # halfway through
    scenario = get_scenario("connector_corrosion")

    print(f"  Fleet: {config.n_miners} miners, all healthy")
    print(f"  Injecting '{scenario.name}' into {target_miner} at step {injection_step}")
    print(f"  Model must find it without being told which miner or when\n")

    # Apply the scenario to the target miner's telemetry
    target_mask = df_healthy["miner_id"] == target_miner
    target_df = df_healthy[target_mask].copy()
    duration = 1500

    for idx, (_, row) in enumerate(target_df.iterrows()):
        step = idx
        if step < injection_step:
            continue

        values = {
            "voltage": row["voltage_v"],
            "power": row["power_w"],
            "temperature": row["temperature_c"],
            "degradation_factor": 1.0,
            "thermal_resistance": 0.012,
        }
        modified = apply_scenario_effects(scenario, step, injection_step, duration, values, rng)

        df_healthy.loc[_, "voltage_v"] = modified.get("voltage", row["voltage_v"])
        df_healthy.loc[_, "power_w"] = modified.get("power", row["power_w"])
        df_healthy.loc[_, "temperature_c"] = modified.get("temperature", row["temperature_c"])

        # Mark failure for evaluation
        if step >= injection_step:
            df_healthy.loc[_, "failure_type"] = scenario.name
        if injection_step - 24*60 <= step < injection_step:
            df_healthy.loc[_, "is_pre_failure"] = True

    # Run through pipeline
    df_clean = preprocess_pipeline(df_healthy)
    df_te = compute_all_te_variants(df_clean)
    df_features = build_feature_matrix(df_te)
    feature_cols = get_feature_columns(df_features)

    X = df_features[feature_cols].fillna(0)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= model.threshold_).astype(int)

    df_features = df_features.copy()
    df_features["ai_score"] = y_prob
    df_features["ai_flag"] = y_pred

    # Analyze: which miners got flagged?
    subheader("Detection results")

    flag_counts = df_features.groupby("miner_id")["ai_flag"].sum()
    max_scores = df_features.groupby("miner_id")["ai_score"].max()

    for miner_id in sorted(df_features["miner_id"].unique()):
        flags = flag_counts.get(miner_id, 0)
        max_s = max_scores.get(miner_id, 0)
        is_target = miner_id == target_miner
        marker = " <── INJECTED" if is_target else ""
        status = "FLAGGED" if flags > 0 else "clean"
        print(f"    {miner_id}: {status:>7s}  (flags={flags:>4d}, max_score={max_s:.4f}){marker}")

    # Was the target miner the most flagged?
    target_flags = flag_counts.get(target_miner, 0)
    target_rank = sorted(flag_counts.values, reverse=True).index(target_flags) + 1 if target_flags > 0 else -1

    # False positive rate on healthy miners
    healthy_miners = [m for m in flag_counts.index if m != target_miner]
    false_flagged = sum(1 for m in healthy_miners if flag_counts[m] > 0)

    print(f"\n  Target detected: {'YES' if target_flags > 0 else 'NO'}")
    print(f"  Target rank by flag count: #{target_rank} of {config.n_miners}")
    print(f"  False flags on healthy miners: {false_flagged}/{len(healthy_miners)} ({false_flagged/max(len(healthy_miners),1):.0%})")

    return {
        "target_miner": target_miner,
        "scenario": scenario.name,
        "detected": target_flags > 0,
        "target_flags": int(target_flags),
        "target_rank": target_rank,
        "false_positive_rate": false_flagged / max(len(healthy_miners), 1),
    }


# ─── Test 4: Noise Resilience ─────────────────────────────────────────

def test_noise(verbose: bool = True) -> Dict:
    """
    Take a known failure pattern, add increasing noise levels.
    At what noise level does the model stop detecting it?
    """
    header("TEST 4: NOISE RESILIENCE")

    try:
        model = MinerFailureClassifier.load()
    except Exception:
        print("  No trained model found. Run 'uv run mdk train' first.")
        return {}

    # Generate base data with known failures
    config = SimulationConfig(n_miners=10, n_days=5, random_seed=500)
    gen = MiningDataGenerator(config)
    df_base = gen.generate()

    noise_levels = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]
    results = {}

    print(f"  Testing model robustness across {len(noise_levels)} noise levels\n")
    print(f"  {'Noise Level':>12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Pre-fail detected':>18s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*18}")

    for noise in noise_levels:
        df = df_base.copy()

        # Add Gaussian noise to sensor columns
        if noise > 0:
            rng = np.random.default_rng(42)
            for col in ["temperature_c", "hashrate_th", "power_w", "voltage_v"]:
                std = df[col].std() * noise
                df[col] += rng.normal(0, std, len(df))
                df[col] = df[col].clip(0)

        # Pipeline
        df_clean = preprocess_pipeline(df)
        df_te = compute_all_te_variants(df_clean)
        df_features = build_feature_matrix(df_te)
        feature_cols = get_feature_columns(df_features)

        X = df_features[feature_cols].fillna(0)
        y_test = df_features["is_pre_failure"].astype(int).values
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        if len(np.unique(y_test)) < 2 or y_test.sum() == 0:
            results[noise] = {"precision": 0, "recall": 0, "f1": 0}
            continue

        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        n_detected = y_pred[y_test == 1].sum()
        n_total = (y_test == 1).sum()

        results[noise] = metrics
        print(f"  {noise:>11.0%} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1']:>10.3f} {n_detected:>8d}/{n_total:<8d}")

    # Find breaking point
    recalls = [(n, r.get("recall", 0)) for n, r in results.items()]
    breaking = next((n for n, r in recalls if r < 0.5), None)

    if breaking:
        print(f"\n  Model breaks at {breaking:.0%} noise (recall drops below 50%)")
    else:
        print(f"\n  Model survives all noise levels tested")

    return results


# ─── Main ─────────────────────────────────────────────────────────────

def main(tests: list = None):
    if tests is None:
        tests = ["holdout", "race", "blind", "noise"]

    all_results = {}
    start = time.time()

    if "holdout" in tests:
        all_results["holdout"] = test_holdout()

    if "race" in tests:
        all_results["race"] = test_race()

    if "blind" in tests:
        all_results["blind"] = test_blind()

    if "noise" in tests:
        all_results["noise"] = test_noise()

    header("VALIDATION COMPLETE")
    print(f"  Total time: {time.time() - start:.0f}s")

    return all_results


if __name__ == "__main__":
    main()
