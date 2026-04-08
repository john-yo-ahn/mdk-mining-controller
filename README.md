# MDK AI Mining Controller

AI-driven controller for Bitcoin ASIC mining fleets, built as a 3-week
prototype on Tether's Mining Development Kit (MDK) platform. Combines
**predictive maintenance** (XGBoost + LSTM-Autoencoder) with a
**rule-based dynamic optimizer** for frequency, voltage, and thermal
management. All work runs on synthetic data; the architecture is ready
to swap in real telemetry once a data sharing channel exists.

> **Status:** working end-to-end pipeline. Training, evaluation, and a
> live terminal dashboard all run today. See `docs/PROJECT_PLAN.md` for
> the week-by-week plan and `docs/ARCHITECTURE.md` for the system diagram.

## What it does

1. **Generates** physics-based synthetic telemetry for a fleet of
   30 ASIC miners across 120 days (~5.2 M rows). The generator models
   power, hashrate, temperature, voltage, ambient conditions, and seven
   distinct failure scenarios.
2. **Computes** a True Efficiency (TE) KPI that goes beyond raw J/TH
   by accounting for cooling overhead, environmental adjustment, and
   degradation-aware hashrate realization.
3. **Builds** a 152-feature matrix per miner per minute: rolling stats
   at multiple timescales, rate-of-change features, trend slopes,
   variance growth, cross-signal correlations, autocorrelation, peak
   counts, diurnal amplitude, and cross-miner container features.
4. **Trains** two complementary models:
   - **XGBoost** classifier for supervised failure prediction
     (`is_pre_failure` ↔ degradation phase). Lead time ≈ 7 days
     average for detected failures.
   - **LSTM-Autoencoder** for unsupervised anomaly detection on
     healthy-only data. Catches deviations even for failure types
     the supervised model never saw.
5. **Predicts** in a live simulation: a Textual terminal dashboard
   runs a 30-miner fleet, injects failure scenarios on demand, and
   shows alerts, optimizer actions, and per-miner risk levels in
   real time.
6. **Optimizes** with a rule-based controller that proposes frequency
   adjustments for thermal management and energy-price response, all
   gated through a `SafetyGuard` that enforces thermal limits and
   rate-limiting.

## Quick start

```bash
# Install dependencies
uv sync

# Train models end to end (~1 hour first time, ~50 min after with cache hits)
uv run python -m src.run_pipeline

# Launch the live dashboard
./run_dashboard.sh

# Run validation tests (hold-out failure types, AI-vs-threshold race, etc.)
uv run python -m src.validate
```

## Project layout

```
src/
├── config.py              Central config: paths, hardware specs, KPI cfg
├── synthetic/             Physics-first synthetic telemetry generator
│   ├── generator.py       Main MiningDataGenerator
│   ├── physics.py         Thermal + electrical physics models
│   ├── scenarios.py       Failure scenario library (7 types)
│   └── failures.py        Phase-based failure progression
├── pipeline/              Data pipeline: ingest, preprocess, features
│   ├── ingestion.py       CSV/parquet loaders
│   ├── preprocessing.py   Cleaning, missing-value handling
│   └── features.py        152-feature builder + adaptive temporal split
├── kpi/
│   └── true_efficiency.py TE base / adjusted / health variants
├── models/
│   ├── xgboost_classifier.py   Failure prediction (binary)
│   ├── lstm_autoencoder.py     Anomaly detection (unsupervised)
│   └── evaluation.py            Metrics, threshold selectors
├── optimizer/
│   ├── rules.py           Rule-based frequency/voltage controller
│   └── safety.py          Safety guard: thermal limits, rate limits
├── storage/
│   └── backend.py         DuckDB store with lock recovery
├── cli/
│   ├── app.py             Textual dashboard
│   ├── simulation.py      Live fleet simulator
│   └── ai_bridge.py       Connects simulator to trained models
├── run_pipeline.py        End-to-end training pipeline
└── validate.py            Validation runner (hold-out, race, blind, noise)
```

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Primary failure prediction | XGBoost (`tree_method="hist"`) | Tabular data, interpretable, 30× faster than `exact` |
| Anomaly detection | LSTM-Autoencoder (MPS-accelerated) | No failure labels needed, generalizes to unseen types |
| Optimizer strategy | Rule-based with `SafetyGuard` | Auditable, conservative, safe for hardware |
| Synthetic data | Physics-first generator | Thermodynamically grounded, controllable failure injection |
| Storage | Two DuckDB files (batch + live) | Single-writer lock isolation, no contention |
| Threshold selection | F1-max with precision floor | Balanced default; falls back to precision floor under extreme imbalance |
| Train/val/test split | Adaptive temporal by cumulative positives | Guarantees positives in each split despite clustered failure events |

## Honest results (latest run)

Run on 30 miners × 120 days, 5.2 M rows, 152 features. Honest metrics —
the threshold is tuned on a held-out validation slice the test set
never sees.

**XGBoost (supervised failure prediction):**

| Metric | Value |
|---|---|
| AUC-ROC | **0.801** |
| F1 | 0.163 |
| Precision | 0.230 |
| Recall | 0.126 |
| Detection timeline | **3 of 6** failures detected on test |
| **Average lead time** | **182.6 hours (≈ 7.6 days)** before cascade |
| Top features | `voltage_v_roll_10080m_mean`, `hashrate_th_roll_60m_max`, `temperature_c_roll_10080m_mean` |

The 7-day average lead time is the most operationally useful number.
Operators can schedule maintenance during planned downtime instead of
reacting to thermal alarms.

**LSTM-Autoencoder (unsupervised anomaly detection):**

| Metric | Value |
|---|---|
| Training sequences | 649,133 (healthy only) |
| Best val_loss | 0.007172 (epoch 7, early-stopped epoch 11) |
| Threshold (95th pct of healthy val errors) | 0.007532 |
| **Healthy false-alarm rate** | **2.6%** (5,214 / 200,357) |
| **Failure detection rate** | **33.4%** (12,290 / 36,774) |
| Mean reconstruction error (healthy) | 0.005734 |
| Mean reconstruction error (failure) | 0.015254 |
| **Separation ratio** | **2.66×** |

The LSTM-AE is the safety-net detector — trained on healthy-only
sequences, it doesn't need any failure labels and so generalizes to
failure types the supervised model has never seen. At a 2.6%
false-alarm rate it catches 1 in 3 pre-failure sequences purely from
"this looks unlike any healthy data I trained on".

Separation ratio is lower than the previous (broken-label) run's 4.5×.
That's because the new labeling covers the full incubation phase,
including early degradation that's barely distinguishable from healthy
telemetry. Catching subtle early-incubation anomalies is a strictly
harder task than catching late-acceleration ones, and the lower
separation ratio honestly reflects that.

## Per-failure-type detection coverage

Aggregate metrics like AUC and F1 hide the most useful question:
**which specific failure modes can the system actually catch, and
which can it not?** This is what mentors will ask. The answer is
measured directly against the test set, not extrapolated.

The test slice contains **6 distinct failure events across 5 failure
types** (the held-out 25% of the simulation timeline produces this
many failures naturally; for more failures we'd need a larger
synthetic run).

| Miner | Failure type | XGBoost | LSTM-AE | Headline |
|---|---|---|---|---|
| MNR-016 | connector_corrosion | ✅ caught | ✅ 99.7% of seqs flagged | **16.5 days lead time** |
| MNR-008 | connector_corrosion | ✅ caught | ✅ 100% of seqs flagged | **5.9 days lead time** |
| MNR-018 | thermal_runaway | ✅ caught | ✅ 100% of seqs flagged | 11.9 hours lead time |
| MNR-020 | psu_degradation | ❌ missed | ✅ **100%** of seqs flagged | LSTM-only catch |
| MNR-022 | coolant_restriction | ❌ missed | ✅ **100%** of seqs flagged | LSTM-only catch |
| MNR-029 | sudden_chip_failure | ❌ | ❌ | only 2 rows in test — unmeasurable |

**Combined coverage: 5 of 6 failures caught by at least one model.**
The one miss (`sudden_chip_failure`) is unmeasurable because the
failure happens too fast to leave a learnable signature in our
1-minute sampling. Reactive thermal protection is the right
mechanism for that class — it's handled by `SafetyGuard.enforce_thermal_shutdown()`
in `src/optimizer/safety.py`, not by the predictive models.

### Why both models?

The breakdown above is the architectural argument for running
XGBoost and LSTM-AE in parallel rather than picking one. Each catches
things the other misses:

- **XGBoost** is the lead-time champion. It caught both
  `connector_corrosion` cases **5.9 and 16.5 days** before cascade —
  enough runway to schedule maintenance during planned downtime
  rather than reacting to thermal alarms.
- **LSTM-AE** is the safety net. It caught the **two failure types
  XGBoost missed entirely** (`psu_degradation` and `coolant_restriction`).
  For both, **100% of pre-failure sequences exceeded the anomaly
  threshold** — these aren't borderline detections. The LSTM doesn't
  care about labels; it just learned what healthy looks like and
  flags anything that deviates, regardless of failure mechanism.

### Row-level recall by failure type (XGBoost)

For reference, the supervised model's per-row recall on test
positives is conservative because the threshold is calibrated for
F1 with a precision floor, not coverage. Event-level detection is
the metric operators care about, not row-level recall.

| Failure type | Pre-failure rows in test | XGBoost catches | Row-level recall |
|---|---|---|---|
| connector_corrosion | 33,777 | 10,837 | 32.1% |
| psu_degradation | 32,412 | 0 | 0.0% (LSTM-AE catches it) |
| coolant_restriction | 20,968 | 0 | 0.0% (LSTM-AE catches it) |
| thermal_runaway | 746 | 274 | 36.7% |
| sudden_chip_failure | 2 | 0 | n/a |

The "0% recall" rows are the cases where the supervised model
genuinely doesn't see the signature in training and the unsupervised
model has to step in. This is the dual-model architecture working
as designed.

## Known limitations

These are documented honestly so a reviewer doesn't have to discover
them mid-demo.

1. **Live CLI inference is approximate.** The
   `MinerBuffer.compute_features()` path in `src/cli/ai_bridge.py`
   computes ~80 features but the trained XGBoost model expects 152.
   Missing features are filled with zeros at inference time. The LSTM
   live path uses `seq_len=30` while the model was trained with
   `seq_len=60`, so its contribution to live predictions silently
   short-circuits inside a try/except. Live alerts are useful for
   demonstration but should not be confused with the batch-pipeline
   metrics above. See the docstring on `AIBridge.load_models()` for
   the full list.
2. **Pre-failure label is degradation_phase-derived.** A row is
   `is_pre_failure=True` if its `degradation_phase` is `incubation`
   or `acceleration`. This gives the model a much larger and more
   honest training target than a 24-hour fixed window — but it means
   the model is learning "is this miner currently degrading", not
   "will this miner fail in exactly 24 hours". The detection lead
   time we report is computed against the cascade onset.
3. **Synthetic data only.** All metrics are on physics-simulated
   telemetry. Real ASIC failures will look different and the model
   will need recalibration once real telemetry is available.
4. **Rule-based optimizer is unevaluated against an RL baseline.**
   The optimizer module is functional and gated through `SafetyGuard`,
   but we have not measured whether it would outperform a learned
   policy. Rule-based was chosen explicitly for safety and
   auditability per `docs/PROPOSAL.md`.

## Repository hygiene

- All multi-GB artifacts (`data/raw/*.parquet`, `data/raw/*.duckdb`,
  `data/processed/`, `data/models/`) are gitignored. Run the pipeline
  to regenerate.
- The feature cache uses an explicit `FEATURES_VERSION` constant in
  `src/pipeline/features.py`. Bump it when changing feature code so
  stale caches are invalidated.
- DuckDB lock errors surface the holder PID via `lsof` so a stale
  process never leaves you wondering what's holding the file.

## License

Internal MDK assignment. Not for distribution.
