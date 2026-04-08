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
4. **Trains** two models:
   - **XGBoost** classifier for supervised failure prediction
     (`is_pre_failure` ↔ degradation phase). Lead time ≈ 7 days
     average for detected failures. This is the working detector.
   - **LSTM-Autoencoder** on healthy-only data as an unsupervised
     safety-net. Currently non-functional on this dataset (see
     Known limitations) — the training/inference plumbing and
     determinism guardrails are all in place, but the model itself
     doesn't separate failure from healthy sequences.
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

## Reading order (for reviewers)

If you have 5 minutes, read these in this order:

1. **`docs/TECHNICAL_REPORT.md`** — the 2-4 page report covering
   problem framing, approach, KPI design, results, and safety.
2. **`docs/ARCHITECTURE.md`** — three mermaid diagrams showing the
   dataflow, the two-model rationale, and the safety control loop.
3. **`notebooks/01_eda.executed.ipynb`** — pre-rendered EDA with
   plots. Shows the synthetic data is physics-plausible and the
   label fix is verified inline.
4. **`notebooks/02_results.executed.ipynb`** — pre-rendered results
   notebook with ROC curves, per-failure-type tables, top features.
5. **`docs/REMAINING_FIXES.md`** — 14-item followup queue prioritized
   P0-P3, for understanding what was intentionally deferred.

If you have an hour, also:

6. Run `uv sync && uv run python -m src.run_pipeline` (~50 min,
   reproduces all the numbers in the report) and then open the
   notebooks live.
7. Skim `src/models/xgboost_classifier.py`,
   `src/models/lstm_autoencoder.py`, and
   `src/pipeline/features.py:split_temporal_tvt` for the
   correctness-critical code paths.
8. Skim `src/optimizer/safety.py:SafetyGuard` for the hardware
   safety layer.

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
| Anomaly detection | LSTM-Autoencoder (CPU inference, MPS training) | No failure labels needed; intended as safety-net detector (currently non-functional — see Known limitations) |
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
| Best val_loss | 0.3654 (epoch 2, early-stopped epoch 6) |
| Threshold (95th pct of healthy val errors) | 1.0002 |
| Healthy false-alarm rate | 3.1% |
| Failure detection rate | 0.2% |
| Mean reconstruction error (healthy) | 0.3377 |
| Mean reconstruction error (failure) | 0.1837 |
| **Separation ratio** | **0.54×** (inverted — does not work) |

**The LSTM-Autoencoder, as trained here, does not work on this
dataset.** Failure sequences reconstruct *better* than healthy
ones (sep < 1), because the healthy manifold is extremely broad
(3.2 M training rows across 26 miners of different hardware models)
while many failure sequences contain "flat/constant" patterns
(shutdown, stuck values) that a small autoencoder reconstructs
trivially well. XGBoost carries the predictive-maintenance
capability on its own.

Previous iterations of this README reported sep=2.66× and a 33%
detection rate for the LSTM. Those numbers were a reporting
artifact from an Apple Silicon MPS kernel bug where `batch_size=128`
with this specific LSTM architecture returns numerically wrong
outputs silently. The training script monitored val/test separation
via that buggy path and saw phantom metrics, while reload via
any other batch size or on CPU gave the real (poor) numbers.
The fix in `src/models/lstm_autoencoder.py:compute_reconstruction_error`
forces CPU inference regardless of training device, eliminating
the bug. See commit history and `scripts/consistency_check.py`
check 9 for the determinism test that keeps this from regressing.

Concrete followups to make the LSTM actually work would include
per-miner-model scaling (separate feature statistics by hardware
type), a larger latent dimension, or replacing the reconstruction
objective with a contrastive one. None of these are in scope for
this 3-week prototype; the LSTM remains in the codebase as the
unsupervised-branch hook for future work, but it is explicitly
**not** a working component of this submission.

## Per-failure-type detection coverage

Aggregate metrics like AUC and F1 hide the most useful question:
**which specific failure modes can the system actually catch, and
which can it not?** This is what mentors will ask. The answer is
measured directly against the test set, not extrapolated.

The test slice contains **6 distinct failure events across 5 failure
types** (the held-out 25% of the simulation timeline produces this
many failures naturally; for more failures we'd need a larger
synthetic run).

| Miner | Failure type | XGBoost | Headline |
|---|---|---|---|
| MNR-016 | connector_corrosion | ✅ caught | **16.5 days lead time** |
| MNR-008 | connector_corrosion | ✅ caught | **5.9 days lead time** |
| MNR-018 | thermal_runaway | ✅ caught | 11.9 hours lead time |
| MNR-020 | psu_degradation | ❌ missed | — |
| MNR-022 | coolant_restriction | ❌ missed | — |
| MNR-029 | sudden_chip_failure | ❌ missed | only 2 rows in test — unmeasurable |

**Coverage: 3 of 6 failures caught by XGBoost.** The best catches
are the two slow-developing `connector_corrosion` events with 5.9
and 16.5 day lead times — enough runway to schedule maintenance
during planned downtime rather than reacting to thermal alarms.
The `thermal_runaway` case is caught with only 11.9 hours of lead
time, which is still operationally useful compared to reactive
thermal shutdown.

The three misses are `psu_degradation`, `coolant_restriction`, and
`sudden_chip_failure`. An earlier iteration of this README claimed
the LSTM-AE caught the first two at 100% — those were phantom
metrics from the MPS `batch_size=128` bug described below. The
LSTM-AE does not actually catch them; its real separation ratio
is inverted (sep=0.54×). `sudden_chip_failure` only leaves 2 rows
in the test window, below anything a statistical model could
learn from; `SafetyGuard.enforce_thermal_shutdown()` in
`src/optimizer/safety.py` is the right mechanism for that class,
not the predictive models.

### Row-level recall by failure type (XGBoost)

For reference, the supervised model's per-row recall on test
positives is conservative because the threshold is calibrated for
F1 with a precision floor, not coverage. Event-level detection is
the metric operators care about, not row-level recall.

| Failure type | Pre-failure rows in test | XGBoost catches | Row-level recall |
|---|---|---|---|
| connector_corrosion | 33,777 | 10,837 | 32.1% |
| psu_degradation | 32,412 | 0 | 0.0% |
| coolant_restriction | 20,968 | 0 | 0.0% |
| thermal_runaway | 746 | 274 | 36.7% |
| sudden_chip_failure | 2 | 0 | n/a |

The two 0% rows are failure modes where the pre-failure telemetry
signature is too close to healthy operation for the supervised
model to learn a decision boundary under the current feature set
and class imbalance. The LSTM-AE was supposed to be the fallback
detector for these cases; see the LSTM section above for why it
currently does not fill that role.

## Known limitations

These are documented honestly so a reviewer doesn't have to discover
them mid-demo.

1. **LSTM-Autoencoder does not detect anomalies on this dataset.**
   The trained model has an inverted separation ratio (sep=0.54×,
   meaning failure sequences reconstruct *better* than healthy)
   because the healthy manifold across 26 miners of mixed hardware
   is too broad for a 64-hidden AE while failure sequences often
   contain easy-to-reconstruct flat patterns. Previous runs reported
   sep=2.66× — that was a phantom metric from an Apple Silicon MPS
   `batch_size=128` kernel bug that silently produced wrong outputs.
   The bug is fixed (`compute_reconstruction_error` now forces CPU
   inference), the model weights are correctly saved, and the real
   numbers are what you see in the LSTM table above. Making the LSTM
   actually work would need per-miner-model scalers, a larger latent,
   or a contrastive objective — out of scope for this prototype.
2. **Live CLI inference is approximate.** The
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
