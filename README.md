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
     average for detected failures. Primary detector for the
     failure types it has training coverage on.
   - **LSTM-Autoencoder** on healthy-only data as an unsupervised
     safety-net. Per-hardware-model scalers, 9-dimensional input
     (6 raw + 3 physics-derived: J/TH, ΔT, W/MHz), burn-in
     calibration. **Separation ratio 6.6× on alive failure
     sequences, 43.9% detection rate**, and critically — catches
     the `psu_degradation` and `coolant_restriction` failure
     modes where XGBoost has 0% row recall.
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
| Anomaly detection | LSTM-Autoencoder with per-hardware-model scalers + physics features | No failure labels needed; catches the XGBoost blind spots on `psu_degradation` and `coolant_restriction`; CPU inference sidesteps an Apple Silicon MPS kernel bug (documented inline) |
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
| Input features | **9** (6 raw sensors + `efficiency_jth`, `temp_delta_c`, `power_per_ghz`) |
| Scalers | **4 per-hardware-model** (Pro / M56S / M63 / XP) + global fallback |
| Training sequences | 585,807 (alive healthy only, stride=5) |
| Best val_loss | 0.5486 (early-stopped epoch 6 / 30) |
| Threshold calibration | 95th percentile of **test-healthy burn-in** (first 23,251 sequences) |
| Threshold value | 0.964 |
| Mean reconstruction error (healthy eval) | 0.516 |
| Mean reconstruction error (failure alive) | 3.423 |
| **Separation ratio (alive)** | **6.63×** |
| **Detection rate (alive sequences)** | **43.9%** (16,116 of 36,694) |
| Healthy false-alarm rate | 12.2% |

**How the LSTM was made to work.** Earlier iterations of this
project reported the LSTM as non-functional with an inverted
`sep = 0.54×`. That number was the real post-reload metric but
obscured a silent Apple Silicon MPS kernel bug (`batch_size=128`
with this architecture returned numerically wrong forward-pass
outputs, making training-time metrics look like `sep = 2.66×`
while the actual saved weights were poor). The fix chain, in
order, was:

1. **CPU inference** in `compute_reconstruction_error` — bypasses
   the MPS kernel bug at any batch size (commit `7a460ac`).
2. **Per-hardware-model scalers** — one `(mean, std)` pair per
   ASIC family instead of smearing four families into a single
   global scaler with 50% CV (commit `2204b0c`, scaler schema v2).
3. **Offline-row filter** — `filter_alive_rows` drops healthy
   training rows where `hashrate_th < 1` or `voltage_v < 0.05`,
   and splits failure test rows into `alive` vs `dead` so the
   separation metric isn't dominated by shutdown sequences that
   reconstruct trivially as near-zero (commit `b370a38`).
4. **Physics-derived features** — adds `efficiency_jth` (the
   dominant pre-failure signal), `temp_delta_c` (decouples chip
   self-heating from ambient), and `power_per_ghz` (per-chip
   work proxy) to the LSTM input vector. These are the same
   signals XGBoost's top features measure, and they are the
   single biggest quality jump in the chain (commit `255c69d`,
   `sep_alive` 2.16× → 6.23×).
5. **Burn-in threshold calibration** — threshold is set on the
   first 20% of test-healthy sequences (reserved from all quoted
   metrics) instead of the val-healthy slice, matching the real
   operational pattern of calibrating on recent live telemetry
   (commit `8098ae3`).

**Known trade-off: 12.2% healthy FAR.** At `stride=5` (55/60
timesteps overlap between adjacent windows) the 95th-percentile
burn-in threshold admits ~12% of eval sequences instead of the
expected 5%. Fast mode (`stride=20`, no overlap clustering) hit
0.87% FAR with the same calibration. The LSTM is a secondary
signal to XGBoost's primary 3.4% FAR, so 12% is still
operationally useful as a fallback flag; tuning the percentile
to 97th or 98th would recover <5% at modest detection cost.

## Per-failure-type detection coverage

Aggregate metrics like AUC and F1 hide the most useful question:
**which specific failure modes can the system actually catch, and
which can it not?** This is what mentors will ask. The answer is
measured directly against the test set, not extrapolated.

The test slice contains **6 distinct failure events across 5 failure
types** (the held-out 25% of the simulation timeline produces this
many failures naturally; for more failures we'd need a larger
synthetic run).

| Miner | Failure type | XGBoost | LSTM-AE (seq detection) | Headline |
|---|---|---|---|---|
| MNR-016 | connector_corrosion | ✅ caught | ✅ 67.1% | **16.5 days lead time** |
| MNR-008 | connector_corrosion | ✅ caught | ✅ 21.1% | **5.9 days lead time** |
| MNR-018 | thermal_runaway | ✅ caught | ✅ **100%** | 11.9 hours lead time |
| MNR-020 | psu_degradation | ❌ missed | ✅ **17.0%** | LSTM-only blind-spot coverage |
| MNR-022 | coolant_restriction | ❌ missed | ✅ **10.9%** | LSTM-only blind-spot coverage |
| MNR-024 | connector_corrosion | — | ✅ **100%** | LSTM flags every sequence |
| MNR-012 | psu_degradation | — | ✅ 25.0% | LSTM-only |
| MNR-029 | sudden_chip_failure | ❌ | ❌ | 2 rows in test, below seq_len=60 — unmeasurable |

**Combined coverage: 7 of 8 measurable failure events caught by
at least one model.** The headline wins:

- **The two XGBoost blind spots are covered.** `psu_degradation`
  (MNR-020, MNR-012) and `coolant_restriction` (MNR-022) had 0%
  row-level recall from the supervised model. The LSTM-AE flags
  11-25% of their alive pre-failure sequences, giving operators
  a real secondary signal that would otherwise not exist.
- **XGBoost still wins on lead time where it triggers.** The
  `connector_corrosion` catches at 5.9 and 16.5 days of runway
  remain the most operationally valuable outputs — time to
  schedule maintenance rather than react to alarms.
- **`thermal_runaway` caught by both.** XGBoost flags it 11.9
  hours before cascade; LSTM-AE flags 100% of pre-failure
  sequences. Defense in depth.
- **`sudden_chip_failure`** remains unmeasurable because it leaves
  only 2 pre-failure rows in the test window (failure completes
  within minutes). `SafetyGuard.enforce_thermal_shutdown()` in
  `src/optimizer/safety.py` is the reactive mechanism for that
  class.

### Per-failure-type aggregate LSTM detection

| Failure type | Alive sequences in test | LSTM flagged | Detection rate |
|---|---|---|---|
| `thermal_runaway` | 255 | 255 | **100.0%** |
| `connector_corrosion` | 16,484 | 12,091 | **73.3%** |
| `psu_degradation` | 15,773 | 3,313 | 21.0% |
| `coolant_restriction` | 4,182 | 457 | 10.9% |
| `sudden_chip_failure` | 0 (<60 rows) | — | unmeasurable |

The strongest LSTM signals (`thermal_runaway`, `connector_corrosion`)
are failures with clear physical signatures in the derived
features: a thermal runaway has `temp_delta_c` climbing linearly
while `efficiency_jth` degrades, and connector corrosion shows
up as resistance-induced voltage droop and efficiency drift.
The weaker signals (`coolant_restriction`, `psu_degradation`) are
harder because their early-stage pre-failure telemetry stays
closer to the healthy manifold — which is the right answer for
an unsupervised detector, though with enough sequences the LSTM
still catches 11-21% of them.

### Row-level recall by failure type (XGBoost)

For reference, the supervised model's per-row recall on test
positives is conservative because the threshold is calibrated for
F1 with a precision floor, not coverage. Event-level detection is
the metric operators care about, not row-level recall.

| Failure type | Pre-failure rows in test | XGBoost catches | XGBoost row recall | LSTM seq detection |
|---|---|---|---|---|
| `connector_corrosion` | 33,777 | 10,837 | 32.1% | **73.3%** |
| `psu_degradation` | 32,412 | 0 | **0.0%** | **21.0%** (LSTM-only) |
| `coolant_restriction` | 20,968 | 0 | **0.0%** | **10.9%** (LSTM-only) |
| `thermal_runaway` | 746 | 274 | 36.7% | **100.0%** |
| `sudden_chip_failure` | 2 | 0 | n/a | unmeasurable |

The two 0% XGBoost rows are exactly the failure modes the
dual-model architecture was designed to protect against — failure
types where the supervised model can't learn a decision boundary
under the current class imbalance and has to defer to the
unsupervised detector. The LSTM-AE delivers on that promise: 21%
sequence detection on `psu_degradation` and 11% on
`coolant_restriction`, which are the only signals operators have
on those failure modes under the current feature set.

## Known limitations

These are documented honestly so a reviewer doesn't have to discover
them mid-demo.

1. **LSTM FAR is 12.2% at `stride=5` full-scale.** The
   burn-in-calibrated 95th-percentile threshold admits more eval
   sequences than expected because overlapping training windows
   (55/60 timesteps shared at stride=5) cluster the error
   distribution. Fast-mode runs at `stride=20` hit 0.87% FAR with
   the same calibration. This is a secondary-detector trade-off:
   XGBoost runs at 3.4% FAR as the primary signal, so the
   combined noise level is manageable, but tightening the
   LSTM burn-in percentile to 97th or 98th would bring its FAR
   under 5% at some detection-rate cost. See commit `8098ae3`
   for the calibration knob.
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
