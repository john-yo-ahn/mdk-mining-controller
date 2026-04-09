# Technical Proposal: AI-Driven Mining Optimization & Predictive Maintenance

## Project: MDK AI Controller
**Author**: John Ahn
**Platform**: Tether Mining Development Kit (MDK) / Mining OS (MOS)
**Duration**: 3 Weeks
**Date**: April 2026

---

## Status update (April 2026) — what actually shipped

This proposal is the original pitch. Below is a short ledger of what
was built, what was exceeded, where we deviated, and what's in the
followup queue. For the canonical results document see
`docs/TECHNICAL_REPORT.md`; for the implementation-level followup list
see `docs/REMAINING_FIXES.md`.

**Delivered as proposed:**
- XGBoost predictive maintenance classifier (binary `is_pre_failure`)
- LSTM-Autoencoder unsupervised anomaly detector
- Rule-based dynamic efficiency optimizer with SafetyGuard chokepoint
- True Efficiency (TE) KPI (base / adjusted / health variants)
- Feature engineering pipeline (152 features, multi-timescale)
- Synthetic physics-first telemetry generator
- CLI dashboard with live fleet simulator
- Adaptive temporal train/val/test split
- Honest held-out evaluation with val-tuned thresholds

**Exceeded proposed scope:**
- Dataset size: 30 miners × 120 days (~5.2 M rows) vs the
  originally-proposed 50 × 30. The longer time axis lets the 7-day
  rolling features fully populate.
- 152 features across 6 rolling-window timescales (2 m / 15 m / 1 h
  / 6 h / 1 d / 7 d) rather than the 3 windows in the proposal.
- 7 distinct failure scenarios vs the 5 listed in the proposal
  (added `coolant_restriction`, `connector_corrosion`,
  `firmware_oscillation`).
- Model metadata sidecars alongside every saved model (git commit,
  training config, val metrics) for auditability.
- Dual-file DuckDB storage with lock recovery via `lsof`.
- Per-failure-type detection coverage analysis (see TECHNICAL_REPORT
  §5.3) rather than the aggregate precision/recall reporting
  originally planned.

**Deviated from the proposal:**
- **Pre-failure label definition** was redefined mid-project to use
  `degradation_phase ∈ {incubation, acceleration}` instead of a
  fixed 24-hour horizon. The original 24 h window stamped labels on
  scheduled timestamps that drifted up to 18 days from the actual
  failure progression, leaving >50% of training positives on
  actually-healthy telemetry. Details in TECHNICAL_REPORT §2 and the
  commit `dba8d25` diff of `src/synthetic/generator.py`. Consequence:
  the model predicts "is this miner currently degrading" rather
  than "will this miner fail in exactly 24 h" — operationally more
  useful but not the framing the proposal used.
- **Optimizer is rule-based, not RL.** The proposal scoped RL as a
  stretch goal; we chose rule-based for hardware-safety and
  auditability reasons (see TECHNICAL_REPORT §2.2). RL remains a
  documented future-work path.
- **Feature engineering is cached** to `data/processed/features.v{N}.parquet`
  with an explicit `FEATURES_VERSION` constant. Wasn't in the proposal
  but was necessary to make iteration tractable — a cold feature
  build takes ~25 minutes, caches hit is instant.

**In the followup queue** (not shipped, see `docs/REMAINING_FIXES.md`):
- F1: live CLI inference feature-set mismatch (streaming path
  computes ~80 of 152 features and silently zeros the rest).
  Documented and warned about at `AIBridge.load_models()` startup;
  planned for fix before demo.
- F12: multi-class XGBoost targeting specific failure types (would
  give the supervised side direct coverage of `psu_degradation` and
  `coolant_restriction` instead of relying solely on the LSTM-AE's
  21% and 11% sequence-detection rates).
- F14: real MDK API client — gated on a data sharing channel with
  Tether mining ops.

**Headline results** (30 × 120 held-out test, see TECHNICAL_REPORT §5):
- XGBoost AUC = **0.851**, avg lead time **11.3 days** on its
  catches. **4 of 6 test failure events** caught (v3 cache with
  TE-derived features; `te_health` rolling variants rank 8 and 9
  in feature importance).
- LSTM-AE **separation ratio 6.63×** on alive failure sequences,
  **43.9% detection rate**, per-hardware-model scalers, 9-feature
  input (6 raw + physics-derived J/TH, ΔT, W/MHz), burn-in
  threshold calibration. Catches the two XGBoost blind spots:
  `psu_degradation` (21% vs XGBoost 0%) and `coolant_restriction`
  (11% vs XGBoost 0%).
- **Combined coverage: 7 of 8 measurable failure events caught by
  at least one model.**
- 6× lower false-alarm rate than a simple temperature + hashrate
  threshold baseline.

---

## 1. Problem Statement

Bitcoin mining profitability depends on two controllable variables: **hardware efficiency (J/TH)** and **energy cost**. In current operations, site operators manually adjust chip frequencies, voltages, and cooling parameters based on experience — a process that doesn't scale across thousands of ASICs and can't react fast enough to changing conditions.

Two critical problems demand AI-driven solutions:

1. **Hardware failures cause costly downtime**: Chip and machine breakdowns lead to expensive repairs and lost hashing revenue. Operators lack early warning systems that can flag degrading machines before critical failure.

2. **Suboptimal efficiency at scale**: Tuning chip frequency and power is done arbitrarily. Each ASIC has a unique optimal operating point that shifts with ambient temperature, coolant conditions, and chip age — finding and maintaining this point for every chip in a fleet is beyond human capacity.

Since mining revenue is continuous (pool mining pays per hash), every percentage point of efficiency improvement or hour of prevented downtime directly translates to revenue.

---

## 2. Proposed Approach: Dual-Track AI Controller

We propose a **combined approach** tackling both problems with a unified data pipeline and two complementary AI modules:

### Track A: Predictive Maintenance (Primary Focus)
An **XGBoost-based classifier** trained on synthetic telemetry to detect pre-failure patterns, supplemented by an **LSTM-Autoencoder** for unsupervised anomaly detection.

**Why this combination:**
- XGBoost achieves ~98% accuracy on tabular sensor data and provides interpretable feature importance (operators can understand WHY a machine is flagged)
- LSTM-Autoencoder captures temporal degradation patterns without requiring labeled failure data — critical since real failure datasets are unavailable
- Together they cover both known failure modes (supervised) and novel anomalies (unsupervised)

**Signals monitored:**
- Chip temperature trends and gradients
- J/TH degradation over time (efficiency slope)
- Hashrate deviation from nominal (>10% drop = flag)
- Power consumption instability (voltage ripple)
- Cross-signal ratios (temperature-per-watt anomalies)

### Track B: Dynamic Efficiency Optimization (Stretch Goal)
A **rule-based optimizer** (upgradeable to RL) that adjusts operating parameters based on real-time conditions:

- When ambient temperature rises → reduce chip frequency to maintain thermal headroom
- When energy price drops → increase frequency to maximize hashrate during cheap energy windows
- When efficiency degrades beyond threshold → flag for maintenance and reduce to safe operating point

**Why rule-based first:**
- The meeting emphasized that the project is in design-thinking mode, not production deployment
- Rule-based controllers are transparent, auditable, and safe — critical for hardware control
- The rule set can be formalized as a SAC/PPO reward function for future RL migration

---

## 3. Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TELEMETRY SOURCES                            │
│  ASIC Miners (via MDK Workers)  │  Containers  │  Power Meters     │
└──────────────────┬──────────────┴──────┬───────┴────────┬──────────┘
                   │                     │                │
                   ▼                     ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION & PREPROCESSING                        │
│  • Parse synthetic CSV/JSON telemetry                               │
│  • Align timestamps across device types                             │
│  • Handle missing values (forward-fill for sensors)                 │
│  • Normalize per-device (Z-score within device history)             │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                               │
│  • Rolling statistics (mean, std, min, max) at 2min/15min/1hr      │
│  • Rate of change (dT/dt, dHashrate/dt, dPower/dt)                 │
│  • Cross-signal ratios (J/TH, temp-per-watt)                       │
│  • True Efficiency KPI computation                                  │
│  • Degradation slope (linear regression on rolling J/TH window)    │
└──────────┬──────────────────────────────────┬──────────────────────┘
           │                                  │
           ▼                                  ▼
┌────────────────────────┐     ┌──────────────────────────────────┐
│  PREDICTIVE MAINTENANCE│     │  EFFICIENCY OPTIMIZER             │
│  • XGBoost classifier  │     │  • Rule-based controller          │
│  • LSTM-Autoencoder    │     │  • Condition-action policies      │
│  • Anomaly scoring     │     │  • Energy-price-aware scheduling  │
└──────────┬─────────────┘     └──────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION & OUTPUT LAYER                           │
│  • Maintenance alerts (Critical/High/Medium severity)               │
│  • Recommended frequency/voltage adjustments                        │
│  • Fleet health dashboard and KPI reporting                         │
│  • Safety constraints enforcement (thermal limits, max power)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. True Efficiency (TE) KPI Design

Standard J/TH only captures chip-level energy-to-hashrate conversion. We propose a **True Efficiency** metric that incorporates the full operational context:

```
TE = Hashrate_actual / (P_chip + P_cooling * α + P_infrastructure * β)
```

Where:
- `P_chip` = Direct ASIC power consumption (W)
- `P_cooling` = Cooling system power (fans, pumps, chillers) allocated per device
- `α` = Cooling overhead factor (proportion attributable to this device)
- `P_infrastructure` = Network, control, lighting overhead allocated per device
- `β` = Infrastructure allocation factor

**Environmental adjustment:**
```
TE_adjusted = TE * (1 - δ_temp * max(0, T_ambient - T_baseline))
```
Where `δ_temp` captures the efficiency loss per degree above baseline (empirically ~0.5-1% per C above 25C for air-cooled systems).

**Degradation-aware variant:**
```
TE_health = TE_adjusted * (Hashrate_actual / Hashrate_nameplate)
```
This captures both environmental conditions and machine aging in a single number.

---

## 5. Expected Operational Benefits

| Benefit | Mechanism | Estimated Impact |
|---|---|---|
| Reduced unplanned downtime | Early failure detection (24-72hr warning) | 10-20% reduction in repair costs |
| Improved fleet efficiency | Dynamic tuning per ambient conditions | 3-8% J/TH improvement |
| Extended hardware lifespan | Avoid thermal cycling stress | 10-15% longer economic life |
| Lower energy costs | Energy-price-aware scheduling | 5-10% energy cost reduction |
| Faster fault diagnosis | Automated root cause via feature importance | Reduce mean-time-to-repair |

---

## 6. Security & Safety Considerations

### Safety Constraints (Non-Negotiable)
- **Thermal hard limit**: Never allow chip temperature to exceed manufacturer-rated max (typically 95C). Controller must have a hardware-independent thermal cutoff.
- **Rate limiting**: Frequency/voltage changes must be rate-limited (max 1 change per 5 minutes) to prevent thermal cycling damage.
- **Fallback mode**: If AI controller fails or produces anomalous output, system defaults to conservative operating point (Normal mode, rated frequency).
- **Human override**: All automated decisions must be overridable by operators. AI recommends, humans confirm for critical actions.

### Security Risks
- **Adversarial sensor data**: Corrupted telemetry could cause the controller to make damaging decisions. Implement sensor plausibility checks (physical bounds validation).
- **Command injection**: The AI controller interfaces with hardware control APIs. All command outputs must be validated against allowed ranges before execution.
- **Model poisoning**: If the model is retrained on operational data, an attacker who compromises sensor data could poison the model. Use data integrity verification.
- **Network isolation**: The AI controller should run on an isolated network segment, not exposed to the internet.

---

## 7. Technology Stack

| Component | Technology | Justification |
|---|---|---|
| Data processing | Python + Pandas/Polars | Standard for data engineering, fast tabular ops |
| Feature engineering | NumPy, SciPy | Rolling statistics, signal processing |
| Predictive model | XGBoost, scikit-learn | Best accuracy on tabular data, interpretable |
| Anomaly detection | PyTorch (LSTM-Autoencoder) | Temporal pattern capture, unsupervised |
| Visualization | Matplotlib, Plotly | Interactive dashboards, time-series plots |
| Synthetic data | NumPy + physics models | Thermodynamically plausible data generation |
| Notebooks | Jupyter | Exploratory analysis, presentation |

---

## 8. Timeline

| Week | Focus | Deliverable |
|---|---|---|
| **Week 1** | Data pipeline + EDA + KPI design | Working ingestion pipeline, TE KPI formula, correlation analysis |
| **Week 2** | Predictive maintenance model | XGBoost classifier, LSTM-Autoencoder, anomaly scoring |
| **Week 3** | Optimizer + report + polish | Rule-based optimizer, technical report, architecture diagram |
