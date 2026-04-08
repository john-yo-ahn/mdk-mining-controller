# Technical Report — MDK AI Mining Controller

**Author:** John Ahn
**Date:** April 2026
**Assignment:** Tether MDK AI Mining Controller (3-week prototype)
**Version:** 1.0

---

## 1. Problem Statement

Bitcoin mining is a cost-management game. Hashprice is determined by the
market; the only levers an operator controls are **chip efficiency** and
**operational cost**. Giorgio (Head of MOS at Tether) identified the two
highest-value pain points in production fleets: (1) unplanned downtime
from chip and machine breakage, and (2) manual operator tuning of
frequency, voltage, and cooling that doesn't scale to thousands of
ASICs per site.

This prototype delivers an AI-driven controller targeting both problems
on Tether's Mining Development Kit (MDK) platform, built on synthetic
physics-plausible telemetry for a 30-miner fleet across 120 days.

### Mining economics framing

The controllable side of the profit equation is:

> **Hash Cost = Electricity Cost × Hashing Efficiency (J/TH)**
> **Gross Profit = (Hash Price − Hash Cost) × Miner Hash Rate**

Every efficiency point gained and every hour of unplanned downtime
avoided translates directly to revenue. The controller addresses this
via two complementary modules:

- **Predictive maintenance** — detects pre-failure degradation days
  before cascade, enabling scheduled replacements during planned
  downtime rather than reactive responses to thermal alarms.
- **Dynamic efficiency optimization** — a rule-based controller that
  adjusts frequency in response to thermal headroom, energy price,
  and AI-predicted degradation, all gated through a `SafetyGuard`.

---

## 2. Approach

### 2.1 Why two models, not one

The system architecture runs **XGBoost** (supervised failure
classifier) and **LSTM-Autoencoder** (unsupervised anomaly detector)
in parallel. The intent was concrete: XGBoost learns patterns
specific to the failure types it has seen during training, so an
under-represented failure type becomes a blind spot at inference
time. An autoencoder trained only on healthy telemetry is supposed
to fill that gap by flagging anything that deviates from the healthy
distribution, regardless of whether the deviation matches a known
failure class.

In practice, **only XGBoost is a working detector in this
submission.** The LSTM-AE plumbing (training, threshold calibration,
scaler persistence, live-inference hook, metadata sidecar, and
determinism guardrails) is all in place, but the trained autoencoder
has an inverted separation ratio on the held-out test set
(sep = 0.54×, healthy mean error 0.338 vs failure mean error 0.184
— failure sequences reconstruct *better* than healthy) because the
healthy manifold across 26 miners of mixed hardware is too broad
for a 64-hidden AE while failure sequences frequently contain
trivially-reconstructable flat/constant patterns (shutdown, stuck
values). Earlier drafts of this report quoted sep = 2.66× for the
LSTM — that was a reporting artifact from an Apple Silicon MPS
kernel bug where `batch_size=128` with this LSTM silently returned
numerically wrong outputs. The fix forces CPU inference for
reconstruction error; the weights themselves are fine; the model
itself simply does not work on this dataset.

The architectural argument for two models remains valid as forward
guidance — making the autoencoder actually work would require
per-miner-model scaling, a larger latent, or a contrastive
objective — and the code paths needed for a second detector are
already integrated. For this submission, XGBoost is the detector
of record.

### 2.2 Why rule-based optimizer (not RL)

The assignment explicitly asks for "design thinking and solution
frameworks over fully tested models". For a system that writes
directly to hardware registers, the design decision tree is dominated
by safety and auditability:

1. **Rule-based** actions are traceable ("this miner was throttled at
   15:42 because temperature = 87°C > 85°C warning threshold").
2. **RL** actions are opaque policies learned from simulated reward
   signals that may not match real operator incentives.
3. Hardware control errors are expensive and uncorrectable — a
   misconfigured RL policy that ran for an hour in production could
   damage thousands of chips.

The optimizer is implemented as condition-action rules in
`src/optimizer/rules.py` covering thermal management, energy price
response, and AI-driven degradation flagging. Every proposed action
flows through `src/optimizer/safety.py:SafetyGuard` which enforces
thermal shutdown overrides, rate limiting, and value bounds. An RL
migration path is documented in `docs/PROPOSAL.md` as stretch work.

---

## 3. Pipeline Architecture

```
Synthetic       Feature         Train/Val/Test       Supervised + Unsupervised
Telemetry  →    Engineering →   Temporal Split   →   Models (XGBoost + LSTM-AE)
(5.2 M rows)    (152 features)  (adaptive)           ↓
                                                     Threshold tuning on VAL only
                                                     ↓
                                                     Held-out TEST evaluation
                                                     (honest metrics)
                                                     ↓
                                                     Combined risk score
                                                     ↓
                                                     Rule-based optimizer
                                                     ↓
                                                     SafetyGuard (hard limits)
                                                     ↓
                                                     Control actions
```

Full component map is in `docs/ARCHITECTURE.md` with three mermaid
diagrams covering dataflow, two-model rationale, and the safety
control loop.

### 3.1 Notable engineering decisions

- **Feature cache with explicit version.** Feature engineering on the
  5.2 M-row dataset takes ~25 minutes; a `FEATURES_VERSION` constant
  in `src/pipeline/features.py` gates an automatic cache in
  `data/processed/features.v{N}.parquet`. Bumping the version forces
  a rebuild, preventing silent train/inference drift from stale
  caches.
- **Adaptive train/val/test split.** The synthetic generator produces
  failure events clustered in time. A naive fixed-fraction temporal
  split (e.g., 60 / 15 / 25 by date) will sometimes land the
  validation window in a temporal dead zone with zero positive
  examples, making F1-based threshold tuning degenerate. The
  `split_temporal_tvt` function places boundaries by cumulative
  positive count rather than by date, guaranteeing every split
  contains failure examples while preserving strict temporal
  ordering.
- **Separate DuckDB files for batch and live.** `data/raw/mdk.duckdb`
  holds training telemetry; `data/raw/mdk_live.duckdb` holds live
  simulator output. Keeping them separate avoids single-writer lock
  contention. Lock errors are surfaced via `lsof` so a stale writer
  never leaves the system wondering what's holding the file.
- **Honest threshold tuning.** The default threshold strategy is
  `f1_with_floor` — maximize F1 on validation scores, falling back
  to a precision floor (0.05) if F1-max would collapse to threshold
  0 under extreme class imbalance. This prevents the degenerate
  "flag everything" and "flag nothing" states.

---

## 4. KPI Design — True Efficiency

Simple J/TH is a poor operational metric because it ignores cooling
overhead, environmental variance, and hardware degradation. The
proposed **True Efficiency (TE)** KPI in `src/kpi/true_efficiency.py`
layers three progressively richer variants:

| KPI | Formula | Captures |
|---|---|---|
| `te_base` | `hashrate / (chip_power × (1 + α_cooling + β_infra))` | Cooling and infrastructure overhead |
| `te_adjusted` | `te_base × (1 − δ × max(0, ambient_temp − baseline))` | Environmental correction |
| `te_health` | `te_adjusted × hashrate_realization` | Degradation awareness |

**Validation:** On the 30-miner dataset, `te_health` separates healthy
from failing miners by **114.4%** (`mean_healthy / mean_failing − 1`),
versus **~14%** separation for naive J/TH. The full distribution
comparison is rendered in `notebooks/01_eda.executed.ipynb` Section 5.

`te_health` is also one of the top 10 features the XGBoost model uses
(rank 10 by gain), which closes the loop — the KPI is both a reporting
metric for operators and a learned input signal for the supervised
model.

---

## 5. Results

All metrics on the held-out test slice (25% of timeline by cumulative
positives) which was not used for model fit, threshold tuning, or any
validation decision.

### 5.1 XGBoost classifier

| Metric | Value |
|---|---|
| AUC-ROC | **0.801** |
| F1 | 0.163 |
| Precision | 0.230 |
| Recall | 0.126 |
| `scale_pos_weight` (sqrt-capped) | 4.5 (from raw 20.2) |
| Decision threshold | 0.119 (tuned on validation only) |

### 5.2 LSTM-Autoencoder (non-functional on this dataset)

| Metric | Value |
|---|---|
| Best validation loss | 0.3654 (epoch 2, early-stopped at 6) |
| Anomaly threshold (95th percentile of healthy val errors) | 1.0002 |
| Healthy false-alarm rate | 3.1% |
| Failure sequence detection rate | 0.2% |
| Mean reconstruction error (healthy) | 0.3377 |
| Mean reconstruction error (failure) | 0.1837 |
| Separation ratio (mean failure / mean healthy error) | **0.54× (inverted)** |

The autoencoder does not separate failure sequences from healthy
ones on this dataset. See §2.1 for the reason (broad healthy
manifold across mixed hardware + flat failure patterns), and the
README's "LSTM-Autoencoder" section for a short postmortem on
the MPS `batch_size=128` reporting bug that hid this for several
iterations of the project.

### 5.3 Per-failure-type detection coverage

The headline operator-facing question is: **which specific failure
modes can the system catch?** Measured directly against the 6
distinct failure events in the held-out test set:

| Miner | Failure type | XGBoost | Best lead time |
|---|---|---|---|
| MNR-016 | connector_corrosion | ✅ caught | **16.5 days** |
| MNR-008 | connector_corrosion | ✅ caught | **5.9 days** |
| MNR-018 | thermal_runaway | ✅ caught | 11.9 hours |
| MNR-020 | psu_degradation | ❌ missed | — |
| MNR-022 | coolant_restriction | ❌ missed | — |
| MNR-029 | sudden_chip_failure | ❌ | 2 rows — unmeasurable |

**Coverage: 3 of 6 test failures caught by XGBoost, with an
average lead time of 7.6 days on its three catches.** The
strongest operational result is the 16.5-day lead time on the
slow-developing `connector_corrosion` case on MNR-016 —
comfortably within the window for a planned maintenance cycle.

The LSTM-AE was intended to catch the `psu_degradation` and
`coolant_restriction` cases that XGBoost misses (earlier drafts
of this report claimed it did) but the real numbers show it does
not (§5.2, §2.1). The `sudden_chip_failure` miss is unmeasurable
rather than unmodelable: the failure leaves only 2 pre-failure
rows in the test set because it completes within minutes.
Predictive detection is fundamentally unsuitable for that failure
class; `SafetyGuard.enforce_thermal_shutdown()` handles it
reactively.

### 5.4 Row-level recall by failure type (XGBoost)

| Failure type | Pre-failure rows | Caught | Row recall |
|---|---|---|---|
| `connector_corrosion` | 33,777 | 10,837 | 32.1% |
| `psu_degradation` | 32,412 | 0 | 0.0% |
| `coolant_restriction` | 20,968 | 0 | 0.0% |
| `thermal_runaway` | 746 | 274 | 36.7% |

The XGBoost blind spots on `psu_degradation` and `coolant_restriction`
are the direct consequence of failure-type imbalance in training.
They were the principal motivation for the dual-model architecture
— an unsupervised detector was supposed to fill this gap — and they
remain unfilled in this submission. Closing these gaps would be the
single highest-value followup: either by rebalancing the training
set with more examples of these failure modes, by adding
hand-designed features targeted at their signatures (PSU ripple,
inlet/outlet delta-T), or by making the autoencoder actually work
(§2.1).

### 5.5 Head-to-head against a simple threshold baseline

The single most honest question a reviewer can ask is: **"is the AI
actually better than a three-line rule?"** We compared XGBoost
against a hand-written baseline:

```python
threshold_flag = (temperature_c > 85) OR (hashrate_th < 80% of nameplate)
```

Same held-out test set, same 6 failures, same detection_timeline
function. Per-failure head-to-head:

| Miner | Failure | AI detected? | AI lead | Threshold detected? | Threshold lead |
|---|---|---|---|---|---|
| MNR-008 | connector_corrosion | ✅ | 5.9 d | ✅ | 7.0 d |
| MNR-016 | connector_corrosion | ✅ | **16.5 d** | ✅ | 3.4 d |
| MNR-018 | thermal_runaway | ✅ | 11.9 h | ✅ | 12.4 h |
| MNR-020 | psu_degradation | ❌ | — | ✅ | 21.5 d |
| MNR-022 | coolant_restriction | ❌ | — | ✅ | 8.9 d |
| MNR-029 | sudden_chip_failure | ❌ | — | ❌ | — |

**The threshold rule catches more distinct failures (5 of 6 vs 3 of
6).** But the comparison that actually matters to an operator is
**signal-to-noise ratio**:

| Metric | XGBoost | Threshold rule |
|---|---|---|
| Total flags on test set | 48,289 | 239,006 |
| **Flag density** | **4.07%** | **20.13%** |
| Row-level recall on pre-failure | 12.6% | 6.5% |
| **False-alarm rate on healthy rows** | **3.38%** | **21.22%** |
| False alarms per correct detection | 3.3 | 40.9 |

**The AI flags 6 × fewer rows overall, with a 6 × lower false-alarm
rate, while catching twice as many pre-failure rows.** An operator
running the threshold rule receives 233,310 false alarms per test
period; running XGBoost they receive 37,178. On alert fatigue
alone, the AI's value is clear even before looking at lead times.

Where both systems catch the same failure, XGBoost wins once by a
large margin (**MNR-016: 16.5 days vs 3.4 days — nearly 5 × earlier**)
and loses twice by small margins (under 1 day each). The large win
matters more than the small losses because 16 days of lead time is
qualitatively different operational value — it covers a full planned
maintenance cycle.

**Overall picture:**

| Detector | Failures caught | False-alarm rate | Operator takeaway |
|---|---|---|---|
| **XGBoost** (this submission) | **3 / 6** | **3.4%** | Clean signal, great lead times on its wins |
| Threshold rule alone | 5 / 6 | 21.2% | Noisy, reactive, floods inbox |
| LSTM-AE alone | 0 / 6 | 3.1% | Non-functional on this dataset — see §5.2 |

### 5.6 Short-dataset degradation (validate.py findings)

`src/validate.py` runs four tests on small (14-day) independent
datasets: hold-out failure type generalization, AI-vs-threshold race,
blind injection, and noise resilience. These tests expose a real
structural limitation worth stating plainly: **the model's longest
rolling features are 7-day windows, which means a 14-day dataset
only has ~7 days of fully-populated features.** On such short
datasets, the model is extremely conservative and recall collapses:

- **Hold-out**: 1 of 3 unseen failure types detected (max score on
  unseen `connector_corrosion` was 0.0003 — the model had never
  seen the pattern and produced almost no signal)
- **AI-vs-threshold (short window)**: 0 clear AI wins; both AI and
  threshold missed 9 of 10 cases because 14 days isn't enough
  runway for degradation signatures to develop or for 7-day rolling
  features to populate
- **Blind injection**: target not detected on a 14-day fleet
- **Noise resilience**: precision stays at 1.0 across 0-20% noise
  (when it flags, it's correct) but recall drops from 2.4% to 0%
  because the model becomes increasingly conservative as noise
  increases

**The correct operational framing:** this model is designed for
continuous multi-week deployment, where the 7-day rolling features
are always fully populated. Running it on short bursts is analogous
to asking a weather forecaster to predict tomorrow from 5 minutes of
barometric readings — there isn't enough history for the signal to
form. The main test set (§5.1-5.5) uses the full production-scale
dataset and shows the system's actual capability.

---

## 6. Security and Safety Analysis

Hardware control is an asymmetric-risk domain: the cost of a wrong
action (damaged chips, burned container, shortened fleet lifespan)
vastly exceeds the cost of a missed optimization. The system is
designed around this asymmetry.

### 6.1 Defense layers

1. **SafetyGuard** (`src/optimizer/safety.py`) is a mandatory
   chokepoint. No control action reaches the "hardware" layer without
   passing three independent checks:
   - **Thermal shutdown override.** If chip temperature ≥ 95°C, any
     non-maintenance action is rejected outright. Even a legitimate
     "boost frequency" signal is blocked if the chip is too hot.
   - **Rate limiting.** No set_frequency or set_voltage action within
     300 seconds of the last set_* action for the same miner.
     Prevents oscillation and PID-style instability.
   - **Value bounds clamping.** Every proposed frequency and voltage
     is clamped to the per-miner spec's `[min, max]` range. The
     model cannot accidentally request values outside hardware
     tolerances.
2. **Two-model architecture is in place for future use.** The
   LSTM-AE code path (training, thresholding, live-inference hook)
   is wired up even though the current trained model is non-functional
   (§5.2). If a working autoencoder or any other unsupervised
   detector replaces it, the rest of the pipeline does not need to
   change. For now, only XGBoost is an active detector.
3. **Interpretable ML.** XGBoost feature importance is inspectable.
   The top features after training are all physically meaningful
   (long-window voltage/hashrate trends, `te_health`, `temp_delta_c`).
   An operator can always ask "why was this flagged?" and get an
   answer.
4. **Threshold calibration on held-out data.** Decision thresholds
   are never tuned on the test set, eliminating the data-leakage
   class of bugs that would produce optimistic metrics in a report.

### 6.2 Known threats and limitations

| Threat | Mitigation |
|---|---|
| Adversarial telemetry (spoofed sensors) | Not addressed at model level — would require anomaly detection on the telemetry source itself. Listed as followup `F13` in `REMAINING_FIXES.md`. |
| Model drift from synthetic → real data | Expected. Pipeline is architected for retraining; the feature schema matches the MDK worker protocol so a data-source swap only requires a client implementation. |
| Compromised model weights on disk | Not addressed. Would require a signed-model registry and runtime verification. |
| Bugs in rule-based optimizer | Lower risk than bugs in an RL policy because rules are auditable and every action goes through SafetyGuard. |
| `sudden_chip_failure` slipping through predictive layer | Intentionally handled by reactive thermal shutdown instead. |

### 6.3 Live inference limitations (documented honestly)

The CLI dashboard in `src/cli/` runs a live fleet simulator and
displays AI predictions in real time. **Four pre-existing bugs in the
streaming inference path silently degrade live dashboard predictions**
and are documented in `src/cli/ai_bridge.py:load_models` and as fix
`F1` in `REMAINING_FIXES.md`. The dashboard should be treated as a
demonstration artifact, not an authoritative accuracy measurement —
the batch pipeline metrics above are the ground truth.

---

## 7. Conclusion and Next Steps

This prototype demonstrates a working AI-driven mining controller with
honest, held-out metrics on physics-plausible synthetic data. The
headline operational result is **3 of 6 test failures detected by
XGBoost, with average lead times of 7.6 days on its catches** —
enough runway to schedule maintenance during planned downtime rather
than reacting to thermal alarms. The strongest single result is the
16.5-day lead on `connector_corrosion`.

The dual-model architecture is in place but currently single-signal:
XGBoost does the detection work on its own, because the LSTM-AE
trained on this dataset does not separate failure sequences from
healthy ones (§2.1, §5.2). Closing that gap is the highest-value
followup. The `sudden_chip_failure` class is handled reactively by
`SafetyGuard` rather than by prediction, because it leaves no
learnable pre-failure signature.

### 7.1 Immediate followup (documented in `docs/REMAINING_FIXES.md`)

- **F1** — Fix CLI live inference feature/scaler mismatch (2-3 h)
- **F4** — Integrate baseline comparison against `temperature > 85`
  threshold rule, formalize numbers in this report
- **F5** — Per-failure-type breakdown (done in §5.3 above, integrated
  into `02_results.ipynb`)
- **F11** — Model versioning metadata sidecar for reproducibility

### 7.2 Gated on external dependency

- **F14** — Real MDK API client — requires data sharing channel with
  Tether's mining operations team. Pipeline is already structured to
  accept a real data source; only the `MDKClient` adapter is missing.
- **F12** — Multi-class XGBoost targeting specific failure types —
  would reduce LSTM dependence on `psu_degradation` and
  `coolant_restriction`, but needs a larger training set with better
  per-type balance than the current synthetic generator produces.

### 7.3 Open research questions

- Is a learned policy (RL) actually safer than the rule-based
  optimizer in the long run, given adequate guardrails? Literature
  suggests 10-17% improvement in performance-per-watt over static
  baselines, but safety evaluation is non-trivial.
- Can `sudden_chip_failure` be predicted at all from coarser
  timescales (hours-level aggregates of telemetry) even if minute-level
  sensors are too slow? Hypothesis: no, by construction of the failure
  mode. Experiment: retrain on hour-aggregated data and measure.

---

## Appendix — Reproducibility

Everything in this report is reproducible from a clean clone:

```bash
uv sync
uv run python -m src.run_pipeline     # ~50 min on Apple Silicon
```

Trained models are written to `data/models/`; rendered notebooks with
plots are in `notebooks/01_eda.executed.ipynb` and
`notebooks/02_results.executed.ipynb`. The full git history of the
refactors and fixes that produced these results is on the `main`
branch (7 commits as of v1.0).
