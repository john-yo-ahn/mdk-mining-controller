# Project Plan & Direction

## Strategic Direction

### Core Thesis
Mining profitability is a cost-management game. Since hashprice is market-determined, the only levers are **efficiency** and **operational cost**. Our AI controller targets both:
- **Predictive maintenance** reduces costs by preventing expensive unplanned downtime
- **Dynamic optimization** improves efficiency by finding optimal operating points per-chip, per-condition

### Why Combined Approach
Giorgio's meeting feedback highlighted that the two biggest problems are (1) breakage and (2) efficiency optimization. Rather than choosing one track, we build a shared data pipeline that feeds both modules. The data engineering is the same — the difference is in what the model outputs.

### Design Philosophy
- **Design thinking over final solutions** (per Giorgio's guidance)
- **Interpretable models** — operators need to trust and understand AI decisions
- **Safety-first** — hardware control demands conservative defaults and human override
- **Synthetic data driven** — all prototyping on generated data, architecture ready for real data swap

---

## Week-by-Week Plan

### Week 1: Data Foundation (Apr 7-13)

**Goal**: Working data pipeline, exploratory analysis, KPI design

Tasks:
- [x] Generate synthetic mining telemetry dataset
  - **Delivered**: 30 ASIC miners × 120 days (5.2 M rows), physics-first generator
  - 7 failure scenarios: gradual_degradation, thermal_runaway, fan_stall,
    psu_degradation, sudden_chip_failure, coolant_restriction, connector_corrosion
  - Physics models: `P = CV²f + leakage(T)`, RC thermal network
  - See `src/synthetic/generator.py`, `src/synthetic/physics.py`
- [x] Build ingestion pipeline
  - Parquet + DuckDB with lock-recovery (`src/storage/backend.py`)
  - Preprocessing in `src/pipeline/preprocessing.py`
  - Per-miner normalization via `DEFAULT_MINER_SPECS` in `src/config.py`
- [x] Exploratory Data Analysis
  - `notebooks/01_eda.ipynb` — 20 cells, all executed
  - `notebooks/01_eda.executed.ipynb` (690 KB) with rendered plots
  - Correlation matrix, phase timelines, J/TH vs TE_health comparison
  - Label-fix verification asserted inline
- [x] Design and implement True Efficiency (TE) KPI
  - `src/kpi/true_efficiency.py` — base, adjusted, health variants
  - **Validated**: `te_health` separates healthy/failing by 114.4%
    vs ~14% for naive J/TH
  - `te_health` is one of the top-10 features XGBoost actually uses
- [x] Create initial visualizations
  - Time-series per-device traces (EDA notebook section 4)
  - Per-miner phase timelines (EDA notebook section 7)
  - KPI distribution plots (EDA notebook section 5)

**Deliverable**: Jupyter notebook with EDA ✅, working pipeline ✅, TE KPI ✅

### Week 2: AI Models (Apr 14-20)

**Goal**: Predictive maintenance prototype, anomaly detection

Tasks:
- [x] Feature engineering for predictive maintenance
  - **152 features** at 6 rolling window scales (2m/15m/1h/6h/1d/7d)
  - Rate of change, trend slopes (7-day OLS), variance trends
  - Cross-signal correlations, autocorrelation, peak counts
  - Diurnal amplitude, cross-miner container features
  - Cached to `data/processed/features.v2.parquet` with explicit
    `FEATURES_VERSION` constant for invalidation
- [x] XGBoost failure prediction model
  - Binary classification, `is_pre_failure` derived from
    `degradation_phase ∈ {incubation, acceleration}`
  - Adaptive three-way temporal split (train 55% / val 15% / test 30%
    of cumulative positives)
  - `tree_method='hist'` (30× faster than exact), sqrt-capped
    `scale_pos_weight`, F1-with-floor threshold strategy
  - **AUC 0.801, 3/6 test failures, avg 7.6-day lead time**
  - Full feature importance analysis in `notebooks/02_results.ipynb`
- [x] LSTM-Autoencoder anomaly detection
  - Trained on healthy-only sequences (649k sequences × 60 min)
  - Persistent global scaler serialized with model weights
  - Early stopping + best-weight restore
  - MPS (Apple Silicon GPU) training, 2120s wall clock
  - **Separation ratio 2.66×, 2.6% healthy FAR, 33.4% seq detection**
  - **Catches 5/5 failures with measurable sequences** — including
    the two failure types XGBoost has blind spots on
- [x] Rule-based efficiency optimizer (stretch)
  - `src/optimizer/rules.py` — thermal, energy price, degradation rules
  - `src/optimizer/safety.py` — SafetyGuard with thermal shutdown
    override, rate limiting, value bounds clamping
  - Already written before session; verified functional and decoupled
    from ML layer

**Deliverable**: Trained models ✅, evaluation metrics ✅, optimizer logic ✅

### Week 3: Integration & Report (Apr 21-27)

**Goal**: End-to-end demo, technical report, architecture diagram

Tasks:
- [x] Integrate pipeline + models into unified controller
  - `src/cli/app.py` — Textual dashboard
  - `src/cli/simulation.py` — 30-miner live fleet simulator
  - `src/cli/ai_bridge.py` — loads trained models, risk-level
    escalation system
  - **Known limitation**: live inference path has four pre-existing
    bugs (feature set mismatch, LSTM seq_len mismatch, scaler ignored,
    hardcoded risk threshold). Documented in `docs/REMAINING_FIXES.md`
    as fix F1. Live alerts are approximate, batch pipeline metrics are
    authoritative.
- [x] Create architecture diagram
  - `docs/ARCHITECTURE.md` — three mermaid diagrams covering dataflow,
    two-model rationale, and safety control loop
- [x] Write technical report (2-4 pages)
  - `docs/TECHNICAL_REPORT.md` — problem, approach, pipeline, KPIs,
    results with per-failure-type breakdown, baseline comparison
    against simple threshold rule, security and safety analysis
- [x] Polish and package
  - `README.md` with quick start, design decisions, honest results
  - `docs/REMAINING_FIXES.md` — 14-item P0-P3 followup queue
  - `docs/ARCHITECTURE.md` with mermaid diagrams
  - `notebooks/01_eda.executed.ipynb` and
    `notebooks/02_results.executed.ipynb` for reviewers who don't
    want to re-run the pipeline

**Deliverable**: Complete submission package ✅

---

## Key Technical Decisions

### Model Selection Rationale

| Decision | Choice | Why |
|---|---|---|
| Primary maintenance model | XGBoost | Best on tabular data, interpretable, fast inference |
| Anomaly detection | LSTM-Autoencoder | Temporal patterns, unsupervised (no failure labels needed) |
| Optimizer approach | Rule-based (not RL) | Transparent, auditable, safe for hardware. RL is stretch goal with clear migration path. |
| Synthetic data | Physics-based generation | Thermodynamically plausible, controllable failure injection |

### KPI Design Rationale
- Simple J/TH ignores cooling overhead, environment, and degradation
- True Efficiency (TE) captures the full picture by including cooling power and environmental factors
- Degradation-aware TE gives operators early warning when a machine is declining economically

### Architecture Choices
- **Python** for all data/ML work (assignment requirement, ecosystem strength)
- **Jupyter notebooks** for exploration and presentation
- **Modular pipeline**: Each stage (ingestion, features, model, output) is a separate module for testability
- **MDK-compatible data schema**: Structure synthetic data to match what MDK workers would produce, so the pipeline is ready for real data integration

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Synthetic data doesn't represent real failures | Use physics-based models grounded in semiconductor failure literature |
| No access to real MDK API for integration testing | Design pipeline to match MDK data schemas from documentation |
| 3-week timeline is tight | Tier 1 is fully achievable in 2 weeks; Tier 2 is stretch with clear cutoff |
| Model overfits to synthetic patterns | Use temporal train/test splits, evaluate on unseen failure scenarios |

---

## Resources

- **MDK Docs**: docs.mdk.tether.io
- **MOS Docs**: docs.mos.tether.io
- **MOS Demo**: mos.tether.io
- **MDK GitHub**: github.com/tetherto/mdk-be
- **Mentor**: Gio Galt (@gio_galt on Telegram)
