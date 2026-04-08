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
- [ ] Generate synthetic mining telemetry dataset
  - Model 50 ASIC miners over 30 days
  - Include: clock freq, voltage, hashrate, temperature, power, ambient temp
  - Inject 5 failure scenarios (gradual degradation, sudden failure, thermal runaway, fan stall, PSU issues)
  - Use physics-based models: P = C*V^2*f + leakage, RC thermal network
- [ ] Build ingestion pipeline
  - CSV/JSON parser with timestamp alignment
  - Missing value handling (forward-fill for sensors, flag for gaps > 5min)
  - Per-device normalization
- [ ] Exploratory Data Analysis
  - Correlation matrix: freq vs hashrate, voltage vs power, temp vs efficiency
  - Distribution analysis per operating mode
  - Identify natural clusters in operating behavior
- [ ] Design and implement True Efficiency (TE) KPI
  - Base TE formula incorporating cooling overhead
  - Environmental adjustment factor
  - Degradation-aware variant
  - Validate KPI captures known inefficiencies better than raw J/TH
- [ ] Create initial visualizations
  - Time-series dashboards per device
  - Fleet-level heatmaps
  - KPI distribution plots

**Deliverable**: Jupyter notebook with EDA, working pipeline, TE KPI implementation

### Week 2: AI Models (Apr 14-20)

**Goal**: Predictive maintenance prototype, anomaly detection

Tasks:
- [ ] Feature engineering for predictive maintenance
  - Rolling statistics at 2min, 15min, 1hr windows
  - Rate of change features (dT/dt, dHashrate/dt)
  - Cross-signal ratios
  - Degradation slope (rolling linear regression on J/TH)
- [ ] XGBoost failure prediction model
  - Binary classification: healthy vs pre-failure (24hr horizon)
  - Train/test split respecting temporal ordering
  - Feature importance analysis
  - Confusion matrix, precision/recall evaluation
- [ ] LSTM-Autoencoder anomaly detection
  - Train on healthy-only operational data
  - Reconstruction error as anomaly score
  - Threshold selection via percentile on validation set
  - Compare detection performance vs XGBoost
- [ ] Rule-based efficiency optimizer (stretch)
  - Define condition-action rules based on meeting insights
  - Thermal management rules (reduce freq when approaching thermal limit)
  - Energy-price-aware rules (boost during cheap energy, throttle during expensive)
  - Simulate optimizer decisions on synthetic data

**Deliverable**: Trained models, evaluation metrics, optimizer logic

### Week 3: Integration & Report (Apr 21-27)

**Goal**: End-to-end demo, technical report, architecture diagram

Tasks:
- [ ] Integrate pipeline + models into unified controller
  - Input: streaming telemetry (simulated)
  - Output: maintenance alerts + optimization recommendations
  - Safety constraint enforcement layer
- [ ] Create architecture diagram
  - End-to-end flow: Hardware -> Pipeline -> Features -> AI -> Commands
  - Show data flow, model placement, safety boundaries
- [ ] Write technical report (2-4 pages)
  - Problem statement and mining economics context
  - Approach and algorithm selection justification
  - Pipeline architecture
  - KPI design and validation
  - Results and evaluation
  - Security and safety analysis
- [ ] Polish and package
  - Clean up notebooks and code
  - Add documentation/comments
  - Ensure reproducibility (requirements.txt, README)

**Deliverable**: Complete submission package

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
