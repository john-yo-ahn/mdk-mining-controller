# Best Practices Research: AI for Mining Optimization

## 1. Predictive Maintenance for ASIC Hardware

### Recommended Algorithms (by priority)

| Algorithm | Strengths | Use Case |
|---|---|---|
| **XGBoost** | ~98% accuracy on tabular sensor data, fast inference, interpretable feature importance | Primary failure prediction model |
| **Random Forest** | Near-identical accuracy, simpler tuning, good uncertainty estimation | Baseline / ensemble component |
| **LSTM-Autoencoder** | Captures temporal degradation trends, unsupervised (no labeled failures needed) | Anomaly detection on normal-only training data |
| **Autoencoders** | Dimensionality reduction, unsupervised anomaly scoring | Feature compression, frequency-domain analysis |

### Critical Features for Failure Prediction
1. **Chip junction temperature** — strongest predictor; every 10C above rated temp ~halves lifespan (Arrhenius)
2. **Temperature gradient / rate of change** — rapid thermal cycling causes solder fatigue
3. **Hashrate deviation from nominal** — 10-15% drop indicates degradation
4. **J/TH trend over time** — efficiency degradation precedes failure
5. **Voltage regulator stability** — ripple/droop indicates capacitor degradation
6. **Nonce rejection rate** — early indicator of computational degradation

### Common ASIC Failure Modes
| Mode | Mechanism | Timeline |
|---|---|---|
| Solder joint fatigue (BGA) | Thermal cycling micro-cracks | 12-36 months |
| Electromigration | High current density metal migration | Accelerates >85C |
| Capacitor degradation | Electrolytic cap loses capacitance (Arrhenius) | 18-36 months |
| Fan/cooling failure | Mechanical bearing wear | 12-24 months |
| PSU degradation | Cap aging, thermal stress | 24-48 months |

**Key insight**: Chips rarely fail outright. Supporting components degrade first, and hashrate slips gradually.

---

## 2. Dynamic Frequency/Voltage Scaling (DVFS) via RL

### Algorithm Choice
- **SAC (Soft Actor-Critic)**: Best for continuous action spaces (freq/voltage are continuous). Off-policy = can learn from historical data. Entropy regularization encourages exploring non-obvious operating points.
- **PPO**: Safer fallback — conservative policy updates prevent hardware-damaging changes. Better for initial deployment.
- **DDQN**: Viable if discretizing to finite operating points (e.g., High/Normal/Low/Sleep modes).

### Reward Function Design
```
R = w1 * hashrate_reward
  - w2 * power_penalty
  - w3 * thermal_violation
  - w4 * switching_cost
```
- **hashrate_reward**: Actual vs target hashrate
- **power_penalty**: Energy consumed (Joules) — maps to opex
- **thermal_violation**: Quadratic penalty above warning threshold (e.g., 80C)
- **switching_cost**: Penalizes rapid freq/voltage changes to reduce thermal cycling

Literature shows RL-based DVFS achieves ~17% improvement in performance-power ratio over static baselines.

---

## 3. KPI Design Beyond Simple J/TH

### Proposed True Efficiency Framework

| KPI | Formula | Purpose |
|---|---|---|
| **Effective J/TH** | Total facility energy (incl. cooling) / actual hashrate | Real-world efficiency |
| **PUE** | Total facility power / IT power | Best facilities: 1.03-1.10 (immersion) |
| **Hashrate Realization** | Actual hashrate / nameplate rated | Healthy fleet: >95% |
| **Degradation Rate** | J/TH slope over time per machine | Tracks aging curve |
| **Cost per BTC** | Total opex / BTC earned | Ultimate operator metric |

### Environmental Adjustments
- Temperature-adjusted efficiency: 5-15% loss in hot climates vs cold
- Humidity impact: reduces cooling efficiency, accelerates corrosion
- Altitude: lower air density reduces air-cooling effectiveness

**Key insight**: Economic lifespan (2-4 years, when profit hits zero) << Physical lifespan (4-7 years). Optimize for economic efficiency.

---

## 4. Anomaly Detection for Streaming Telemetry

### Layered Architecture
1. **Statistical baselines** (lightweight, real-time): EWMA for drift, CUSUM for persistent deviations, Z-score on rolling windows
2. **LSTM-Autoencoder** (deep learning): Train on normal data, anomaly = high reconstruction error, captures cross-signal correlations
3. **Graph attention models** (advanced): Model both temporal and inter-sensor dependencies

### Window Sizes
| Window | Duration | Use |
|---|---|---|
| Short | 30s - 2min | Point anomalies (sudden chip failure, fan stall) |
| Medium | 5-15min | Contextual anomalies (gradual overheating, hashrate drift) |
| Long | 1-24hr | Trend anomalies, degradation, diurnal cycles |

### Feature Engineering
- Rolling statistics (mean, std, min, max, skew, kurtosis) across multiple windows
- Rate of change (first derivative) of temperature, hashrate, power
- Cross-signal ratios: J/TH, temp-per-watt, fan-RPM-per-degree
- Frequency domain: FFT/wavelet for oscillatory failure signatures

---

## 5. Synthetic Data Generation

### Hybrid Approach: Physics + Generative

**Physics-based baseline:**
- Thermal dynamics: RC thermal network models (RthJC <= 0.18 C/W for healthy chips)
- Power: P = C * V^2 * f + leakage (exponential with temperature) — fundamental CMOS equation
- Hashrate: function of frequency, voltage, and temperature (throttling model)

**Degradation overlays:**
- Electromigration: gradual resistance increase (Arrhenius kinetics)
- Capacitor aging: exponential capacitance loss
- Solder fatigue: increasing thermal resistance at chip-board interface
- Fan degradation: RPM decline, bearing vibration frequency increase

**Generative models:**
- **TimeGAN** (recommended): Learns static + dynamic time-series characteristics simultaneously
- **Diffusion models**: Best for generating rare failure signatures from limited real examples

**Failure signature templates:**
- Sudden chip failure: step function hashrate drop
- Thermal runaway: exponential temperature increase with positive feedback
- Fan stall: RPM -> 0, followed by rapid temp rise
- PSU degradation: increasing voltage ripple, intermittent sags
- Gradual degradation: slow linear hashrate decline over weeks
