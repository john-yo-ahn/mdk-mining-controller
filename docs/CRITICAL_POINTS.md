# Critical Points & Meeting Insights

## Assignment Requirements (3-Week Deadline)

### Deliverables
1. **Technical Report** (2-4 pages): Problem, approach, pipeline, KPIs, benefits, safety
2. **Prototype Codebase** (Python): Ingestion, feature engineering, KPI, model, visualization
3. **Architecture Diagram**: Hardware -> Telemetry -> Features -> AI Controller -> Commands

### Tier 1 (Required): Telemetry Analysis & Data Pipeline
- Ingest synthetic mining telemetry data
- Map relationships: clock frequency, voltage, hashrate, temperature, power consumption
- Identify correlations, trade-offs, anomalies
- Design a **True Efficiency (TE)** KPI beyond simple J/TH incorporating cooling, voltage, environment, operating mode

### Tier 2 (Stretch): AI Model Prototype
- **Option A**: Dynamic Overclocking/Underclocking (RL agent adjusting settings based on energy price, ambient temp, efficiency, chip performance)
- **Option B**: Predictive Maintenance (RF/XGBoost/LSTM detecting pre-failure patterns in temperature, power instability, hashrate anomalies)

---

## Key Meeting Insights from Giorgio (Gio)

### What the team actually needs
- **Automate manual operator decisions**: Operators currently adjust chip frequency, cooling, and power modes based on experience and environmental cues — this doesn't scale
- **Two biggest pain points**: (1) Chip/machine breakage causing costly repairs and downtime, (2) Optimizing chip efficiency at scale across thousands of machines
- **Pool mining revenue model**: Operators are paid per hash (continuous), NOT dependent on finding blocks — so every efficiency improvement directly translates to revenue

### What "AI agent" means in this context
- NOT large language models — local intelligent algorithms running on streaming sensor data
- Think: control loops that react to changing conditions faster than a human operator can
- The agent should make the same decisions an experienced operator makes, but at scale and continuously

### Platform context
- **MOS** = Mining Operating System (full application, React frontend + Node.js backend)
- **MDK** = Mining Development Kit (underlying SDK/framework that MOS is built on)
- MDK v0.1 targeted for end of April 2026
- Both are open-source (Apache 2.0) by Tether

### Data situation
- Real operational data is sensitive — management won't share breakage/performance data externally
- Must work with **synthetic data** (provided or self-generated)
- The demo at mos.tether.io shows the live monitoring interface
- Focus on **design thinking and solution frameworks** over fully tested models

### Economic fundamentals (critical for KPI design)
- `Hashprice = (Subsidy + Fees) * BTC Price / (Difficulty * 2^32)` — market-determined, NOT controllable
- `Hash Cost = Electricity Cost * Hashing Efficiency` — THIS is what we optimize
- `Gross Profit = (Hash Price - Hash Cost) * Miner Hash Rate`
- **Controllable**: Miner hash rate, direct operational cost (energy + maintenance + efficiency)
- **Not controllable**: Network hash rate, BTC price, block subsidy, difficulty
- Efficiency (J/TH) is THE key competitive metric
- Mining investment cycles: 3-5 years amortization
- Breakeven = where revenue equals all-in cost (Capex + Opex per MWh)

---

## Platform Technical Details

### MOS Telemetry Available
| Category | Data Points |
|---|---|
| Performance | Hashrate (real-time + historical), efficiency, power mode (High/Normal/Low/Sleep) |
| Thermal | Chip temperature, oil temp, water temp, supply liquid temp, container humidity |
| Power | Watt consumption (5-sec intervals), site power thresholds |
| Pool | Connection status, hashrate distribution, pool config, worker name |
| Environmental | SENECA temp probes, OpenWeather integration |
| Device | IP, MAC, serial, model, position, firmware version |

### MDK Architecture
1. **Adapters/Workers** — device-specific modules (per brand/model)
2. **Orchestrator (ORK)** — lifecycle management, safety, event propagation
3. **API Layer (App Node)** — vendor-agnostic REST API

### Supported Hardware
- Bitmain Antminer (S19 XP, S21, S21 Pro)
- MicroBT Whatsminer (M30S+, M53S, M56S, M63)
- Canaan Avalon (A1346)
- Immersion and hydro cooling containers (Bitdeer, Bitmain Antspace)

---

## Evaluation Criteria (from assignment)
- Clarity and quality of technical reasoning
- Quality of data pipeline and data modeling
- Relevance and originality of KPI design
- Feasibility of AI use case
- Understanding of mining operations and constraints
- Clarity of code and documentation
- Awareness of security, safety, and control risks
