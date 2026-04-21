# Submission Guide — MDK AI Mining Controller

A reviewer's map of this submission. Start here if you're looking at
the repo for the first time.

## For the reviewer in a hurry (5 minutes)

Read these three things in order:

1. **`docs/TECHNICAL_REPORT.pdf`** (or `.docx` / `.md`) — the
   2-4 page end-to-end report. Problem, approach, KPI, results,
   safety, conclusion.
2. **`notebooks/02_results.executed.ipynb`** — pre-rendered metrics,
   ROC curves, per-failure-type detection tables, LSTM reconstruction
   error histograms. No need to run anything; plots are embedded.
3. **`README.md` §Known Limitations** — honest list of what's
   approximate, what's synthetic-data-only, and what's in the
   followup queue.

## For the reviewer with an hour

Plus:

4. **`docs/ARCHITECTURE.md`** — three mermaid diagrams covering
   dataflow, two-model rationale, and safety control loop.
5. **`notebooks/01_eda.executed.ipynb`** — pre-rendered EDA proving
   the synthetic data is physics-plausible and verifying the
   `is_pre_failure` label fix inline with assertions.
6. **`docs/REMAINING_FIXES.md`** — 14-item followup queue prioritized
   P0–P3, each with concrete file paths, effort estimates, and
   verification steps. This is the "what the author knows is
   incomplete" doc.

Then run it yourself:

```bash
git clone https://github.com/john-yo-ahn/mdk-mining-controller
cd mdk-mining-controller
uv sync

# Fast green-path sanity (≈12 min total — works on a fresh clone with no data)
uv run python -m src.cli test-te       # 10 KPI unit tests,         ~2s
uv run python -m src.cli test-cli      # 4 dashboard regressions,   ~15s

# To run `check` and `validate` (which need the synthetic dataset),
# download the companion data from HuggingFace:
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('johnahn/mdk-mining-controller-data',
                  repo_type='dataset', local_dir='data')
"

# Then verify end-to-end without retraining:
uv run python -m src.cli check         # 13 pipeline invariants,    ~11 min
uv run python -m src.cli validate      # 4 end-to-end tests,         ~9 min

# Or regenerate everything from the seeded generator (≈50 min):
uv run python -m src.run_pipeline
```

**Artifacts:**
- Code + trained models: https://github.com/john-yo-ahn/mdk-mining-controller
- Synthetic dataset (4 GB): https://huggingface.co/datasets/johnahn/mdk-mining-controller-data

## Assignment criteria → evidence map

Every evaluation criterion from `docs/CRITICAL_POINTS.md` maps to a
specific file or section:

| Criterion | Evidence | Where |
|---|---|---|
| **Clarity of technical reasoning** | Approach (§2), pipeline architecture (§3), baseline comparison (§5.5) | `docs/TECHNICAL_REPORT.md` |
| **Data pipeline quality** | 152-feature builder with caching; two-DuckDB-file storage with lock recovery; adaptive temporal split | `src/pipeline/features.py`, `src/storage/backend.py`, `01_eda.executed.ipynb` |
| **KPI design relevance and originality** | Assignment-compliant True Efficiency formula using all four §3.1.b variables (cooling power, chip voltage, environmental conditions, device operating mode); per-miner healthy-vs-failing separation +32.8% on `te_health`, and the rolling variants rank 8–9 in XGBoost feature importance | `src/kpi/true_efficiency.py`, `scripts/test_te_formula.py`, `TECHNICAL_REPORT.md` §4 |
| **Feasibility of AI use case** | Per-failure-type detection table shows 7 of 8 measurable failures caught by at least one model; XGBoost alone catches 4/6 with **11.3-day avg lead time**; head-to-head vs threshold baseline shows 6× better signal-to-noise | `TECHNICAL_REPORT.md` §5.3, §5.5; `02_results.executed.ipynb` |
| **Understanding of mining operations and constraints** | Mining economics framing (Hash Cost = Electricity × Efficiency); operator pain points from Giorgio's meeting notes | `TECHNICAL_REPORT.md` §1, `docs/CRITICAL_POINTS.md` |
| **Code and documentation clarity** | README, ARCHITECTURE, REMAINING_FIXES, TECHNICAL_REPORT, PROJECT_PLAN, SUBMISSION (this file), 10+ clean commits | `README.md`, `docs/` |
| **Awareness of security, safety, and control risks** | SafetyGuard three-layer chokepoint; threat table; live-inference limitations documented honestly | `TECHNICAL_REPORT.md` §6, `src/optimizer/safety.py` |

## Honest state as of submission

### What works

- **End-to-end training pipeline**: `src.run_pipeline` runs clean in
  ~50 minutes on Apple Silicon, producing XGBoost + LSTM-AE models
  with honest val-tuned metrics on a held-out test slice.
- **Per-failure-type detection**: on the held-out test set, 5 of 6
  distinct failure events are caught by at least one model (the one
  miss is `sudden_chip_failure`, which has only 2 rows and is
  intentionally handled by reactive thermal shutdown in SafetyGuard).
- **XGBoost lead times**: on the failures it catches, XGBoost flags
  **5.9 and 16.5 days** before cascade on `connector_corrosion`,
  giving operators meaningful runway for scheduled maintenance.
- **Baseline comparison**: honest head-to-head against a
  `temperature > 85 OR hashrate < 80%` threshold rule on the same
  test set shows XGBoost has **6× lower false-alarm rate** while
  catching twice as many pre-failure rows. See TECHNICAL_REPORT §5.5.
- **Reproducibility**: trained models ship with `.metadata.json`
  sidecars recording the git commit, training config, dataset sizes,
  validation metrics, feature hash, and threshold value. Review the
  sidecars without running anything.
- **Validation suite**: `src.validate` runs 4 tests (hold-out
  generalization, AI-vs-threshold race, blind injection, noise
  resilience) in ~5 minutes on independent synthetic datasets.

### What's approximate

- **Live CLI dashboard inference** is a streaming approximation, not
  production-accurate. The `MinerBuffer` in `src/cli/ai_bridge.py`
  computes ~80 of 152 features and silently zero-fills the rest at
  inference time. LSTM contribution is also degraded by a `seq_len`
  mismatch. Documented in `load_models()` startup output and in
  `REMAINING_FIXES.md` F1. Authoritative metrics come from the
  batch pipeline, not the dashboard.
- **Synthetic data only**. Every number is on physics-simulated
  telemetry. Real ASIC failures will look different; the pipeline
  is architected for real-data retraining once a data sharing
  channel exists.
- **Rule-based optimizer is unevaluated against an RL baseline**.
  Rule-based was chosen deliberately for hardware safety and
  auditability (see TECHNICAL_REPORT §2.2), but we do not have
  measured numbers comparing it to a learned policy.

### What's missing (acknowledged)

See `docs/REMAINING_FIXES.md` for the full 14-item catalog with
priority, effort, and verification steps. The highest-priority items
are:

- **F1**: CLI live inference fix (2-3 h) — biggest visible bug
- **F12**: multi-class XGBoost — would give the supervised side
  direct coverage of `psu_degradation` and `coolant_restriction`
  instead of relying solely on the LSTM-AE (which currently
  catches these at 21% and 11% sequence-level detection — the
  only signal on those failure modes today)
- **F14**: real MDK API integration — gated on external dependency

## Reproducibility

Everything in the submission is reproducible from a clean clone:

```bash
# One-time setup
uv sync

# Train end-to-end (~50 min on Apple Silicon M-series)
uv run python -m src.run_pipeline

# Validation tests (~5 min)
uv run python -m src.validate

# Launch the live dashboard
./run_dashboard.sh
```

Regenerable artifacts (all gitignored):
- `data/raw/mining_telemetry.parquet` — 5.2 M-row synthetic telemetry
- `data/raw/mdk.duckdb` — batch DuckDB (lock-recoverable)
- `data/processed/features.v2.parquet` — 3.2 GB engineered features
- `data/models/xgboost_failure.joblib`, `lstm_ae.pt` — trained models
  (each with a `.metadata.json` sidecar committed to git for
  reviewer inspection)

Pre-rendered artifacts (committed, viewable without running):
- `notebooks/01_eda.executed.ipynb` — 690 KB with embedded plots
- `notebooks/02_results.executed.ipynb` — 232 KB with embedded plots
- `docs/TECHNICAL_REPORT.docx` — 23 KB, opens in Word
- `docs/TECHNICAL_REPORT.pdf` — if `pandoc` or LibreOffice was
  available at submission time

## Commit history

The full history of refactors and fixes that produced these results
lives on the `main` branch. Key commits:

- `fix(ml): label alignment, val split, F1-floor, scale defaults` —
  the single biggest correctness fix of the project
- `docs: per-failure-type detection breakdown` — the results story
- `fix(cli,validate): document live-inference limits` — the honesty
- `docs(report): add TECHNICAL_REPORT in markdown and Word` — the
  headline deliverable

Run `git log --oneline` for the full picture.
