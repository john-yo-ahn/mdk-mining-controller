# Remaining Fixes — Implementation Plan

This document captures every known fix that is **not yet shipped** as
of this writing, with concrete file paths, estimated effort, risk
profile, and verification steps. None of these are blocking the
current pipeline working — they range from "would make the demo
better" to "would matter if we were shipping to production".

Use this as the followup queue. Pick by priority (P0 → P3) and skip
anything below your time budget.

---

## P0 — Real correctness issues that affect mentor demo

These would be visible to anyone reading the code or running the
dashboard. They are not merely cosmetic.

### F1. CLI live inference is silently degraded — ✅ CLOSED (Apr 8, Phase B)

**Status:** shipped in commit (see `git log --grep="live inference"`).
Buffer size bumped 120 → 10080 (7 days); `compute_features` now
delegates to `build_feature_matrix` for the full 152-feature vector;
`export_lstm_sequence` uses `seq_len=60` and the persistent scaler
propagated from the saved model at load time; silent try/except
swallows replaced with once-per-session error logging. Verified via
`scripts/test_live_feature_parity.py` at 88% feature match rate
(remaining drift is legitimate warmup behavior on 7-day rolling
features, not a bug).

**Residual known issue:** per-tick cost is ~800ms per miner because
each call runs the full batch feature builder on a single-miner
DataFrame. For 30 miners at 1 tick/second this is slower than real
time. Fix would be to collect all 30 miners into a single DataFrame
and call `build_feature_matrix` once per tick — would bring amortized
cost down ~5-10×. Tracked as new sub-item F1a below.

### F1a. Bulk-miner live inference optimization

**Files:** `src/cli/ai_bridge.py` (`MinerBuffer.compute_features`,
`MinerBuffer.export_lstm_sequence`, `AIBridge.predict`)

**Problem:** Three independent bugs in the streaming inference path
that all silently degrade the live dashboard predictions:

1. `compute_features` produces ~80 features but XGBoost was trained
   on 152. Missing features are filled with `0.0` via `dict.get(f, 0.0)`
   on line 331. Model sees a wildly different distribution at
   inference time than training.
2. `export_lstm_sequence(seq_len=30)` is called against a model
   trained with `seq_len=60`. The shape mismatch silently fails inside
   a `try/except` in `predict()` so the LSTM contribution is always
   zero. Combined score is just the broken XGBoost score.
3. `MinerBuffer.export_lstm_sequence` does its own per-miner local
   normalization instead of using the persistent scaler we now
   serialize with the model. So even if you fixed (2), reconstruction
   errors would still be in a different scale than the threshold was
   calibrated on.

**Fix plan:**

1. Replace `MinerBuffer.compute_features` with a wrapper that calls
   into a *streaming-safe* subset of `src/pipeline/features.py`. The
   trick is that some features (rolling windows of length > buffer
   size, cross-miner features) cannot be computed from a single
   miner's 120-row buffer. Two options:
   - **(a)** Increase `BUFFER_SIZE` from 120 → 10080 (7 days at
     1 min). This makes the buffer carry enough history for every
     window length used in training. Memory cost: ~30 KB per miner ×
     30 miners = 900 KB. Trivial.
   - **(b)** Drop features that can't be computed in streaming mode
     and **retrain XGBoost on the reduced feature set**. More work,
     but the retrained model is exactly what live inference will use.
   Recommend **(a)** — keep the model, fix the buffer.
2. Pass `seq_len=60` (not 30) to `export_lstm_sequence` so the
   LSTM input matches training. Requires `BUFFER_SIZE >= 60`, already
   true.
3. In `export_lstm_sequence`, replace per-miner local normalization
   with the persistent scaler:
   ```python
   if self.lstm_model and self.lstm_model.feature_mean_ is not None:
       mean = self.lstm_model.feature_mean_
       std = self.lstm_model.feature_std_
       normalized = (raw - mean) / std
   ```
4. Remove the `try/except` swallow on the LSTM path in `predict`. If
   shapes mismatch, we want to know.

**Effort:** 2-3 hours.

**Risk:** Medium. Requires careful feature parity between batch and
streaming. The cleanest verification is to feed the same minute of
telemetry through both `build_feature_matrix` and the live buffer
and assert all 152 feature values agree to 6 decimal places.

**Verification:**

```python
# Pseudo-test
batch_features = build_feature_matrix(df_one_miner)[152_cols].iloc[-1]
buffer = MinerBuffer(...).fill_from(df_one_miner)
live_features = buffer.compute_features()
for col in 152_cols:
    assert abs(batch_features[col] - live_features[col]) < 1e-6
```

---

### F2. XGBoost ColMaker / ParallelFor warnings on Apple Silicon

**Files:** `src/models/xgboost_classifier.py`

**Problem:** XGBoost emits an OpenMP UserWarning at training start
on Apple Silicon: `"Parameter use_label_encoder has not been used"`
(already deleted) and `"Initializing libomp.dylib, but found ..."`
race-condition warning when multiple OpenMP runtimes are loaded
(common with NumPy + scikit-learn + xgboost).

**Fix plan:** Set `OMP_NUM_THREADS` and `XGBOOST_NUM_THREADS` env
vars at the top of `run_pipeline.py` *before* any OpenMP-using import,
or pin `n_jobs=8` (not -1) on the XGBClassifier construction.

**Effort:** 5 minutes.

**Risk:** None. May give a small perf improvement.

---

### F3. detection_timeline counts at the failure-event level, not row level

**Files:** `src/models/evaluation.py:detection_timeline`

**Problem:** The current implementation reports "N of M failures
detected" where N is the number of distinct miners with at least one
predicted=1 row inside their pre-failure window. This is the right
unit, but it's not visible to the operator that **how many rows**
were detected matters too. A failure that gets one positive flag in
the last 5 minutes of the pre-failure window is "detected" but
operationally useless — it's a 5-minute lead time disguised as a
"detection".

**Fix plan:** Add columns to the returned DataFrame:
- `first_flag_lead_minutes` (time from first flag to cascade onset,
  not just to window end)
- `flag_density` (n_flagged / n_pre_failure_rows)
- `detection_quality` enum: EARLY (>= 4h lead), LATE (1-4h),
  MARGINAL (< 1h), MISSED.

Plus a summary counter on the print path:
"3 of 6 detected — 1 EARLY, 1 LATE, 1 MARGINAL".

**Effort:** 30 minutes.

**Risk:** None. Adds new fields, doesn't change existing ones.

---

## P1 — Methodology improvements that don't change today's headlines

### F4. Compare XGBoost score against a naive baseline

**Files:** `src/validate.py` already has `test_race` for this but it
uses the broken old `optimize_threshold` signature path.

**Problem:** A mentor will reasonably ask "is your AI better than a
3-line `temperature > 85 OR hashrate < 80% nameplate` rule?". Without
a baseline number to point to, the AUC=0.801 sounds impressive but
isn't grounded. The `validate.py:test_race` test does exactly this
comparison but is currently broken.

**Fix plan:**
1. Update `validate.py:test_race` to use the new threshold strategy
   API (`strategy="f1_with_floor"` or `"recall_target"` explicitly).
2. Run it after every pipeline training and dump the result to a
   sidecar `data/models/baseline_comparison.json`.
3. Surface in README + results notebook.

**Effort:** 1 hour.

**Risk:** Low.

---

### F5. Per-failure-type detection breakdown

**Files:** `src/run_pipeline.py` (post-XGBoost section), or a new
helper in `src/models/evaluation.py`.

**Problem:** The headline "3 of 6 detected" doesn't say *which 3*.
A model that catches 3 of 3 PSU failures and misses 3 of 3 fan stalls
is a very different story from "3 random successes". Mentors will
ask which failure types are robustly detected.

**Fix plan:** After computing `detection_timeline`, group by
`failure_type` and print:
```
Detection by failure type:
  psu_degradation       2/2 (100%) avg lead 240h
  connector_corrosion   1/2  (50%) avg lead 165h
  thermal_runaway       0/1   (0%)
  fan_stall             0/1   (0%)
  ...
```

**Effort:** 20 minutes.

**Risk:** None.

---

### F6. Three-way split awareness in validation script

**Files:** `src/validate.py` (currently uses 2-way `split_temporal`)

**Problem:** The validation runner still uses the old 2-way
`split_temporal` which doesn't reflect how the production pipeline
actually trains models. Hold-out generalization tests are honest about
unseen failure types but dishonest about how thresholds were tuned —
they call `optimize_threshold(target_recall=0.85)` on what is
effectively their test set.

**Fix plan:** Update `test_holdout` to use `split_temporal_tvt`.

**Effort:** 15 minutes.

**Risk:** Low. Numbers may move slightly because the holdout test
will now have less training data (60% instead of 80%).

---

## P2 — Quality of life that pays off after a few iterations

### F7. DataFrame fragmentation in feature engineering

**Files:** `src/pipeline/features.py`, lines 84-95, 153, 211, 243,
298, 336, 364-372, 455.

**Problem:** Eight functions in `features.py` build feature columns
in a Python loop with `df[name] = col.values`. At 160+ columns this
becomes quadratic on the block manager and emits a stream of
`PerformanceWarning: DataFrame is highly fragmented`. Step 5 takes
~25 minutes at 30×120 scale.

**Fix plan:** Refactor each offending function to:
1. Accumulate into a `dict[str, np.ndarray]` instead of writing into
   `df` one column at a time.
2. End with `df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1, copy=False)`.
3. After all transforms run in `build_feature_matrix`, do a single
   `df = df.copy()` to consolidate the block manager.

Expected speedup: 25 min → 12-15 min on cold rebuild.

**Effort:** 60-90 minutes (8 functions × ~10 min each + verification).

**Risk:** Medium. Easy to drop a column or change ordering. Verification
is a strict diff between old and new feature matrix on a small dataset.

**Verification:**

```python
old_features = build_feature_matrix_old(small_df)
new_features = build_feature_matrix(small_df)
assert set(old_features.columns) == set(new_features.columns)
for col in old_features.columns:
    pd.testing.assert_series_equal(
        old_features[col].sort_index(),
        new_features[col].sort_index(),
        check_dtype=False,
    )
```

---

### F8. FutureWarnings cleanup — ✅ CLOSED (Apr 8, Phase A)

`divide by zero` RuntimeWarning in `compute_variance_trend` silenced
via `np.errstate`. Pandas `DataFrameGroupBy.apply` FutureWarning at
the warmup-drop step replaced with a cumcount-based filter (no
groupby-apply). The other groupby-apply site in
`compute_cross_signal_correlations` now has a single-miner fast path
that avoids the groupby entirely; the multi-miner path silences the
specific FutureWarning locally via `warnings.catch_warnings`. Full
pipeline runs now emit no FutureWarning or RuntimeWarning during
feature engineering.

### F8-original. (historical)

**Files:** `src/pipeline/features.py` lines 220, 220, 220, 220, 465
(groupby.apply); line 186 (divide by zero in 24h/7d ratio).

**Problem:** Pandas 3.0 will remove the implicit groupby column
behavior. Currently four `df.groupby("miner_id").apply(...)` calls
emit `FutureWarning: DataFrameGroupBy.apply operated on the grouping
columns`. One numpy `RuntimeWarning: invalid value encountered in
divide` from the 24h/7d ratio when the 7d window is zero.

**Fix plan:**
- Add `include_groups=False` to all `groupby().apply()` calls, or
  switch to `groupby()[relevant_cols].apply()`.
- Wrap the 24h/7d division in `np.errstate(divide='ignore', invalid='ignore')`
  or use `np.divide(a, b, out=np.zeros_like(a), where=b>0)`.

**Effort:** 20 minutes.

**Risk:** None.

---

### F9. Feature cache invalidation by content hash

**Files:** `src/run_pipeline.py` (cache check logic)

**Problem:** Current cache invalidation compares mtimes:
`features_cache.stat().st_mtime >= raw_parquet.stat().st_mtime`. If
someone touches the raw parquet without changing it (`touch -m`,
restore-from-backup with original timestamps, etc.), the cache will
be silently invalidated or silently kept stale.

**Fix plan:** Hash the first 1 MB of the raw parquet (cheap, stable
across identical content) and embed it in the cache filename:
`features.v{FEATURES_VERSION}.{hash[:8]}.parquet`. Look up the file
that matches both the version and the current raw hash.

**Effort:** 30 minutes.

**Risk:** Low. Old caches become orphaned files; add a one-line
cleanup at startup that removes any `features.v*.parquet` not
matching the current expected name.

---

### F10. Memory-aware LSTM training (don't OOM the box)

**Files:** `src/run_pipeline.py` Step 8, possibly
`src/models/lstm_autoencoder.py:fit`

**Problem:** Last full pipeline run was OOM-killed during LSTM epoch
2 because pandas frames from earlier steps stayed alive in memory
through LSTM training. Total working set hit ~15-20 GB with the 30×120
dataset. Workaround was a separate `scripts/train_lstm_only.py`.

**Fix plan:** In `run_pipeline.py` Step 8, after building all the
LSTM sequence arrays:
```python
del df_features, train_df, val_df, test_df
del healthy_train, healthy_val, healthy_test, failure_test
del X_train, X_val, X_test  # numpy slices
gc.collect()
```
Should drop ~6-8 GB before the LSTM `.fit()` call. Verify with
`psutil.Process().memory_info().rss` printed before and after.

**Effort:** 15 minutes.

**Risk:** Low. The deleted frames are not used by anything downstream
in step 8.

---

## P3 — Production-ish polish (likely out of scope for this assignment)

### F11. Model versioning and metadata sidecar — ✅ CLOSED (Apr 8, Phase A)

Shipped in `src/models/metadata.py` and wired into `run_pipeline.py`
step 7/8 and `scripts/train_lstm_only.py`. Every saved model now gets
a `.metadata.json` sidecar recording `git_commit`, `git_dirty`,
`training_date_utc`, `n_train_rows`, `n_val_rows`, `n_test_rows`,
`val_metrics`, `feature_names_hash`, and `training_config`. Sidecars
are committed to git (while the model binaries themselves remain
gitignored) so reviewers can inspect training provenance without
running anything.

### F11-original. (historical)

**Files:** `src/models/xgboost_classifier.py:save`, `lstm_autoencoder.py:save`

**Problem:** Saved models contain only the weights + a few config
fields. There's no record of: which git commit produced them, which
data version they trained on, what the validation metrics were, what
the feature column order is. Hard to reproduce or audit.

**Fix plan:** Add a sidecar JSON next to each saved model:
`xgboost_failure.metadata.json` with `{git_commit, training_date,
n_train_rows, val_metrics, feature_names_hash, scale_pos_weight}`.

**Effort:** 1 hour.

**Risk:** None.

---

### F12. Multi-class XGBoost (predict failure_type)

**Files:** `src/models/xgboost_classifier.py`

**Problem:** Current model is binary `is_pre_failure`. A multi-class
model that predicts `failure_type` directly would let the optimizer
take type-specific actions (reduce frequency for thermal runaway,
flag for manual replacement on connector corrosion, etc.).

**Fix plan:** Add `MinerFailureClassifierMulticlass` that subclasses
the existing one but uses `XGBClassifier(objective="multi:softprob",
num_class=8)`. Map predictions back to the canonical `failure_type`
enum. Train alongside the binary model so we can compare.

**Effort:** 4-6 hours including evaluation tooling.

**Risk:** Medium. Multi-class with extreme imbalance per type is its
own modeling problem.

---

### F13. Drift detection on streaming features

**Files:** new `src/models/drift.py`, hooked into `cli/ai_bridge.py`

**Problem:** The model is trained on synthetic data with a specific
ambient temperature distribution and seasonal drift. Real ASICs in
real containers will have different distributions. We need to detect
when live feature distributions diverge from training so the operator
knows when retraining is due.

**Fix plan:** Compute KL divergence between rolling 1-day live
feature histograms and the training distribution (saved as a
`feature_histograms.json` sidecar at training time). Raise an alert
when divergence exceeds a threshold.

**Effort:** 4-8 hours.

**Risk:** Medium. KL divergence on sparse histograms is finicky.

---

### F14. Real MDK API integration

**Files:** new `src/integration/mdk_client.py`

**Problem:** Project plan says "MDK-compatible data schema: structure
synthetic data to match what MDK workers would produce". We've done
half the job — the schema matches — but there's no actual client
that fetches from a real MDK worker.

**Fix plan:** Per `docs/PROPOSAL.md`, wait until a data sharing
channel exists with Tether's mining team. When it does:
1. Implement `MDKClient` with the same `(timestamp, miner_id, ...)`
   row schema as the synthetic generator emits.
2. Add a config flag `data_source: synthetic | mdk_live`.
3. Run the same pipeline against real telemetry, retrain, compare
   distributions.

**Effort:** 1-2 days, gated on external dependency.

**Risk:** High variance — depends entirely on what the real telemetry
looks like.

---

## Implementation order recommendation

If I were picking from this list with a finite time budget:

1. **F1** (CLI live inference fix) — biggest mentor-visible bug,
   2-3 hours.
2. **F5** (per-failure-type breakdown) — 20 minutes, dramatically
   improves the results story.
3. **F4** (baseline comparison) — 1 hour, gives the AUC number a
   reference frame.
4. **F10** (memory cleanup before LSTM) — 15 minutes, prevents future
   OOMs in the main pipeline.
5. **F8** (warnings cleanup) — 20 minutes, cleans up the log noise.
6. **F11** (model versioning sidecar) — 1 hour, makes the project
   feel less hand-rolled.

Total: ~6 hours. Skips F2/F3/F6/F7/F9/F12/F13/F14, all of which are
either nice-to-have or out of scope for the assignment timeline.

### F15. Narrow remaining broad-except blocks in `src/cli/`

**Priority:** P2 (post-submission hardening)
**Effort:** 1-2 hours
**Files:** `src/cli/app.py` (~8 blocks: `_update_kpis`, `_update_alerts`, `_update_detail`, `_update_actions`, `_update_scenarios`, `on_button_pressed`, `_update_status_bar`, `_update_ai_text`; plus nested `try/except Exception` inside `on_button_pressed` around L636-639), `src/cli/ai_bridge.py` (~7 blocks, including the unlogged `except Exception: pass` around L605).

**What:** Narrow each `except Exception:` to the specific expected exception type (commonly `NoMatches`, `CellDoesNotExist`, `QueryError`, `DuckDBPyConnection` I/O errors). Any remaining catch-alls should at minimum log to a debug file.

**Why:** This is the same class of bug as Flaw 4 in `mdk test-cli` — a broad catch-all silently swallowed a `CellDoesNotExist` that masked the fleet table never updating. The 15 remaining blocks are latent copies of that pattern.

**Verification:** add a `scripts/test_cli_exception_narrowness.py` that monkey-patches each target method's inner call to raise an unexpected `AttributeError` and asserts propagation. Same pattern as `test_flaw4_fleet_update_raises_not_swallowed`.

### F16. Decouple DataTable column display labels from update-cell keys

**Priority:** P3 (nice-to-have)
**Effort:** 30-60 min
**Files:** `src/cli/app.py` (`_FLEET_COLUMNS` constant + `on_mount` and `_update_fleet_table`).

**What:** Replace the pattern

```python
for label in ("ID", "Mode", ...):
    table.add_column(label, key=label)
# and in _update_fleet_table:
table.update_cell(miner.miner_id, "Mode", miner.mode.value)
```

with stable short keys, e.g.

```python
_FLEET_COLUMNS = [("id", "ID"), ("mode", "Mode"), ...]
for key, label in _FLEET_COLUMNS:
    table.add_column(label, key=key)
# and:
table.update_cell(miner.miner_id, "mode", miner.mode.value)
```

**Why:** Today the column key IS the display label. Renaming "Temp (C)" → "Temp °C" silently breaks every `update_cell` call. The short-key variant decouples them — safe to rename display labels without breaking code.

**Verification:** unit test that programmatically renames one column label and asserts updates still work. Fold into the existing `test_cli_dashboard_flaws.py` harness.

### F17. Eliminate pandas `DataFrame is highly fragmented` PerformanceWarnings

**Priority:** P3 (performance polish)
**Effort:** 1-2 hours
**Files:** `src/pipeline/features.py:113` — the repeated `df[name] = col.values` inside the rolling/trend/correlation loops.

**What:** Refactor the feature builder to batch new columns via a `pd.concat([df, new_cols_df], axis=1)` at the end of each phase (ratios, rolling, trends, correlations, etc.) rather than assigning them one at a time. The warning fires ~25 times per `mdk check` run and obscures real warnings in logs.

**Why:** Performance warning is cosmetic but noisy. A `pd.concat`-based rewrite also reduces memory fragmentation on the 5.2M-row feature matrix build.

**Verification:** before → `mdk check 2>&1 | grep -c "highly fragmented"` returns some positive number. After → returns 0. Build time should not regress (measure with the same `mdk check` wall-clock; currently ~650s).
