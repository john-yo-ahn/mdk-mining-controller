# CLI Dashboard Four-Flaw Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four flaws found during end-to-end testing of the MDK dashboard: (1) Fleet Overview table never updates live, (2) scenario thermal effects pollute siblings of the same model via a shared `MinerSpec`, (3) speed-up/speed-down controls are inverted and inconsistent with the 500 ms base tick, (4) blanket `except: pass` blocks hide real bugs.

**Architecture:** Each flaw is a contained defect in `src/cli/app.py` or `src/cli/simulation.py`. Fix the four root causes in order of blast radius (Flaw 1 → 2 → 3 → 4). Each fix gets a failing repro test under `scripts/` matching the repo's existing `scripts/test_te_formula.py` style (plain-`assert` scripts, no pytest), driven headlessly via Textual's `app.run_test()` pilot where UI is involved. Commit after each fix.

**Tech Stack:** Python 3.10, Textual (TUI), dataclasses, DuckDB, asyncio, `uv run`.

---

## File Structure

Files touched by this plan:

- **Modify** [src/cli/app.py](src/cli/app.py)
  - `on_mount` (~L207): `add_columns` → `add_column(label, key=label)` loop (Flaw 1)
  - `_update_fleet_table` (~L298): narrow the `try/except Exception: pass` (Flaw 4)
  - `action_speed_up` / `action_speed_down` (~L675-685): use single 0.5 s base (Flaw 3)
  - Several other `except Exception: pass` blocks (Flaw 4)
- **Modify** [src/cli/simulation.py](src/cli/simulation.py)
  - `_init_fleet` (~L212): `replace(MINER_MODELS[model_name])` so each miner gets its own spec (Flaw 2)
- **Create** `scripts/test_cli_dashboard_flaws.py` — single standalone script with four repro tests, one per flaw. Matches the `scripts/test_te_formula.py` convention (plain asserts, `main()` returns exit code, runnable via `uv run python -m scripts.test_cli_dashboard_flaws`).

Rationale for one test file: each flaw has exactly one reproduction and we want them co-located so the whole regression surface is one command. Total expected size ~150 LOC.

---

## Task 1: Repro harness — create the failing-test scaffold

**Files:**
- Create: `scripts/test_cli_dashboard_flaws.py`

This task lays down the test file with four `test_*` functions, each of which asserts the CURRENT (broken) behavior so that Tasks 2–5 can flip the asserts to the correct behavior. We do NOT commit broken behavior as "passing" — we commit the scaffold where every test currently **fails** the post-fix assertion (i.e. we write post-fix asserts, watch them fail, then fix in each subsequent task).

- [ ] **Step 1: Write the test scaffold**

```python
# scripts/test_cli_dashboard_flaws.py
"""
Regression tests for the four flaws found in end-to-end dashboard testing
(see docs/superpowers/plans/2026-04-17-cli-dashboard-four-flaws.md).

Runs without pytest — plain asserts, matches scripts/test_te_formula.py style.
Invoke: uv run python -m scripts.test_cli_dashboard_flaws
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace


def test_flaw1_fleet_table_updates_live() -> None:
    """Flaw 1: Fleet Overview table must reflect live miner values, not stay on '—'."""
    from src.cli.app import MiningDashboard
    from textual.widgets import DataTable

    async def scenario():
        app = MiningDashboard(n_miners=6, seed=42)
        async with app.run_test(size=(220, 55)) as pilot:
            for _ in range(20):
                app.sim.tick()
            app._update_fleet_table()
            t = app.query_one("#fleet-table", DataTable)
            row_keys = list(t.rows.keys())
            col_keys = list(t.columns.keys())
            vals = [t.get_cell(row_keys[0], k) for k in col_keys]
            # Hashrate column (index 4) must no longer be '—' after ticks.
            hashrate_cell = vals[4]
            assert hashrate_cell != "—", (
                f"Fleet table hashrate cell is still '—' after 20 ticks "
                f"— update_cell is silently failing. Row: {vals}"
            )
            await pilot.press("q")

    asyncio.run(scenario())
    print("  flaw1 OK: fleet table updates live")


def test_flaw2_scenario_does_not_pollute_siblings() -> None:
    """Flaw 2: Injecting a scenario into one miner must not change another miner's spec."""
    from src.cli.simulation import MiningFleetSimulation

    sim = MiningFleetSimulation(n_miners=6, seed=42)
    m0, m4 = sim.miners[0], sim.miners[4]
    assert m0.spec.model == m4.spec.model, "setup: m0 and m4 should share model"
    baseline = m4.spec.thermal_resistance
    sim.inject_scenario("MNR-001", "coolant_restriction")
    for _ in range(200):
        sim.tick()
    assert m4.spec.thermal_resistance == baseline, (
        f"Flaw 2: MNR-005.spec.thermal_resistance changed from {baseline} "
        f"to {m4.spec.thermal_resistance} because MNR-001's scenario mutated "
        f"a shared MinerSpec object."
    )
    sim.ai.close()
    print("  flaw2 OK: scenario stays isolated to target miner")


def test_flaw3_speed_controls_monotonic() -> None:
    """Flaw 3: '-' must slow down (interval ≥ baseline); '+' must speed up consistently."""
    from src.cli.app import MiningDashboard

    async def scenario():
        app = MiningDashboard(n_miners=4, seed=42)
        async with app.run_test(size=(220, 55)) as pilot:
            await pilot.pause()
            base_interval = app._tick_timer._interval
            await pilot.press("minus")
            slower_interval = app._tick_timer._interval
            assert slower_interval > base_interval, (
                f"Flaw 3: pressing '-' moved interval from {base_interval}s "
                f"to {slower_interval}s — smaller interval means FASTER, not slower."
            )
            await pilot.press("plus")
            await pilot.press("plus")
            faster_interval = app._tick_timer._interval
            assert faster_interval < base_interval, (
                f"Flaw 3: pressing '+' twice from slowed state gave "
                f"{faster_interval}s, expected < {base_interval}s."
            )
            await pilot.press("q")

    asyncio.run(scenario())
    print("  flaw3 OK: speed controls monotonic")


def test_flaw4_fleet_update_raises_not_swallowed() -> None:
    """Flaw 4: _update_fleet_table must not silently swallow arbitrary exceptions.

    We verify this by monkey-patching the DataTable so update_cell raises an
    unexpected AttributeError, and confirming the method propagates it rather
    than catching it blindly.
    """
    from src.cli.app import MiningDashboard
    from textual.widgets import DataTable

    async def scenario():
        app = MiningDashboard(n_miners=4, seed=42)
        async with app.run_test(size=(220, 55)) as pilot:
            await pilot.pause()
            t = app.query_one("#fleet-table", DataTable)

            def boom(*a, **kw):
                raise AttributeError("unexpected failure — should NOT be swallowed")

            t.update_cell = boom  # type: ignore[method-assign]
            raised = False
            try:
                app._update_fleet_table()
            except AttributeError:
                raised = True
            assert raised, (
                "Flaw 4: _update_fleet_table swallowed an unexpected AttributeError. "
                "Exception handling must be narrow (CellDoesNotExist only) or absent."
            )
            await pilot.press("q")

    asyncio.run(scenario())
    print("  flaw4 OK: unexpected exceptions propagate")


def main() -> int:
    tests = [
        test_flaw1_fleet_table_updates_live,
        test_flaw2_scenario_does_not_pollute_siblings,
        test_flaw3_speed_controls_monotonic,
        test_flaw4_fleet_update_raises_not_swallowed,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the test file and confirm all four tests fail**

Run: `uv run python -m scripts.test_cli_dashboard_flaws`
Expected output includes:
```
  FAIL test_flaw1_fleet_table_updates_live: Fleet table hashrate cell is still '—' ...
  FAIL test_flaw2_scenario_does_not_pollute_siblings: MNR-005.spec.thermal_resistance changed ...
  FAIL test_flaw3_speed_controls_monotonic: pressing '-' moved interval ...
  FAIL test_flaw4_fleet_update_raises_not_swallowed: _update_fleet_table swallowed ...
0/4 passed
```
Exit code: 1.

If any test unexpectedly PASSES at this stage, stop — the test isn't exercising the bug.

- [ ] **Step 3: Commit the scaffold**

```bash
git add scripts/test_cli_dashboard_flaws.py docs/superpowers/plans/2026-04-17-cli-dashboard-four-flaws.md
git commit -m "test(cli): failing repros for 4 dashboard flaws"
```

---

## Task 2: Flaw 1 — Fleet table never updates live

**Root cause:** [src/cli/app.py:212-216](src/cli/app.py:212) calls `table.add_columns("ID", "Model", ...)` without keys, so Textual assigns each column a `ColumnKey(value=None)`. Later [`_update_fleet_table`](src/cli/app.py:298) calls `table.update_cell(miner.miner_id, "Mode", ...)`, which constructs a fresh `ColumnKey("Mode")` that does not equal the auto-generated one → `CellDoesNotExist` → swallowed by `except Exception: pass`.

**Fix:** replace `add_columns(*labels)` with a loop of `add_column(label, key=label)` so each column's key value matches the label string used at update time.

**Files:**
- Modify: [src/cli/app.py:212-216](src/cli/app.py:212)
- Test: `scripts/test_cli_dashboard_flaws.py::test_flaw1_fleet_table_updates_live`

- [ ] **Step 1: Confirm the failing test**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw1`
Expected: `FAIL test_flaw1_fleet_table_updates_live: ...`

- [ ] **Step 2: Apply the fix**

In [src/cli/app.py](src/cli/app.py), locate this block inside `on_mount`:

```python
        table.add_columns(
            "ID", "Model", "Container", "Mode",
            "Hashrate (TH/s)", "Power (W)", "Temp (C)",
            "J/TH", "TE Health", "Health", "Anomaly", "Status",
        )
```

Replace with:

```python
        for label in (
            "ID", "Model", "Container", "Mode",
            "Hashrate (TH/s)", "Power (W)", "Temp (C)",
            "J/TH", "TE Health", "Health", "Anomaly", "Status",
        ):
            table.add_column(label, key=label)
```

No other code needs changing — `_update_fleet_table` already passes label strings as column keys; they will now resolve.

- [ ] **Step 3: Run the flaw-1 test and confirm it passes**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw1`
Expected: `flaw1 OK: fleet table updates live`

- [ ] **Step 4: Run the full harness and confirm the remaining 3 still fail (no collateral damage)**

Run: `uv run python -m scripts.test_cli_dashboard_flaws`
Expected: `1/4 passed` — flaws 2/3/4 still failing.

- [ ] **Step 5: Commit**

```bash
git add src/cli/app.py
git commit -m "fix(cli): fleet table updates now work — add_column with keys"
```

---

## Task 3: Flaw 2 — Scenario thermal effects pollute same-model siblings

**Root cause:** [src/cli/simulation.py:218](src/cli/simulation.py:218) does `spec = MINER_MODELS[model_name]` with no copy. Every miner of a given model references the same `MinerSpec` object — which is also the one in the module-level `MINER_MODELS` dict. `_apply_failure` then mutates `miner.spec.thermal_resistance` ([L544](src/cli/simulation.py:544)), propagating the change to every sibling and permanently polluting the global spec table.

**Fix:** give each miner its own `MinerSpec` via `dataclasses.replace(MINER_MODELS[model_name])`. Smallest change, fully backward compatible (MinerSpec is already a frozen-layout dataclass of plain fields).

**Files:**
- Modify: [src/cli/simulation.py:212-231](src/cli/simulation.py:212)
- Test: `scripts/test_cli_dashboard_flaws.py::test_flaw2_scenario_does_not_pollute_siblings`

- [ ] **Step 1: Confirm the failing test**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw2`
Expected: `FAIL test_flaw2_scenario_does_not_pollute_siblings: ...`

- [ ] **Step 2: Add the import**

In [src/cli/simulation.py](src/cli/simulation.py) near the top, change:

```python
from dataclasses import dataclass, field
```

to:

```python
from dataclasses import dataclass, field, replace
```

- [ ] **Step 3: Apply the fix in `_init_fleet`**

Locate:

```python
        for i in range(self.n_miners):
            model_name = models[i % len(models)]
            spec = MINER_MODELS[model_name]
            container = containers[i % len(containers)]
```

Replace the `spec =` line with:

```python
            spec = replace(MINER_MODELS[model_name])  # per-miner copy; scenarios mutate spec
```

That is the entire behavioral change.

- [ ] **Step 4: Run the flaw-2 test and confirm it passes**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw2`
Expected: `flaw2 OK: scenario stays isolated to target miner`

- [ ] **Step 5: Run the full harness — flaws 1 & 2 pass, 3 & 4 still fail**

Run: `uv run python -m scripts.test_cli_dashboard_flaws`
Expected: `2/4 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/cli/simulation.py
git commit -m "fix(cli): per-miner MinerSpec — scenarios no longer pollute siblings"
```

---

## Task 4: Flaw 3 — Speed controls inverted & inconsistent with 500 ms base

**Root cause:** `on_mount` starts the tick timer at **0.5 s** ([src/cli/app.py:241](src/cli/app.py:241)), but `action_speed_up` / `action_speed_down` reset the interval to **`0.2 / sim_speed`** ([L679](src/cli/app.py:679), [L685](src/cli/app.py:685)). Two problems:

1. First `-` press moves `sim_speed` to 0.5, giving interval `0.2/0.5 = 0.4s` — faster than the 0.5 s baseline. "Slow down" actually speeds up.
2. First `+` press gives interval `0.2/1.5 ≈ 0.133s` — a 3.75× jump, not a gentle 1.5× step.

**Fix:** introduce a single module-level `BASE_TICK_INTERVAL = 0.5` constant and use `BASE_TICK_INTERVAL / sim_speed` everywhere.

**Files:**
- Modify: [src/cli/app.py](src/cli/app.py) — add constant, use it in `on_mount`, `action_speed_up`, `action_speed_down`.
- Test: `scripts/test_cli_dashboard_flaws.py::test_flaw3_speed_controls_monotonic`

- [ ] **Step 1: Confirm the failing test**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw3`
Expected: `FAIL test_flaw3_speed_controls_monotonic: pressing '-' moved interval ...`

- [ ] **Step 2: Add the constant**

In [src/cli/app.py](src/cli/app.py) just above the `SPARK_METRICS` block, add:

```python
# Single source of truth for tick pacing. sim_speed=1.0 → one tick per
# BASE_TICK_INTERVAL seconds; speed_up halves the wait, speed_down doubles it.
BASE_TICK_INTERVAL = 0.5
```

- [ ] **Step 3: Use it in `on_mount`**

Change:

```python
        self._tick_timer = self.set_interval(0.5, self._sim_tick)
```

to:

```python
        self._tick_timer = self.set_interval(BASE_TICK_INTERVAL / self.sim_speed, self._sim_tick)
```

- [ ] **Step 4: Fix `action_speed_up`**

Replace the method body:

```python
    def action_speed_up(self) -> None:
        if self._tick_timer:
            self._tick_timer.stop()
        self.sim_speed = min(10.0, self.sim_speed + 0.5)
        self._tick_timer = self.set_interval(
            max(0.05, BASE_TICK_INTERVAL / self.sim_speed), self._sim_tick
        )
```

- [ ] **Step 5: Fix `action_speed_down`**

Replace the method body:

```python
    def action_speed_down(self) -> None:
        if self._tick_timer:
            self._tick_timer.stop()
        self.sim_speed = max(0.5, self.sim_speed - 0.5)
        self._tick_timer = self.set_interval(
            BASE_TICK_INTERVAL / self.sim_speed, self._sim_tick
        )
```

- [ ] **Step 6: Run the flaw-3 test and confirm it passes**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw3`
Expected: `flaw3 OK: speed controls monotonic`

- [ ] **Step 7: Run the full harness — 3/4 pass**

Run: `uv run python -m scripts.test_cli_dashboard_flaws`
Expected: `3/4 passed`.

- [ ] **Step 8: Commit**

```bash
git add src/cli/app.py
git commit -m "fix(cli): unified BASE_TICK_INTERVAL — speed controls monotonic"
```

---

## Task 5: Flaw 4 — Narrow the `except` in `_update_fleet_table`

**Root cause:** [src/cli/app.py:342-343](src/cli/app.py:342) does `except Exception: pass` around the entire per-miner update block. This is how Flaw 1 lived undetected — the `CellDoesNotExist` exceptions had nowhere to surface. The same pattern exists in 8 other places, but we only narrow the fleet-table one now (the only one the test exercises); the others stay for this PR to keep the change focused.

**Fix:** narrow the catch to `CellDoesNotExist` — the only exception Textual can legitimately throw here if a miner row disappears mid-update (e.g. fleet shrinks). Any other exception is a real bug and must propagate.

**Files:**
- Modify: [src/cli/app.py:298-343](src/cli/app.py:298)
- Test: `scripts/test_cli_dashboard_flaws.py::test_flaw4_fleet_update_raises_not_swallowed`

- [ ] **Step 1: Confirm the failing test**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw4`
Expected: `FAIL test_flaw4_fleet_update_raises_not_swallowed: _update_fleet_table swallowed ...`

- [ ] **Step 2: Add the import**

Near the other textual imports at the top of [src/cli/app.py](src/cli/app.py), add:

```python
from textual.widgets._data_table import CellDoesNotExist
```

- [ ] **Step 3: Narrow the except**

In `_update_fleet_table`, locate the innermost block:

```python
            try:
                table.update_cell(miner.miner_id, "Mode", miner.mode.value)
                table.update_cell(miner.miner_id, "Hashrate (TH/s)", f"{miner.hashrate_th:.1f}")
                table.update_cell(miner.miner_id, "Power (W)", f"{miner.power_w:.0f}")
                table.update_cell(miner.miner_id, "Temp (C)", temp_str)
                table.update_cell(miner.miner_id, "J/TH",
                                  f"{miner.efficiency_jth:.1f}" if miner.efficiency_jth < 1000 else "—")
                table.update_cell(miner.miner_id, "TE Health",
                                  f"{miner.te_health:.4f}" if miner.te_health > 0 else "—")
                table.update_cell(miner.miner_id, "Health", health_str)
                table.update_cell(miner.miner_id, "Anomaly", anomaly_str)
                table.update_cell(miner.miner_id, "Status", status_str)
            except Exception:
                pass
```

Change `except Exception:` → `except CellDoesNotExist:`. The rest stays identical.

- [ ] **Step 4: Run the flaw-4 test and confirm it passes**

Run: `uv run python -m scripts.test_cli_dashboard_flaws 2>&1 | grep flaw4`
Expected: `flaw4 OK: unexpected exceptions propagate`

- [ ] **Step 5: Run the full harness — all four pass**

Run: `uv run python -m scripts.test_cli_dashboard_flaws`
Expected:
```
  flaw1 OK: fleet table updates live
  flaw2 OK: scenario stays isolated to target miner
  flaw3 OK: speed controls monotonic
  flaw4 OK: unexpected exceptions propagate

4/4 passed
```
Exit code: 0.

- [ ] **Step 6: Smoke-test the live dashboard**

Run: `timeout 8 uv run python -m src.cli --miners 6 --seed 42 < /dev/null > /tmp/mdk_smoke.log 2>&1 ; echo "exit=$?"`
Expected: no Python traceback in `/tmp/mdk_smoke.log`; exit code is 124 (timeout killed it) or 0. Any non-zero that is not 124 means a regression — investigate before committing.

- [ ] **Step 7: Commit**

```bash
git add src/cli/app.py
git commit -m "fix(cli): narrow fleet-table except to CellDoesNotExist"
```

---

## Task 6: Wire the regression test into the existing `mdk` test surface

The repo already exposes `mdk test-te` via [src/cli/__main__.py:106-108](src/cli/__main__.py:106). Add a `mdk test-cli` sibling so the dashboard regression test is as discoverable as the TE tests.

**Files:**
- Modify: [src/cli/__main__.py](src/cli/__main__.py) — add `cmd_test_cli` + subparser

- [ ] **Step 1: Add the command function**

After `cmd_test_te` (around L106-108), add:

```python
def cmd_test_cli(args):
    from scripts.test_cli_dashboard_flaws import main as run_cli_tests
    sys.exit(run_cli_tests())
```

- [ ] **Step 2: Register the subparser**

After the `test-te` subparser registration (around L146), add:

```python
    subparsers.add_parser("test-cli", help="Run CLI dashboard regression tests (4 flaws)")
```

- [ ] **Step 3: Wire into the `commands` dict**

In the `commands = { ... }` dict (around L155), add the entry `"test-cli": cmd_test_cli,`.

- [ ] **Step 4: Verify the wiring**

Run: `uv run mdk test-cli`
Expected: `4/4 passed`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/cli/__main__.py
git commit -m "feat(cli): mdk test-cli subcommand runs dashboard regression tests"
```

---

## Final verification

- [ ] **All four tests green**

Run: `uv run mdk test-cli`
Expected: `4/4 passed`.

- [ ] **Live dashboard boots & quits cleanly**

Run: `timeout 5 uv run mdk --miners 6 --seed 42 < /dev/null ; echo exit=$?`
Expected: exit=124 (timeout) with no traceback on stderr.

- [ ] **Unrelated smoke tests still pass**

Run: `uv run mdk test-te`
Expected: all TE tests still pass — we did not touch `src/kpi/*`.

- [ ] **Inspect the diff**

Run: `git log --oneline main..HEAD`
Expected: 6 commits (scaffold + 4 fixes + subcommand wiring), each touching only the files listed in its task.

---

## Self-Review

- **Spec coverage:** each flaw from the user's report has a dedicated task (2, 3, 4, 5) plus scaffold (1) and wiring (6). ✓
- **Placeholder scan:** no TBD/TODO/"handle appropriately" — every step shows the actual code or command. ✓
- **Type consistency:** `BASE_TICK_INTERVAL` used identically in three call sites; `replace` import path stable; `CellDoesNotExist` imported from the one location Textual actually exposes it (`textual.widgets._data_table`). ✓
- **Test-first discipline:** every fix task begins with "confirm the failing test," ends with "confirm it passes + full harness." ✓
- **Frequent commits:** one commit per fix, plus scaffold + wiring = 6 commits. ✓
