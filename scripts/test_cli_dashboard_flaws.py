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
    """Flaw 3: '-' must slow down (interval >= baseline); '+' must speed up consistently."""
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
