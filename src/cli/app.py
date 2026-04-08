"""
MDK Mining Fleet Dashboard — Live terminal UI built with Textual.
Run: python -m src.cli
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    RichLog,
    Rule,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.binding import Binding

from rich.text import Text

from .simulation import (
    AlertSeverity,
    FailureType,
    MinerState,
    MiningFleetSimulation,
    OperatingMode,
)
from ..synthetic.scenarios import list_scenarios, get_scenario, SCENARIOS


# ─── Metric display widget ───────────────────────────────────────────

class MetricCard(Static):
    """A single KPI metric card."""

    def __init__(self, label: str, value: str = "—", unit: str = "", card_id: str = ""):
        super().__init__(id=card_id)
        self._label = label
        self._value = value
        self._unit = unit

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="metric-label")
        yield Label(f"{self._value} {self._unit}", id=f"{self.id}-val", classes="metric-value")

    def update_value(self, value: str, unit: str = ""):
        try:
            self.query_one(f"#{self.id}-val", Label).update(f"{value} {unit}")
        except Exception:
            pass


# ─── Main Dashboard App ──────────────────────────────────────────────

class MiningDashboard(App):
    """MDK AI Mining Fleet Dashboard."""

    CSS = """
    Screen { background: $surface; }
    #main-container { height: 100%; }

    #kpi-bar {
        height: 5; dock: top; layout: horizontal;
        padding: 0 1; background: $boost;
    }
    .metric-label { text-align: center; color: $text-muted; text-style: dim; }
    .metric-value { text-align: center; text-style: bold; color: $text; }
    #content-area { height: 1fr; }
    DataTable { height: 1fr; }
    #alert-log { height: 1fr; border: round $warning; }
    #detail-panel { height: 1fr; padding: 1; }
    #actions-log { height: 1fr; border: round $success; }

    /* ── Scenarios tab ── */
    #scenarios-panel { height: 1fr; padding: 1; }
    #active-failures-table { height: auto; max-height: 12; }
    #inject-bar { layout: horizontal; height: 3; padding: 0 1; }
    #inject-bar Select { width: 1fr; }
    #inject-bar Button { width: 16; margin: 0 1; }
    #inject-status { height: 1; padding: 0 1; }
    #scenario-library { height: 1fr; padding: 0 1; }

    #status-bar {
        dock: bottom; height: 1; padding: 0 1;
        background: $primary; color: $text;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("f", "focus_fleet", "Fleet View"),
        Binding("a", "focus_alerts", "Alerts"),
        Binding("d", "focus_detail", "Detail"),
        Binding("o", "focus_actions", "Optimizer"),
        Binding("s", "focus_scenarios", "Scenarios"),
        Binding("plus", "speed_up", "Speed +"),
        Binding("minus", "speed_down", "Speed -"),
    ]

    TITLE = "MDK Mining Fleet Dashboard"
    SUB_TITLE = "AI-Driven Mining Optimization Controller"

    paused = reactive(False)
    sim_speed = reactive(1.0)
    selected_miner_id = reactive("MNR-001")

    def __init__(self, n_miners: int = 24, seed: int = 42):
        super().__init__()
        self.sim = MiningFleetSimulation(n_miners=n_miners, seed=seed)
        self._tick_timer: Timer | None = None
        self._alert_count = 0
        self._actions_shown = 0

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            with Horizontal(id="kpi-bar"):
                yield MetricCard("FLEET HASHRATE", "0", "PH/s", card_id="kpi-hashrate")
                yield MetricCard("FLEET POWER", "0", "MW", card_id="kpi-power")
                yield MetricCard("EFFICIENCY", "0", "J/TH", card_id="kpi-efficiency")
                yield MetricCard("TE HEALTH", "0", "", card_id="kpi-te")
                yield MetricCard("ENERGY PRICE", "0", "$/kWh", card_id="kpi-energy")
                yield MetricCard("FLEET STATUS", "0/0/0", "H/W/C", card_id="kpi-status")

            with TabbedContent(id="content-area"):
                # Tab 1: Fleet Overview
                with TabPane("Fleet Overview", id="tab-fleet"):
                    yield DataTable(id="fleet-table")

                # Tab 2: Alerts
                with TabPane("Alerts", id="tab-alerts"):
                    yield RichLog(id="alert-log", highlight=True, max_lines=200, markup=True)

                # Tab 3: Miner Detail
                with TabPane("Miner Detail", id="tab-detail"):
                    with VerticalScroll(id="detail-panel"):
                        yield Label("Select a miner from Fleet Overview (click a row)", id="detail-header")
                        yield Rule()
                        yield Static(id="detail-content")

                # Tab 4: Optimizer Actions
                with TabPane("Optimizer Actions", id="tab-actions"):
                    yield RichLog(id="actions-log", highlight=True, max_lines=200, markup=True)

                # Tab 5: Scenarios (NEW)
                with TabPane("Scenarios", id="tab-scenarios"):
                    with VerticalScroll(id="scenarios-panel"):
                        yield Label("[bold]Active Failures[/]", classes="metric-label")
                        yield DataTable(id="active-failures-table")
                        yield Rule()
                        yield Label("[bold]Inject Failure[/]", classes="metric-label")
                        with Horizontal(id="inject-bar"):
                            yield Select(
                                [(m.miner_id, m.miner_id) for m in self.sim.miners],
                                prompt="Select miner",
                                id="inject-miner-select",
                            )
                            yield Select(
                                [(name, name) for name in list_scenarios()],
                                prompt="Select scenario",
                                id="inject-scenario-select",
                            )
                            yield Button("Inject", id="inject-btn", variant="error")
                        yield Label("", id="inject-status")
                        yield Rule()
                        yield Label("[bold]Scenario Library[/]", classes="metric-label")
                        yield Static(id="scenario-library")

        yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        # Fleet table
        table = self.query_one("#fleet-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(
            "ID", "Model", "Container", "Mode",
            "Hashrate (TH/s)", "Power (W)", "Temp (C)",
            "J/TH", "TE Health", "Health", "Anomaly", "Status",
        )
        for miner in self.sim.miners:
            table.add_row(
                miner.miner_id, miner.spec.model[:16], miner.container_id,
                miner.mode.value, "—", "—", "—", "—", "—", "—", "—", "OK",
                key=miner.miner_id,
            )

        # Active failures table
        af_table = self.query_one("#active-failures-table", DataTable)
        af_table.zebra_stripes = True
        af_table.add_columns("Miner", "Scenario", "Progress", "AI Detected", "Score", "Status")

        # Scenario library (static, rendered once)
        lib = self.query_one("#scenario-library", Static)
        lines = []
        for name in list_scenarios():
            s = get_scenario(name)
            lines.append(f"  [bold cyan]{name}[/]")
            lines.append(f"  {s.description[:90]}")
            lines.append(f"  [dim]Detect: {s.detection_hint}[/]")
            lines.append("")
        lib.update("\n".join(lines))

        # Start sim
        self._tick_timer = self.set_interval(0.2, self._sim_tick)

    # ── Tick ──────────────────────────────────────────────────────

    def _sim_tick(self) -> None:
        if self.paused:
            return
        self.sim.tick()
        self._update_kpis()
        self._update_fleet_table()
        self._update_alerts()
        self._update_actions()
        self._update_detail()
        if self.sim.step % 5 == 0:
            self._update_scenarios()
        self._update_status_bar()

    # ── KPI bar ───────────────────────────────────────────────────

    def _update_kpis(self) -> None:
        sim = self.sim
        try:
            self.query_one("#kpi-hashrate", MetricCard).update_value(f"{sim.total_hashrate_th / 1000:.1f}", "PH/s")
            self.query_one("#kpi-power", MetricCard).update_value(f"{sim.total_power_w / 1e6:.2f}", "MW")
            self.query_one("#kpi-efficiency", MetricCard).update_value(f"{sim.fleet_efficiency_jth:.1f}", "J/TH")
            self.query_one("#kpi-te", MetricCard).update_value(f"{sim.fleet_te_health:.4f}", "")
            self.query_one("#kpi-energy", MetricCard).update_value(f"${sim.energy_price_usd:.3f}", "/kWh")
            self.query_one("#kpi-status", MetricCard).update_value(
                f"{sim.healthy_count}/{sim.warning_count}/{sim.critical_count}", "H/W/C")
        except Exception:
            pass

    # ── Fleet table ───────────────────────────────────────────────

    def _update_fleet_table(self) -> None:
        table = self.query_one("#fleet-table", DataTable)
        for miner in self.sim.miners:
            if miner.health_score > 0.8:
                health_str = Text(f"{miner.health_score:.0%}", style="green bold")
                status_str = Text("OK", style="green")
            elif miner.health_score > 0.4:
                health_str = Text(f"{miner.health_score:.0%}", style="yellow bold")
                status_str = Text("WARN", style="yellow bold")
            else:
                health_str = Text(f"{miner.health_score:.0%}", style="red bold")
                status_str = Text("CRIT", style="red bold")

            if miner.is_flagged:
                status_str = Text("MAINT", style="magenta bold")
            if miner.mode == OperatingMode.SHUTDOWN:
                status_str = Text("DOWN", style="red bold")

            if miner.anomaly_score > 0.7:
                anomaly_str = Text(f"{miner.anomaly_score:.2f}", style="red bold")
            elif miner.anomaly_score > 0.4:
                anomaly_str = Text(f"{miner.anomaly_score:.2f}", style="yellow")
            else:
                anomaly_str = Text(f"{miner.anomaly_score:.2f}", style="green")

            if miner.temperature_c > 90:
                temp_str = Text(f"{miner.temperature_c:.1f}", style="red bold")
            elif miner.temperature_c > 80:
                temp_str = Text(f"{miner.temperature_c:.1f}", style="yellow")
            else:
                temp_str = Text(f"{miner.temperature_c:.1f}", style="green")

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

    # ── Alerts ────────────────────────────────────────────────────

    def _update_alerts(self) -> None:
        log = self.query_one("#alert-log", RichLog)
        new_alerts = self.sim.alerts[self._alert_count:]
        self._alert_count = len(self.sim.alerts)

        for alert in new_alerts:
            h, m = self.sim.step // 60, self.sim.step % 60
            ts = f"[dim]{h:02d}:{m:02d}[/dim]"
            mid = f"[bold]{alert.miner_id}[/bold]"
            msg = alert.message

            if "THERMAL SHUTDOWN" in msg:
                temp = msg.split("SHUTDOWN")[1].strip().split("C")[0] + "C"
                log.write(Text.from_markup(
                    f"  {ts}  [bold red reverse] SHUTDOWN [/bold red reverse]  {mid}\n"
                    f"        Chip overheated to {temp} — powered off to prevent damage\n"
                    f"        [dim]Action: Inspect cooling before restart[/dim]\n"))

            elif "Temperature critical" in msg:
                temp = msg.split(":")[1].strip()
                log.write(Text.from_markup(
                    f"  {ts}  [bold red] TEMP [/bold red]  {mid}  at [red]{temp}[/red]\n"
                    f"        [dim]Nearing shutdown — frequency being reduced[/dim]\n"))

            elif "Temperature warning" in msg:
                temp = msg.split(":")[1].strip()
                log.write(Text.from_markup(
                    f"  {ts}  [yellow] TEMP [/yellow]  {mid}  at [yellow]{temp}[/yellow]  [dim]— monitoring[/dim]\n"))

            elif "Hashrate drop" in msg:
                pct = msg.split(":")[1].strip()
                log.write(Text.from_markup(
                    f"  {ts}  [bold yellow] HASH [/bold yellow]  {mid}  running at [yellow]{pct}[/yellow]\n"
                    f"        [dim]Below expected — check throttling or hardware[/dim]\n"))

            elif "CRITICAL" in msg and "AI Risk" in msg:
                mins = msg.split("sustained")[1].split("min")[0].strip() if "sustained" in msg else "?"
                log.write(Text.from_markup(
                    f"  {ts}  [bold red reverse] AI CRITICAL [/bold red reverse]  {mid}\n"
                    f"        Anomaly sustained [bold]{mins} min[/bold] — immediate inspection needed\n"
                    f"        [dim]AI detects degradation consistent with impending failure[/dim]\n"))

            elif "HIGH" in msg and "AI Risk" in msg:
                mins = msg.split("sustained")[1].split("min")[0].strip() if "sustained" in msg else "?"
                log.write(Text.from_markup(
                    f"  {ts}  [bold yellow] AI HIGH [/bold yellow]  {mid}  anomaly for [bold]{mins} min[/bold]\n"
                    f"        [dim]Schedule inspection within the next few hours[/dim]\n"))

            elif "ELEVATED" in msg and "AI Risk" in msg:
                mins = msg.split("(")[1].split("min")[0].strip() if "(" in msg else "?"
                log.write(Text.from_markup(
                    f"  {ts}  [yellow] AI WATCH [/yellow]  {mid}  [dim]anomaly {mins} min — monitoring[/dim]\n"))

            elif "Heuristic" in msg:
                log.write(Text.from_markup(f"  {ts}  [yellow] HEUR [/yellow]  {mid}  {msg}\n"))

            else:
                log.write(Text.from_markup(f"  {ts}  [dim] INFO [/dim]  {mid}  {msg}\n"))

    # ── Optimizer Actions ─────────────────────────────────────────

    def _update_actions(self) -> None:
        log = self.query_one("#actions-log", RichLog)
        new_actions = self.sim.actions[self._actions_shown:]
        self._actions_shown = len(self.sim.actions)

        for action in new_actions:
            h, m = self.sim.step // 60, self.sim.step % 60
            ts = f"[dim]{h:02d}:{m:02d}[/dim]"
            mid = f"[bold]{action.miner_id}[/bold]"

            if action.action == "FLAG_MAINTENANCE":
                score = action.reason.split("=")[-1].strip() if "=" in action.reason else "?"
                risk = self.sim.ai._risk_levels.get(action.miner_id, "?")
                log.write(Text.from_markup(
                    f"  {ts}  [bold magenta reverse] MAINT [/bold magenta reverse]  {mid}  flagged for maintenance\n"
                    f"        Risk: [bold]{risk}[/bold]  Score: {score}\n"
                    f"        [dim]Inspect before next shift[/dim]\n"))

            elif action.action == "REDUCE_FREQ":
                delta = action.old_value - action.new_value
                if "critical" in action.reason.lower():
                    tag = "[bold red reverse] COOL [/bold red reverse]"
                    note = "Aggressive throttle — thermal damage risk"
                else:
                    tag = "[red] COOL [/red]"
                    note = "Gentle throttle for thermal headroom"
                log.write(Text.from_markup(
                    f"  {ts}  {tag}  {mid}  [cyan]{action.old_value:.0f}[/cyan] -> "
                    f"[cyan]{action.new_value:.0f} MHz[/cyan]  [dim](-{delta:.0f})[/dim]\n"
                    f"        [dim]{note}[/dim]\n"))

            elif action.action == "BOOST_FREQ":
                delta = action.new_value - action.old_value
                price = action.reason.split("$")[-1].strip() if "$" in action.reason else "?"
                log.write(Text.from_markup(
                    f"  {ts}  [bold green reverse] EARN [/bold green reverse]  {mid}  "
                    f"[cyan]{action.old_value:.0f}[/cyan] -> [cyan]{action.new_value:.0f} MHz[/cyan]  "
                    f"[dim](+{delta:.0f})[/dim]\n"
                    f"        [dim]Energy ${price} — boosting for revenue[/dim]\n"))

            elif action.action == "THROTTLE_FREQ":
                delta = action.old_value - action.new_value
                price = action.reason.split("$")[-1].strip() if "$" in action.reason else "?"
                log.write(Text.from_markup(
                    f"  {ts}  [yellow reverse] SAVE [/yellow reverse]  {mid}  "
                    f"[cyan]{action.old_value:.0f}[/cyan] -> [cyan]{action.new_value:.0f} MHz[/cyan]  "
                    f"[dim](-{delta:.0f})[/dim]\n"
                    f"        [dim]Energy ${price} — cutting costs[/dim]\n"))

            else:
                log.write(Text.from_markup(
                    f"  {ts}  [dim] ACT [/dim]  {mid}  {action.reason}\n"))

    # ── Miner Detail (enhanced with AI score breakdown) ───────────

    def _update_detail(self) -> None:
        miner = next((m for m in self.sim.miners if m.miner_id == self.selected_miner_id), None)
        if not miner:
            return
        try:
            self.query_one("#detail-header", Label).update(
                f"  {miner.miner_id} — {miner.spec.model} | {miner.container_id} Pos {miner.position}")

            content = self.query_one("#detail-content", Static)
            L = []

            hc = "green" if miner.health_score > 0.8 else "yellow" if miner.health_score > 0.4 else "red"
            L.append(f"[bold]Operating Mode:[/] {miner.mode.value}")
            L.append(f"[bold]Health Score:[/]   [{hc}]{miner.health_score:.0%}[/]")
            L.append(f"[bold]Uptime:[/]         {miner.uptime_hours:.0f} hours")
            L.append("")

            L.append("[bold underline]Live Telemetry[/]")
            L.append(f"  Frequency:     {miner.frequency_mhz:.0f} MHz  (range: {miner.spec.freq_min_mhz}-{miner.spec.freq_max_mhz})")
            L.append(f"  Voltage:       {miner.voltage_v:.3f} V")
            tc = "red" if miner.temperature_c > 90 else "yellow" if miner.temperature_c > 80 else "green"
            L.append(f"  Temperature:   [{tc}]{miner.temperature_c:.1f} C[/]  (ambient: {miner.ambient_c:.1f} C)")
            L.append(f"  Hashrate:      {miner.hashrate_th:.1f} TH/s  (nameplate: {miner.spec.hashrate_nameplate_th})")
            L.append(f"  Power:         {miner.power_w:.0f} W")
            L.append("")

            L.append("[bold underline]KPIs[/]")
            L.append(f"  Efficiency:    {miner.efficiency_jth:.1f} J/TH" if miner.efficiency_jth < 1000 else "  Efficiency:    N/A")
            L.append(f"  TE Health:     {miner.te_health:.5f}")
            L.append(f"  Realization:   {miner.hashrate_realization:.1%}")
            L.append("")

            # AI Score Breakdown (XGBoost + LSTM + Risk Level)
            L.append("[bold underline]AI Predictions[/]")
            scores = self.sim.ai.get_detailed_scores(miner.miner_id)
            xgb = scores.get("xgb_score", 0)
            lstm = scores.get("lstm_score", 0)
            combined = scores.get("combined", 0)
            sustained = scores.get("sustained_minutes", 0)
            risk = scores.get("risk_level", "LOW")

            xc = "red" if xgb > 0.1 else "yellow" if xgb > 0.01 else "green"
            lc = "red" if lstm > 0.5 else "yellow" if lstm > 0.2 else "green"
            cc = "red" if combined > 0.1 else "yellow" if combined > 0.01 else "green"

            L.append(f"  XGBoost score:   [{xc}]{xgb:.4f}[/]")
            L.append(f"  LSTM score:      [{lc}]{lstm:.4f}[/]")
            L.append(f"  Combined score:  [{cc}]{combined:.4f}[/]")
            L.append(f"  Sustained:       {sustained} min above threshold")
            L.append("")

            # Risk level display with source
            ml_risk = scores.get("ml_risk", "LOW")
            health_risk = scores.get("health_risk", "LOW")
            risk_source = scores.get("risk_source", "")

            risk_colors = {"LOW": "green", "ELEVATED": "yellow", "HIGH": "red", "CRITICAL": "red bold"}
            risk_actions = {
                "LOW": "Normal operation — no action needed",
                "ELEVATED": "Worth watching — early signs of degradation",
                "HIGH": "Schedule inspection — confirmed degradation",
                "CRITICAL": "Immediate action — failure imminent or occurring",
            }
            rc = risk_colors.get(risk, "white")
            L.append(f"  Risk Level:      [{rc}]{risk}[/]")
            L.append(f"  Action:          {risk_actions.get(risk, '')}")
            L.append("")

            # Show what's driving the risk
            L.append(f"  [dim]Risk breakdown:[/]")
            mlc = risk_colors.get(ml_risk, "white")
            hrc = risk_colors.get(health_risk, "white")
            L.append(f"    AI models:     [{mlc}]{ml_risk}[/]  [dim](score {combined:.4f}, sustained {sustained} min)[/]")
            L.append(f"    Health state:   [{hrc}]{health_risk}[/]  [dim](health {miner.health_score:.0%})[/]")
            if risk_source == "health":
                L.append(f"    [dim]Driven by: direct health measurement (not model prediction)[/]")
            elif risk_source == "ai_model":
                L.append(f"    [dim]Driven by: AI model detecting pattern before health drops[/]")
            elif risk_source == "both":
                L.append(f"    [dim]Driven by: both AI model and health agree[/]")
            L.append(f"  Maint flagged:   {'[magenta]YES[/]' if miner.is_flagged else 'No'}")
            L.append("")

            # Top feature contributions
            contribs = self.sim.ai.get_feature_contributions(miner.miner_id, top_n=5)
            if contribs:
                L.append("[bold underline]Top Feature Contributions[/]")
                for feat, val in contribs:
                    short = feat.replace("_roll_", " ").replace("_", " ")
                    L.append(f"  {short:35s} {val:>10.2f}")
                L.append("")

            # Failure ground truth
            if miner._scenario_name:
                L.append("[bold underline]Scenario (Ground Truth)[/]")
                L.append(f"  Type:     [red]{miner._scenario_name}[/]")
                L.append(f"  Onset:    Step {miner._failure_onset_step}")
                L.append(f"  Duration: {miner._failure_duration} steps")
                L.append(f"  Progress: {miner.failure_progress:.0%}")

            content.update("\n".join(L))
        except Exception:
            pass

    # ── Scenarios tab ─────────────────────────────────────────────

    def _update_scenarios(self) -> None:
        """Update the active failures table."""
        try:
            table = self.query_one("#active-failures-table", DataTable)
            active = self.sim.get_active_failures()

            table.clear()
            for f in active:
                prog = f"{f['progress']:.0%}"
                detected = "[green]YES[/]" if f["ai_detected"] else "[red]NO[/]"
                score = f"{f.get('anomaly_score', 0):.4f}"
                status = f["status"]

                table.add_row(
                    f["miner_id"],
                    f["scenario"],
                    prog,
                    Text.from_markup(detected),
                    score,
                    status,
                )
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle inject button click."""
        if event.button.id == "inject-btn":
            try:
                miner_select = self.query_one("#inject-miner-select", Select)
                scenario_select = self.query_one("#inject-scenario-select", Select)
                status_label = self.query_one("#inject-status", Label)

                miner_id = miner_select.value
                scenario_name = scenario_select.value

                if miner_id is Select.BLANK or scenario_name is Select.BLANK:
                    status_label.update("[yellow]Select both a miner and a scenario first[/]")
                    return

                ok = self.sim.inject_scenario(str(miner_id), str(scenario_name))
                if ok:
                    status_label.update(
                        f"[green bold]Injected '{scenario_name}' into {miner_id} at step {self.sim.step}[/]")
                else:
                    status_label.update(f"[red]Failed to inject scenario[/]")
            except Exception as e:
                try:
                    self.query_one("#inject-status", Label).update(f"[red]Error: {e}[/]")
                except Exception:
                    pass

    # ── Status bar ────────────────────────────────────────────────

    def _update_status_bar(self) -> None:
        try:
            bar = self.query_one("#status-bar", Label)
            pause_str = "PAUSED" if self.paused else "RUNNING"
            ai_str = "AI:REAL" if self.sim._ai_ready else "AI:HEURISTIC"
            db_count = self.sim.ai.get_db_count()
            n_active = len(self.sim.get_active_failures())
            bar.update(
                f" {pause_str} | {ai_str} | Step: {self.sim.step} | "
                f"Sim: {self.sim.step / 60:.1f}h | "
                f"Failures: {n_active} | "
                f"DB: {db_count:,} | "
                f"SPACE=Pause  +/-=Speed  s=Scenarios  q=Quit")
        except Exception:
            pass

    # ── Events ────────────────────────────────────────────────────

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self.selected_miner_id = str(event.row_key.value)
            self._update_detail()
            tabs = self.query_one(TabbedContent)
            tabs.active = "tab-detail"

    # ── Key actions ───────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused

    def action_speed_up(self) -> None:
        if self._tick_timer:
            self._tick_timer.stop()
        self.sim_speed = min(10.0, self.sim_speed + 0.5)
        self._tick_timer = self.set_interval(max(0.05, 0.2 / self.sim_speed), self._sim_tick)

    def action_speed_down(self) -> None:
        if self._tick_timer:
            self._tick_timer.stop()
        self.sim_speed = max(0.5, self.sim_speed - 0.5)
        self._tick_timer = self.set_interval(0.2 / self.sim_speed, self._sim_tick)

    def action_focus_fleet(self) -> None:
        self.query_one(TabbedContent).active = "tab-fleet"

    def action_focus_alerts(self) -> None:
        self.query_one(TabbedContent).active = "tab-alerts"

    def action_focus_detail(self) -> None:
        self.query_one(TabbedContent).active = "tab-detail"

    def action_focus_actions(self) -> None:
        self.query_one(TabbedContent).active = "tab-actions"

    def action_focus_scenarios(self) -> None:
        self.query_one(TabbedContent).active = "tab-scenarios"
