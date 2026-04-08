"""
Entry point for the MDK Mining Controller.

Usage:
    python -m src.cli                         # Dashboard (24 miners)
    python -m src.cli --miners 50             # Custom fleet size
    python -m src.cli scenarios               # List all failure scenarios
    python -m src.cli scenarios --detail       # Show scenario details
    python -m src.cli train                    # Run full training pipeline
"""

import argparse
import sys


def cmd_dashboard(args):
    from .app import MiningDashboard
    app = MiningDashboard(n_miners=args.miners, seed=args.seed)
    app.run()


def cmd_scenarios(args):
    from ..synthetic.scenarios import list_scenarios, get_scenario, create_custom_scenario

    if args.add:
        # Interactive scenario creation
        print("Create a custom failure scenario")
        print("=" * 50)
        name = input("Name (e.g., 'my_thermal_issue'): ").strip()
        desc = input("Description: ").strip()
        hint = input("Detection hint (what would an operator notice?): ").strip()

        effects = []
        print("\nAdd signal effects (empty name to finish):")
        print("  Signals:  hashrate, temperature, power, voltage, thermal_resistance, degradation_factor")
        print("  Modes:    scale, offset, noise")
        print("  Curves:   linear, exponential, step, intermittent, sine")

        while True:
            signal = input("\n  Signal name (or Enter to finish): ").strip()
            if not signal:
                break
            mode = input("  Mode [scale]: ").strip() or "scale"
            curve = input("  Curve [linear]: ").strip() or "linear"
            magnitude = float(input("  Magnitude (e.g., -0.3 for 30% drop, 10 for +10C): "))
            effects.append({"signal": signal, "mode": mode, "curve": curve, "magnitude": magnitude})

        if effects:
            scenario = create_custom_scenario(name, desc, effects, detection_hint=hint)
            print(f"\nCreated scenario '{name}' with {len(effects)} effects")
        else:
            print("No effects added. Scenario not created.")
        return

    # List scenarios
    scenarios = list_scenarios()
    print(f"\nFailure Scenarios ({len(scenarios)} available)")
    print("=" * 70)

    for name in scenarios:
        s = get_scenario(name)
        print(f"\n  {name}")
        if args.detail:
            print(f"  Description: {s.description}")
            print(f"  Duration: {s.duration_range[0]:,}-{s.duration_range[1]:,} steps "
                  f"({s.duration_range[0]//60}h-{s.duration_range[1]//60}h)")
            print(f"  Detection: {s.detection_hint}")
            print(f"  Effects:")
            for e in s.effects:
                print(f"    - {e.signal}: {e.mode} / {e.curve} / magnitude={e.magnitude}")
        else:
            print(f"  {s.description[:75]}...")
            print(f"  Detect: {s.detection_hint}")

    if not args.detail:
        print(f"\n  Use --detail for full scenario specs")


def cmd_train(args):
    from ..run_pipeline import main as run_pipeline
    run_pipeline()


def cmd_validate(args):
    from ..validate import main as run_validate
    tests = []
    if args.test:
        tests = [args.test]
    else:
        tests = ["holdout", "race", "blind", "noise"]
    run_validate(tests)


def main():
    parser = argparse.ArgumentParser(
        description="MDK Mining Controller — AI-Driven Mining Optimization"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default: dashboard
    parser.add_argument("--miners", "-n", type=int, default=24)
    parser.add_argument("--seed", "-s", type=int, default=42)

    # Subcommand: scenarios
    sc_parser = subparsers.add_parser("scenarios", help="List/add failure scenarios")
    sc_parser.add_argument("--detail", action="store_true", help="Show full details")
    sc_parser.add_argument("--add", action="store_true", help="Create a custom scenario")

    # Subcommand: train
    subparsers.add_parser("train", help="Run training pipeline")

    # Subcommand: validate
    val_parser = subparsers.add_parser("validate", help="Run model validation tests")
    val_parser.add_argument("--test", "-t", choices=["holdout", "race", "blind", "noise"],
                            help="Run a specific test (default: all)")

    args = parser.parse_args()

    if args.command == "scenarios":
        cmd_scenarios(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        cmd_dashboard(args)


if __name__ == "__main__":
    main()
