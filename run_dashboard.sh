#!/bin/bash
# MDK Mining Fleet Dashboard
# Usage: ./run_dashboard.sh [--miners N] [--seed S]
cd "$(dirname "$0")"
python3 -m src.cli "$@"
