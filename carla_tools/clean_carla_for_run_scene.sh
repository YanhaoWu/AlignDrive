#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -ne 1 || -z "$1" ]]; then
    echo "Usage: bash carla_tools/clean_carla_for_run_scene.sh SCREEN_SESSION" >&2
    exit 2
fi
exec "${PYTHON:-python3}" "$ROOT/leaderboard/scripts/process_kill.py" --session "$1"
