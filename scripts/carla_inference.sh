#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CARLA_ROOT:?Set CARLA_ROOT to the CARLA 0.9.15 installation}"
if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/carla_inference.sh CHECKPOINT OUTPUT_DIR [evaluator arguments]" >&2
    exit 2
fi
CHECKPOINT=$("${PYTHON:-python}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")
OUTPUT_DIR=$("${PYTHON:-python}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")
shift 2
cd "$ROOT"
test -f "$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"
export ALIGNDRIVE_ROOT="$ROOT"
export SCENARIO_RUNNER_ROOT="$ROOT/scenario_runner"
export LEADERBOARD_ROOT="$ROOT/leaderboard"
export IS_BENCH2DRIVE=1
export SAVE_PATH="$OUTPUT_DIR"
export ALIGNDRIVE_COLLISION_MODE="${ALIGNDRIVE_COLLISION_MODE:-fast}"
export PYTHONPATH="$ROOT:$ROOT/leaderboard:$ROOT/scenario_runner:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
shopt -s nullglob
CARLA_EGGS=("$CARLA_ROOT"/PythonAPI/carla/dist/carla-*-py3.*.egg)
if [[ -n "${CARLA_EGG:-}" ]]; then
    test -f "$CARLA_EGG"
    export PYTHONPATH="$CARLA_EGG:$PYTHONPATH"
elif [[ ${#CARLA_EGGS[@]} -eq 1 ]]; then
    export PYTHONPATH="${CARLA_EGGS[0]}:$PYTHONPATH"
elif [[ ${#CARLA_EGGS[@]} -gt 1 ]]; then
    echo "Multiple Python 3 CARLA eggs found; set CARLA_EGG explicitly." >&2
    exit 2
fi
"${PYTHON:-python}" -c 'import carla; assert hasattr(carla, "TrafficLightState"), "Set CARLA_EGG or install the matching CARLA Python API"'
exec "${PYTHON:-python}" -u leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes "$ROOT/leaderboard/route_data/bench2drive220_0.xml" \
    --routes-subset 1711 --repetitions 1 --track SENSORS \
    --checkpoint "$OUTPUT_DIR/routes.json" --debug-checkpoint "$OUTPUT_DIR/live_results.txt" \
    --agent "$ROOT/leaderboard/team_code/aligndrive_b2d_agent.py" \
    --agent-config "${ALIGNDRIVE_CONFIG:-$ROOT/projects_sp/configs/inference/aligndrive_inference.py}+$CHECKPOINT" \
    --save_name "$OUTPUT_DIR" --pid_param "$ROOT/leaderboard/pid/control_2m5hz.json" \
    --port "${CARLA_PORT:-2000}" --traffic-manager-port "${TM_PORT:-8000}" \
    --gpu-rank "${GPU_RANK:-0}" "$@"
