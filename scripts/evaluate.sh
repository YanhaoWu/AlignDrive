#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/evaluate.sh --checkpoint CHECKPOINT [evaluation options]"
    echo "Runs Bench2Drive's 220 routes with the default AlignDrive inference configuration."
    if [[ $# -eq 0 ]]; then exit 2; else exit 0; fi
fi
case "$1" in
    --checkpoint)
        if [[ $# -lt 2 || "$2" == -* ]]; then
            echo "--checkpoint requires a file path" >&2
            exit 2
        fi
        CHECKPOINT="$2"
        shift 2
        ;;
    --checkpoint=*) CHECKPOINT="${1#--checkpoint=}"; shift ;;
    *) CHECKPOINT="$1"; shift ;;
esac
if [[ -z "$CHECKPOINT" ]]; then
    echo "--checkpoint requires a file path" >&2
    exit 2
fi
CHECKPOINT=$("${PYTHON:-python3}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$CHECKPOINT")
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
: "${CARLA_ROOT:?Set CARLA_ROOT as described in doc/INSTALL.md}"
if [[ ! -d "$CARLA_ROOT" ]]; then
    echo "CARLA_ROOT is not a directory: $CARLA_ROOT" >&2
    exit 2
fi
export CARLA_ROOT=$(cd "$CARLA_ROOT" && pwd -P)
GPUS=${GPUS:-2}
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer" >&2
    exit 2
fi
GPU_IDS=()
for ((GPU_INDEX=0; GPU_INDEX<GPUS; GPU_INDEX++)); do
    GPU_IDS+=("$GPU_INDEX")
done
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec "${PYTHON:-python3}" leaderboard/scripts/launch_evaluation.py \
    --checkpoint "$CHECKPOINT" --num_gpu "$GPUS" --used_gpu "${GPU_IDS[@]}" \
    --no-auto-restart "$@"
