#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/prepare_data.sh [BENCH2DRIVE_ROOT] [converter options]"
    echo "Creates data/Bench2Drive when given an external dataset, then generates metadata and waypoints."
    exit 0
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${ALIGNDRIVE_DATA_ROOT:-$ROOT/data/Bench2Drive}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    DATA_ROOT="$1"
    shift
fi
DATA_ROOT=$(cd "$DATA_ROOT" && pwd -P)
cd "$ROOT"
mkdir -p data
if [[ ! -e data/Bench2Drive && ! -L data/Bench2Drive ]]; then
    ln -s "$DATA_ROOT" data/Bench2Drive
elif [[ "$(cd data/Bench2Drive && pwd -P)" != "$DATA_ROOT" ]]; then
    echo "data/Bench2Drive already points to another dataset; update it explicitly before preprocessing" >&2
    exit 2
fi
export ALIGNDRIVE_ROOT="$ROOT"
export ALIGNDRIVE_DATA_ROOT="$DATA_ROOT"
export ALIGNDRIVE_MAP_ROOT="${ALIGNDRIVE_MAP_ROOT:-$DATA_ROOT/maps}"
export ALIGNDRIVE_ANNO_ROOT="${ALIGNDRIVE_ANNO_ROOT:-$ROOT/data/bench2drive_meta/infos_b2d_full}"
if [[ -z "${ALIGNDRIVE_SPLIT_FILE:-}" && -f "$DATA_ROOT/bench2drive_base_train_val_split.json" ]]; then
    export ALIGNDRIVE_SPLIT_FILE="$DATA_ROOT/bench2drive_base_train_val_split.json"
fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec "${PYTHON:-python3}" tools/data_converter/B2D_converter.py --split train val "$@"
