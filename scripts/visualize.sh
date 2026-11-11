#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/visualize.sh SAVED_SCENE [simulation rendering options]"
    echo "Renders a multimodal video into work_dirs/visualization by default."
    if [[ $# -eq 0 ]]; then exit 2; else exit 0; fi
fi
SCENE=$(cd "$1" && pwd -P)
shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec "${PYTHON:-python3}" tools/visualization/visualize.py simulation \
    --scene-dir "$SCENE" --output-dir work_dirs/visualization "$@"
