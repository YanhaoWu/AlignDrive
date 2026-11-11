#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export ALIGNDRIVE_ROOT="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec "${PYTHON:-python}" -u tools/test.py --config projects_sp/configs/inference/aligndrive_inference.py "$@"
