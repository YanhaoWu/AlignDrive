#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python3}" "$ROOT/tools/kmeans_b2d/generate_anchors.py" "$@"
