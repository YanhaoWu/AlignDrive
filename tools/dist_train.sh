#!/usr/bin/env bash
# Single-node distributed train launcher.
# Usage: bash tools/dist_train.sh <config> <gpus> [extra train.py args...]

set -euo pipefail
CONFIG=${1:-}
GPUS=${2:-}
PORT=${MASTER_PORT:-${PORT:-29500}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
if [ "$NNODES" -gt 1 ]; then
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export B2D_MP_START=${B2D_MP_START:-fork}

if [ -z "${CONFIG}" ] || [ -z "${GPUS}" ]; then
  echo "Usage: bash tools/dist_train.sh <config> <gpus> [extra args...]"
  exit 1
fi

PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-} \
"${PYTHON:-python3}" -m torch.distributed.launch --nproc_per_node="$GPUS" \
    --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$PORT" \
    "$(dirname "$0")/train.py" --config "$CONFIG" --launcher pytorch "${@:3}"
