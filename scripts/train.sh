#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/train.sh [perception|spatial|lon] [--checkpoint PATH] [train.py options]"
    echo "--checkpoint: ResNet50 backbone for perception; previous-stage model for spatial/lon."
    echo "Defaults: stage=lon, GPUS=2. Run each stage separately."
    exit 0
fi
CALLER_DIR="$PWD"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export ALIGNDRIVE_ROOT="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
STAGE=lon
if [[ $# -gt 0 && "$1" != -* ]]; then
    STAGE="$1"
    shift
fi
case "$STAGE" in
    perception|spatial|lon) ;;
    *) echo "Unknown training stage: $STAGE" >&2; exit 2 ;;
esac
CHECKPOINT=""
CHECKPOINT_OPTION=false
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "--checkpoint requires a file path" >&2
                exit 2
            fi
            CHECKPOINT="$2"
            CHECKPOINT_OPTION=true
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT="${1#--checkpoint=}"
            CHECKPOINT_OPTION=true
            shift
            ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done
set -- ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
if [[ "$CHECKPOINT_OPTION" == true ]]; then
    if [[ -z "$CHECKPOINT" ]]; then
        echo "--checkpoint requires a file path" >&2
        exit 2
    fi
    CHECKPOINT=$("${PYTHON:-python3}" -c 'import os,sys; print(os.path.abspath(os.path.join(sys.argv[2], os.path.expanduser(sys.argv[1]))))' "$CHECKPOINT" "$CALLER_DIR")
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Checkpoint not found: $CHECKPOINT" >&2
        exit 2
    fi
fi
GPUS=${GPUS:-2}
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer" >&2
    exit 2
fi
RESUMING=false
for OPTION in "$@"; do
    case "$OPTION" in --resume-from|--resume-from=*) RESUMING=true ;; esac
done
export ALIGNDRIVE_RESNET_CHECKPOINT="${ALIGNDRIVE_RESNET_CHECKPOINT:-$ROOT/ckpt/resnet50-19c8e357.pth}"
export ALIGNDRIVE_PERCEPTION_CHECKPOINT="${ALIGNDRIVE_PERCEPTION_CHECKPOINT:-$ROOT/work_dirs/perception/last.pth}"
export ALIGNDRIVE_SPATIAL_CHECKPOINT="${ALIGNDRIVE_SPATIAL_CHECKPOINT:-$ROOT/work_dirs/spatial/last.pth}"
if [[ "$CHECKPOINT_OPTION" == true ]]; then
    case "$STAGE" in
        perception) export ALIGNDRIVE_RESNET_CHECKPOINT="$CHECKPOINT" ;;
        spatial) export ALIGNDRIVE_PERCEPTION_CHECKPOINT="$CHECKPOINT" ;;
        lon) export ALIGNDRIVE_SPATIAL_CHECKPOINT="$CHECKPOINT" ;;
    esac
fi
if [[ ! -f "$ALIGNDRIVE_RESNET_CHECKPOINT" ]]; then
    echo "Missing ResNet50 checkpoint: $ALIGNDRIVE_RESNET_CHECKPOINT (see doc/INSTALL.md)" >&2
    exit 2
fi
WORK_DIR="$ROOT/work_dirs/$STAGE"
INITIALIZATION=""
case "$STAGE" in
    spatial) INITIALIZATION="$ALIGNDRIVE_PERCEPTION_CHECKPOINT" ;;
    lon)
        INITIALIZATION="$ALIGNDRIVE_SPATIAL_CHECKPOINT"
        WORK_DIR="$ROOT/work_dirs/aligndrive"
        ;;
esac
if [[ "$RESUMING" == false && -n "$INITIALIZATION" && ! -f "$INITIALIZATION" ]]; then
    echo "Missing $STAGE initialization checkpoint: $INITIALIZATION" >&2
    exit 2
fi
exec bash tools/dist_train.sh "projects_sp/configs/train/aligndrive_stage_${STAGE}.py" "$GPUS" \
    --work-dir "$WORK_DIR" "$@"
