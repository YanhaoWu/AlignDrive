#!/bin/bash
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
# Must set CARLA_ROOT
: "${CARLA_ROOT:?Set CARLA_ROOT to CARLA 0.9.15}"
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
if [ -n "${CARLA_EGG:-}" ]; then export PYTHONPATH="$CARLA_EGG:$PYTHONPATH"; fi
export PYTHONPATH="$ROOT:$PYTHONPATH:$ROOT/leaderboard:$ROOT/leaderboard/team_code:$ROOT/scenario_runner"
export SCENARIO_RUNNER_ROOT="$ROOT/scenario_runner"

export LEADERBOARD_ROOT="$ROOT/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}
export PID_PARAM=${11}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8


echo "CUDA_VISIBLE_DEVICES=${GPU_RANK} python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
--save_name=${SAVE_PATH}" 

CUDA_VISIBLE_DEVICES="${GPU_RANK}" "${PYTHON:-python3}" "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
--routes="${ROUTES}" \
--repetitions="${REPETITIONS}" \
--track="${CHALLENGE_TRACK_CODENAME}" \
--checkpoint="${CHECKPOINT_ENDPOINT}" \
--agent="${TEAM_AGENT}" \
--agent-config="${TEAM_CONFIG}" \
--debug="${DEBUG_CHALLENGE}" \
--record="${RECORD_PATH}" \
--resume="${RESUME}" \
--port="${PORT}" \
--traffic-manager-port="${TM_PORT}" \
--gpu-rank="${GPU_RANK}" \
--save_name="${SAVE_PATH}" \
--pid_param="${PID_PARAM}"
