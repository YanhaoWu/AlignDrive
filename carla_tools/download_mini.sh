#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${ALIGNDRIVE_MINI_DIR:-Bench2Drive-mini}"
TMP_DIR="${ALIGNDRIVE_MINI_TMP_DIR:-${OUTPUT_DIR}.download_tmp}"
mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

# Check if huggingface-hub is installed
if ! python -m pip show huggingface-hub > /dev/null 2>&1; then
  echo "huggingface-hub is not installed. Installing now..."
  python -m pip install huggingface-hub
else
  echo "huggingface-hub is already installed."
fi

download_one() {
  local archive="$1"
  local tmp_archive="$TMP_DIR/$archive"
  local final_archive="$OUTPUT_DIR/$archive"
  if [[ -f "$final_archive" ]]; then
    echo "Already present: $final_archive"
    return
  fi
  huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "$archive" --local-dir "$TMP_DIR" --local-dir-use-symlinks False
  [[ -f "$tmp_archive" ]] || { echo "Download did not produce $tmp_archive" >&2; exit 1; }
  mv "$tmp_archive" "$final_archive"
}

download_one "HardBreakRoute_Town01_Route30_Weather3.tar.gz"
download_one "DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz"
download_one "Accident_Town03_Route156_Weather0.tar.gz"
download_one "YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz"
download_one "ConstructionObstacle_Town05_Route68_Weather8.tar.gz"
download_one "ParkedObstacle_Town10HD_Route371_Weather7.tar.gz"
download_one "ControlLoss_Town11_Route401_Weather11.tar.gz"
download_one "AccidentTwoWays_Town12_Route1444_Weather0.tar.gz"
download_one "OppositeVehicleTakingPriority_Town13_Route600_Weather2.tar.gz"
download_one "VehicleTurningRoute_Town15_Route443_Weather1.tar.gz"

echo "Downloads are in $OUTPUT_DIR; resumable temporary files remain isolated in $TMP_DIR."
