#!/usr/bin/env bash
# Download KITTI raw drive 2011_09_26_drive_0005 (sync), calibration and tracklets into ./data.
# KITTI is CC BY-NC-SA 3.0: https://www.cvlibs.net/datasets/kitti/raw_data.php
set -euo pipefail

DATE=2011_09_26
DRIVE=${DATE}_drive_0005
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"

manual_steps() {
  cat >&2 <<EOF

Download failed. Manual steps:
  1. Open https://www.cvlibs.net/datasets/kitti/raw_data.php and accept the license/register if asked.
  2. Download these three files:
       [synced+rectified data]  ${DRIVE}_sync.zip
       [tracklets]              ${DRIVE}_tracklets.zip
       [calibration]            ${DATE}_calib.zip
  3. Put the zips in ${DATA_DIR} and re-run this script (it unzips existing zips).
EOF
  exit 1
}

fetch() { # url dest
  if [ ! -f "$2" ]; then
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
  fi
}

for tool in curl unzip; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Missing host tool: $tool. Install it (Ubuntu/Debian: sudo apt install curl unzip; macOS: both ship with the OS or brew install $tool) and re-run." >&2
    exit 1
  }
done

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d "${DATE}/${DRIVE}_sync/velodyne_points/data" ]; then
  fetch "${BASE}/${DRIVE}/${DRIVE}_sync.zip" "${DRIVE}_sync.zip" || manual_steps
  # Skip the camera images: only the LiDAR sweeps and GPS/IMU are used.
  unzip -qn "${DRIVE}_sync.zip" "${DATE}/${DRIVE}_sync/velodyne_points/*" "${DATE}/${DRIVE}_sync/oxts/*"
fi

if [ ! -f "${DATE}/calib_imu_to_velo.txt" ]; then
  fetch "${BASE}/${DATE}_calib.zip" "${DATE}_calib.zip" || manual_steps
  unzip -qn "${DATE}_calib.zip"
fi

if [ ! -f "${DATE}/${DRIVE}_sync/tracklet_labels.xml" ]; then
  fetch "${BASE}/${DRIVE}/${DRIVE}_tracklets.zip" "${DRIVE}_tracklets.zip" || manual_steps
  unzip -qn "${DRIVE}_tracklets.zip"
fi

rm -f "${DRIVE}_sync.zip" "${DATE}_calib.zip" "${DRIVE}_tracklets.zip"

n_bin=$(ls "${DATE}/${DRIVE}_sync/velodyne_points/data" | wc -l | tr -d ' ')
n_ts=$(wc -l < "${DATE}/${DRIVE}_sync/velodyne_points/timestamps.txt" | tr -d ' ')
echo "OK: ${n_bin} sweeps, ${n_ts} timestamps in ${DATA_DIR}/${DATE}/${DRIVE}_sync"
