#!/usr/bin/env bash
# Download a KITTI raw drive (sync), its tracklets and the date's calibration into ./data.
#   ./scripts/download_kitti.sh                          # 2011_09_26_drive_0005, the drive every default command uses
#   ./scripts/download_kitti.sh 2011_09_26_drive_0011    # the second drive shown in the README
# Only these two drives are accepted. Every zip is checked against a pinned SHA-256 before it is unzipped.
# KITTI is CC BY-NC-SA 3.0: https://www.cvlibs.net/datasets/kitti/raw_data.php
set -euo pipefail

ACCEPTED="2011_09_26_drive_0005 2011_09_26_drive_0011"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [drive]  (accepted: ${ACCEPTED// /, })" >&2
  exit 2
fi
DRIVE=${1-2011_09_26_drive_0005}  # an empty argument is refused, not read as the default

# SHA-256 of each zip, measured from the KITTI mirror below in Nov 2026. KITTI publishes no checksums, so a match
# proves you have the same files the results were produced from, not that KITTI signed them.
CALIB_SHA=e0108cfd000cf802c14ed94fba38601185c792e5640abba015a34b7a85b812e0
case "$DRIVE" in
  2011_09_26_drive_0005)
    SYNC_SHA=eaa3635b6ad6a707a8bab53ca38567ada8c78eb743e96d558497872fa436c136
    TRACKLETS_SHA=ae896496bcc762c4b0a489730b7d1b2a24513806f5e5bd4e2eeafffc86d486a4 ;;
  2011_09_26_drive_0011)
    SYNC_SHA=2c2138a5442a5bdc96918e6edec9e4785e63c8b2ac14ff45a57363923e44066f
    TRACKLETS_SHA=b5cd8b16dff0e127d14e717fda986173bd3d20de964da47865f0820370e136f4 ;;
  *)
    echo "Unknown drive '$DRIVE'. Accepted: ${ACCEPTED// /, }" >&2
    exit 2 ;;
esac
DATE=${DRIVE:0:10}

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
  3. Put the zips in ${DATA_DIR} and re-run this script with the same drive (it checks and unzips existing zips).
EOF
  exit 1
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() { # url dest expected_sha256: download to dest.part, check the hash, then rename. A bad file is removed.
  local got
  if [ ! -f "$2" ]; then
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2" || { rm -f "$2.part"; return 1; }
  fi
  got=$(sha256_of "$2")
  if [ "$got" != "$3" ]; then
    echo "Checksum mismatch for $2: expected $3, got $got. The file was removed." >&2
    rm -f "$2"
    return 1
  fi
}

check_entries() { # refuse an archive with an absolute path, a .. component or an option-like name
  local entries
  entries=$(unzip -Z1 "$1") || { echo "Cannot list $1" >&2; return 1; }
  if printf '%s\n' "$entries" | grep -qE '^/|(^|/)\.\.(/|$)|^-'; then
    echo "Refusing $1: it has an entry outside the data folder." >&2
    rm -f "$1"
    return 1
  fi
}

for tool in curl unzip; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Missing host tool: $tool. Install it (Ubuntu/Debian: sudo apt install curl unzip; macOS: both ship with the OS or brew install $tool) and re-run." >&2
    exit 1
  }
done
command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || {
  echo "Missing host tool: sha256sum or shasum (Ubuntu/Debian: coreutils; macOS: ships with the OS)." >&2
  exit 1
}

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d "${DATE}/${DRIVE}_sync/velodyne_points/data" ]; then
  fetch "${BASE}/${DRIVE}/${DRIVE}_sync.zip" "${DRIVE}_sync.zip" "$SYNC_SHA" || manual_steps
  check_entries "${DRIVE}_sync.zip" || manual_steps
  # Skip the camera images: only the LiDAR sweeps and GPS/IMU are used.
  unzip -qn "${DRIVE}_sync.zip" "${DATE}/${DRIVE}_sync/velodyne_points/*" "${DATE}/${DRIVE}_sync/oxts/*"
fi

if [ ! -f "${DATE}/calib_imu_to_velo.txt" ]; then
  fetch "${BASE}/${DATE}_calib.zip" "${DATE}_calib.zip" "$CALIB_SHA" || manual_steps
  check_entries "${DATE}_calib.zip" || manual_steps
  unzip -qn "${DATE}_calib.zip"
fi

if [ ! -f "${DATE}/${DRIVE}_sync/tracklet_labels.xml" ]; then
  fetch "${BASE}/${DRIVE}/${DRIVE}_tracklets.zip" "${DRIVE}_tracklets.zip" "$TRACKLETS_SHA" || manual_steps
  check_entries "${DRIVE}_tracklets.zip" || manual_steps
  unzip -qn "${DRIVE}_tracklets.zip"
fi

rm -f "${DRIVE}_sync.zip" "${DATE}_calib.zip" "${DRIVE}_tracklets.zip"

n_bin=$(ls "${DATE}/${DRIVE}_sync/velodyne_points/data" | wc -l | tr -d ' ')
n_ts=$(wc -l < "${DATE}/${DRIVE}_sync/velodyne_points/timestamps.txt" | tr -d ' ')
echo "OK: ${n_bin} sweeps, ${n_ts} timestamps in ${DATA_DIR}/${DATE}/${DRIVE}_sync"
