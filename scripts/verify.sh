#!/usr/bin/env bash
# Verify command: build both images, smoke-test the publisher's imports, run the unit tests.
# Needs the KITTI drive in ./data (scripts/download_kitti.sh); real-data tests fail without it.
# Extra arguments go to pytest, e.g. scripts/verify.sh -k pose_check -rP
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose build
docker compose run --rm --no-deps publisher python3 -c \
  "import core.kitti, core.poses; from nav_msgs.msg import Odometry; from tf2_ros import TransformBroadcaster; print('publisher imports OK')"
# docs/samples is mounted read-only for the sample-regeneration check (E4.5) and the repo for the README test (E5.7):
# the images copy neither
docker compose run --rm --no-deps -v "$PWD/docs/samples:/samples:ro" -v "$PWD:/repo:ro" processor python3 -m pytest -q -p no:cacheprovider /app/tests "$@"
