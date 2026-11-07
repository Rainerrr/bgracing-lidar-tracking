#!/usr/bin/env bash
# Verify command: build both images, smoke-test the publisher's imports, run the unit tests.
# Needs the KITTI drive in ./data (scripts/download_kitti.sh); real-data tests fail without it.
# Extra arguments go to pytest, e.g. scripts/verify.sh -k pose_check -rP
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose build
docker compose run --rm --no-deps publisher python3 -c \
  "import core.kitti, core.poses; from nav_msgs.msg import Odometry; from tf2_ros import TransformBroadcaster; print('publisher imports OK')"

# The list of project files for the submission check (tests/test_submission.py): the tracked files in a git
# checkout, otherwise every file of the downloaded tree except the drive and the outputs.
files="$(mktemp)"
trap 'rm -f "$files"' EXIT
if [ "$(git rev-parse --show-toplevel 2>/dev/null)" = "$(pwd -P)" ]; then
  git -c core.quotepath=off ls-files > "$files"
else
  find . -type f ! -name .DS_Store ! -path './.git/*' \
    \( \( ! -path './data/*' ! -path './output/*' \) -o -name .gitkeep \) | sed 's|^\./||' | sort > "$files"
fi
chmod 644 "$files"

# docs/samples is mounted read-only for the sample-regeneration check, the repo for the README and submission
# checks, and the file list next to it: the images copy none of them
docker compose run --rm --no-deps -v "$PWD/docs/samples:/samples:ro" -v "$PWD:/repo:ro" -v "$files:/files.txt:ro" \
  -e TRACKED_LIST=/files.txt processor python3 -m pytest -q -p no:cacheprovider /app/tests "$@"
