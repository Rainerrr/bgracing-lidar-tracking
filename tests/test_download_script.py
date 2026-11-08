"""scripts/download_kitti.sh: the drive argument, the pinned checksums and the archive-entry guard.

The script is copied into a temporary tree, so its data folder is <tmp>/data. curl, unzip and sha256sum are replaced
by stubs on PATH that log every call, so nothing is downloaded and the tests run without the KITTI data.
- curl writes the zip's file name as its content.
- sha256sum prints the pinned hash for that content, unless the test marks the zip as corrupted.
- unzip -Z1 lists STUB_ENTRIES; unzip -qn creates the files the real archive would add.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_kitti.sh"
DRIVE_A, DRIVE_B = "2011_09_26_drive_0005", "2011_09_26_drive_0011"

CURL = """#!/bin/sh
echo "curl $*" >> "$STUB_LOG"
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
url=""; for a in "$@"; do url="$a"; done
basename "$url" > "$out"
"""
SHA = """#!/bin/sh
echo "sha256sum $*" >> "$STUB_LOG"
name=$(cat "$1")
if [ "$name" = "$STUB_CORRUPT" ]; then echo "0000000000000000000000000000000000000000000000000000000000000000  $1"; exit 0; fi
grep "^$name " "$STUB_PINS" | cut -d' ' -f2 | sed "s|\\$|  $1|"
"""
UNZIP = """#!/bin/sh
echo "unzip $*" >> "$STUB_LOG"
if [ "$1" = "-Z1" ]; then printf '%s\\n' "${STUB_ENTRIES:-2011_09_26/ok.txt}"; exit 0; fi
zip="$2"; date=2011_09_26
case "$zip" in
  *_sync.zip) d="$date/${zip%.zip}/velodyne_points"; mkdir -p "$d/data" "$date/${zip%.zip}/oxts/data"
              : > "$d/data/0000000000.bin"; echo "2011-09-26 13:04:32.335337762" > "$d/timestamps.txt" ;;
  *_tracklets.zip) d="$date/${zip%_tracklets.zip}_sync"; mkdir -p "$d"; : > "$d/tracklet_labels.xml" ;;
  *_calib.zip) mkdir -p "$date"; : > "$date/calib_imu_to_velo.txt" ;;
esac
"""


def pinned_hashes():
    """{zip name: pinned sha256} read from the script itself."""
    text = SCRIPT.read_text(encoding="utf-8")
    calib = re.search(r"CALIB_SHA=([0-9a-f]{64})", text).group(1)
    pins = {"2011_09_26_calib.zip": calib}
    for drive in (DRIVE_A, DRIVE_B):
        block = text.split(f"  {drive})", 1)[1].split(";;", 1)[0]
        pins[f"{drive}_sync.zip"] = re.search(r"SYNC_SHA=([0-9a-f]{64})", block).group(1)
        pins[f"{drive}_tracklets.zip"] = re.search(r"TRACKLETS_SHA=([0-9a-f]{64})", block).group(1)
    return pins


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "download_kitti.sh")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (("curl", CURL), ("sha256sum", SHA), ("unzip", UNZIP)):
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    pins = tmp_path / "pins.txt"
    pins.write_text("".join(f"{k} {v}\n" for k, v in pinned_hashes().items()))
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "STUB_LOG": str(log), "STUB_PINS": str(pins),
           "STUB_CORRUPT": ""}

    def run(*args, **extra):
        r = subprocess.run(["bash", str(tmp_path / "scripts" / "download_kitti.sh"), *args],
                           env={**env, **extra}, capture_output=True, text=True, timeout=60)
        return r, log.read_text().splitlines()
    run.root = tmp_path
    return run


def test_the_pins_cover_both_drives_and_the_calibration():
    pins = pinned_hashes()
    assert len(pins) == 5 and len(set(pins.values())) == 5


@pytest.mark.parametrize("args", [["../x"], ["2011_09_26_drive_0011/../.."], ["-x"], ["a b"], [DRIVE_B + "\n"],
                                  ["2011_09_26_drive_0013"], ["2011_09_26"], [""], [DRIVE_A, DRIVE_B]])
def test_a_bad_or_unknown_drive_is_refused_before_any_action(sandbox, args):
    r, calls = sandbox(*args)
    assert r.returncode == 2, r.stderr
    assert "Accepted" in r.stderr or "accepted" in r.stderr
    assert calls == [] and not (sandbox.root / "data").exists()


@pytest.mark.parametrize("args, drive_name", [([], DRIVE_A), ([DRIVE_B], DRIVE_B)])
def test_the_happy_path_checks_every_zip_then_unzips_it(sandbox, args, drive_name):
    r, calls = sandbox(*args)
    assert r.returncode == 0, r.stderr
    assert f"OK: 1 sweeps, 1 timestamps in {sandbox.root / 'data' / '2011_09_26' / (drive_name + '_sync')}" in r.stdout
    assert sum(c.startswith("curl ") for c in calls) == 3
    assert sum(c.startswith("sha256sum ") for c in calls) == 3
    assert sum(c.startswith("unzip -qn ") for c in calls) == 3
    assert not list((sandbox.root / "data").glob("*.zip")) and not list((sandbox.root / "data").glob("*.part"))


def test_a_second_run_with_complete_data_does_nothing(sandbox):
    sandbox(DRIVE_B)
    r, calls = sandbox(DRIVE_B)
    assert r.returncode == 0 and "OK: 1 sweeps" in r.stdout
    assert sum(c.startswith("curl ") for c in calls) == 3  # only the first run fetched


def test_a_corrupted_zip_is_removed_and_never_unzipped(sandbox):
    r, calls = sandbox(DRIVE_B, STUB_CORRUPT=f"{DRIVE_B}_sync.zip")
    assert r.returncode == 1
    assert "Checksum mismatch" in r.stderr and "Manual steps" in r.stderr
    assert not any(c.startswith("unzip") for c in calls)
    data = sandbox.root / "data"
    assert not list(data.glob("*.zip")) and not list(data.glob("*.part")) and not (data / "2011_09_26").exists()


def test_an_existing_zip_with_the_wrong_content_is_also_refused(sandbox):
    data = sandbox.root / "data"
    data.mkdir()
    (data / f"{DRIVE_A}_sync.zip").write_text("not the real zip")
    r, calls = sandbox()
    assert r.returncode == 1 and "Checksum mismatch" in r.stderr
    assert not any(c.startswith("curl") or c.startswith("unzip") for c in calls)


@pytest.mark.parametrize("entry", ["../evil", "2011_09_26/../../evil", "/abs/path", "-x"])
def test_an_archive_entry_outside_the_data_folder_is_refused(sandbox, entry):
    r, calls = sandbox(DRIVE_B, STUB_ENTRIES=f"2011_09_26/ok.txt\n{entry}")
    assert r.returncode == 1 and "Refusing" in r.stderr
    assert not any(c.startswith("unzip -qn") for c in calls)
    assert not list((sandbox.root / "data").glob("*.zip"))
