"""scripts/live_check.py compare: E2.5 / E2.6 verdicts on fabricated live runs (no ROS, no data)."""
import json
import subprocess
import sys
from pathlib import Path

from processor import records as rec
from test_records import FRAMES, arr, det, result  # fabricated frames and marker arrays

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_check.py"


def build(tmp_path, live_frames, arrays, processed=None, offline_frames=FRAMES):
    live, off = tmp_path / "live", tmp_path / "live" / "offline"
    off.mkdir(parents=True)
    det_l, frm_l = ["pass," + rec.DETECTION_HEADER], ["pass," + rec.FRAMES_HEADER]
    for p, f, s, r, ds in live_frames:
        frm_l.append(f"{p},{rec.frames_row(f, s, r)}")
        det_l += [f"{p},{row}" for row in rec.detection_rows(f, s, ds)]
    (live / "detections.csv").write_text("\n".join(det_l) + "\n")
    (live / "frames.csv").write_text("\n".join(frm_l) + "\n")
    (live / "metrics.json").write_text(json.dumps({"frames_processed": len(live_frames) if processed is None else processed}))
    (off / "detections.csv").write_text(rec.csv_text(rec.DETECTION_HEADER, [r for f, s, _, ds in offline_frames for r in rec.detection_rows(f, s, ds)]))
    (off / "frames.csv").write_text(rec.csv_text(rec.FRAMES_HEADER, [rec.frames_row(f, s, r) for f, s, r, _ in offline_frames]))
    (live / "live_markers.json").write_text(json.dumps({"arrays": arrays}))
    return live


def compare(live, extra=()):
    return subprocess.run([sys.executable, str(SCRIPT), "compare", "--live-dir", str(live),
                           "--offline-dir", str(live / "offline"), "--markers", str(live / "live_markers.json"), *extra],
                          capture_output=True, text=True)


GOOD_LIVE = [(0, f, s, r, d) for f, s, r, d in FRAMES]
GOOD_ARRAYS = [arr(1000, 2), arr(2000, 0), arr(3000, 1)]


def test_good_run_passes_when_enough_distinct_frames(tmp_path):
    live = build(tmp_path, GOOD_LIVE, GOOD_ARRAYS)
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.argv=['x','compare','--live-dir',%r,'--offline-dir',%r,'--markers',%r];"
                        "import scripts.live_check as m; m.MIN_FRAMES=3; sys.exit(m.main())" % (
                            str(live), str(live / "offline"), str(live / "live_markers.json"))],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "E2.5 markers: 3 arrays checked, 0 problems -> OK" in r.stdout and "consistency" in r.stdout


def test_too_few_distinct_frames_fails_the_real_gate(tmp_path):
    r = compare(build(tmp_path, GOOD_LIVE, GOOD_ARRAYS))
    assert r.returncode == 1 and "need >= 140" in r.stdout and "E2.6 parity" in r.stdout and "FAIL" in r.stdout


def test_a_marker_array_without_deleteall_fails(tmp_path):
    r = compare(build(tmp_path, GOOD_LIVE, [arr(1000, 2, first=0)]))
    assert r.returncode == 1 and "E2.5 markers: 1 arrays checked, 1 problems -> FAIL" in r.stdout


def test_metrics_ahead_of_the_csv_fails_the_consistency_check(tmp_path):
    r = compare(build(tmp_path, GOOD_LIVE, GOOD_ARRAYS, processed=4))
    assert r.returncode == 1 and "consistency" in r.stdout and "-> FAIL" in r.stdout.splitlines()[-1]


def test_missing_input_files_exit_1_with_a_hint(tmp_path):
    r = compare(tmp_path / "empty")
    assert r.returncode == 1 and "not found" in r.stderr
