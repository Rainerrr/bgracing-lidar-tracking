"""scripts/live_check.py compare (markers, detection parity) and files (output files) on fabricated live runs (no ROS, no data).

/perception/bounding_boxes carries one box per confirmed track (README, Architecture), so the marker check
compares the three track topics with the live tracks.csv; the detection-marker properties (DELETEALL first, counts equal
to the live rows of that stamp) are the same."""
import json
import subprocess
import sys
from pathlib import Path

from processor import records as rec
from test_records import FRAMES, arr, det, result  # fabricated frames and marker arrays

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_check.py"


TOPICS = ("/perception/bounding_boxes", "/perception/obstacle_vectors", "/perception/track_labels")
TRACKS = {1000: ["moving", "static"], 2000: [], 3000: ["static"]}  # live confirmed tracks per stamp


def trow(p, f, s, tid, state):
    return f"{p},{f},{s},{tid},1.0000,2.0000,0.5000,0.0000,0.5000,{state},4.5000,1.8000,1.4000"


def tarr(stamp, states, p=0, first=rec.DELETEALL):
    """The three arrays the node publishes for one frame."""
    n, mv = len(states), sum(x == "moving" for x in states)
    out = []
    for topic, field, k in zip(TOPICS, ("n_line_list", "n_arrow", "n_text"), (n, mv, n)):
        a = {"topic": topic, "pass": p, "stamp_ns": stamp, "first_action": first, "n_line_list": 0, "n_text": 0,
             "n_arrow": 0, "n_total": 1 + k}
        a[field] = k
        out.append(a)
    return out


def build(tmp_path, live_frames, arrays, processed=None, offline_frames=FRAMES):
    live, off = tmp_path / "live", tmp_path / "live" / "offline"
    off.mkdir(parents=True)
    det_l, frm_l = ["pass," + rec.DETECTION_HEADER], ["pass," + rec.FRAMES_HEADER]
    for p, f, s, r, ds in live_frames:
        frm_l.append(f"{p},{rec.frames_row(f, s, r)}")
        det_l += [f"{p},{row}" for row in rec.detection_rows(f, s, ds)]
    (live / "detections.csv").write_text("\n".join(det_l) + "\n")
    (live / "frames.csv").write_text("\n".join(frm_l) + "\n")
    trk = [rec.TRACK_HEADER] + [trow(p, f, s, i + 1, st) for p, f, s, _, _ in live_frames for i, st in enumerate(TRACKS[s])]
    (live / "tracks.csv").write_text("\n".join(trk) + "\n")
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
GOOD_ARRAYS = tarr(1000, TRACKS[1000]) + tarr(2000, TRACKS[2000]) + tarr(3000, TRACKS[3000])


def test_good_run_passes_when_enough_distinct_frames(tmp_path):
    live = build(tmp_path, GOOD_LIVE, GOOD_ARRAYS)
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.argv=['x','compare','--live-dir',%r,'--offline-dir',%r,'--markers',%r];"
                        "import scripts.live_check as m; m.MIN_FRAMES=3; sys.exit(m.main())" % (
                            str(live), str(live / "offline"), str(live / "live_markers.json"))],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "markers: 9 arrays checked" in r.stdout and "0 problems -> OK" in r.stdout and "consistency" in r.stdout


def test_too_few_distinct_frames_fails_the_real_gate(tmp_path):
    r = compare(build(tmp_path, GOOD_LIVE, GOOD_ARRAYS))
    assert r.returncode == 1 and "need >= 140" in r.stdout and "detection parity" in r.stdout and "FAIL" in r.stdout


def test_a_marker_array_without_deleteall_fails(tmp_path):
    r = compare(build(tmp_path, GOOD_LIVE, GOOD_ARRAYS[:-1] + tarr(3000, TRACKS[3000], first=0)[-1:]))
    assert r.returncode == 1 and "1 problems -> FAIL" in r.stdout


def test_an_extra_box_or_a_missing_arrow_fails(tmp_path):
    extra = GOOD_ARRAYS[:6] + tarr(3000, ["static", "static"])
    r = compare(build(tmp_path, GOOD_LIVE, extra))
    assert r.returncode == 1 and "-> FAIL" in r.stdout.splitlines()[0]
    no_arrow = tarr(1000, ["static", "static"]) + GOOD_ARRAYS[3:]
    out = rec.compare_track_markers(no_arrow, rec.read_live_rows(tmp_path / "live" / "detections.csv", tmp_path / "live" / "frames.csv"),
                                    rec.read_live_tracks(tmp_path / "live" / "tracks.csv"))
    assert not out["ok"] and any(p["topic"] == TOPICS[1] for p in out["problems"])


def test_a_marker_array_of_another_pass_fails(tmp_path):
    r = compare(build(tmp_path, GOOD_LIVE, GOOD_ARRAYS[:6] + tarr(3000, TRACKS[3000], p=1)))
    assert r.returncode == 1 and "not processed live" in r.stdout


def test_metrics_ahead_of_the_csv_fails_the_consistency_check(tmp_path):
    r = compare(build(tmp_path, GOOD_LIVE, GOOD_ARRAYS, processed=4))
    assert r.returncode == 1 and "consistency" in r.stdout and "-> FAIL" in r.stdout.splitlines()[-1]


def test_missing_input_files_exit_1_with_a_hint(tmp_path):
    r = compare(tmp_path / "empty")
    assert r.returncode == 1 and "not found" in r.stderr


# ---------------------------------------------------------------- files

def files(live):
    return subprocess.run([sys.executable, str(SCRIPT), "files", "--live-dir", str(live)], capture_output=True, text=True)


def complete_run(tmp_path):
    live = build(tmp_path, GOOD_LIVE, GOOD_ARRAYS)
    m = json.loads((live / "metrics.json").read_text())
    m["processed_frames"] = [[0, f] for f, _, _, _ in FRAMES]
    m["track_rows"] = len((live / "tracks.csv").read_text().splitlines()) - 1
    (live / "metrics.json").write_text(json.dumps(m))
    from processor import report
    report.render_report(live / "obstacle_tracking_report.png", (live / "tracks.csv").read_text().splitlines()[1:], [(0, 0)])
    return live


def test_files_ok_on_a_complete_run(tmp_path):
    r = files(complete_run(tmp_path))
    assert r.returncode == 0 and "0 problems -> OK" in r.stdout, r.stdout


def test_files_control_a_truncated_tracks_csv_fails(tmp_path):
    live = complete_run(tmp_path)
    text = (live / "tracks.csv").read_text()
    (live / "tracks.csv").write_text(text[: len(text) - 20])
    r = files(live)
    assert r.returncode == 1 and "malformed" in r.stdout


def test_files_control_a_truncated_png_fails(tmp_path):
    live = complete_run(tmp_path)
    png = live / "obstacle_tracking_report.png"
    png.write_bytes(png.read_bytes()[:200])
    r = files(live)
    assert r.returncode == 1 and "PNG" in r.stdout and "FAIL" in r.stdout


def test_files_control_a_tracks_csv_missing_its_last_frame_fails(tmp_path):
    """A cut at a row boundary (a stale periodic write) keeps every row well formed; the row count catches it."""
    live = complete_run(tmp_path)
    lines = (live / "tracks.csv").read_text().splitlines()
    (live / "tracks.csv").write_text("\n".join(lines[:-1]) + "\n")
    r = files(live)
    assert r.returncode == 1 and "metrics.json says" in r.stdout
