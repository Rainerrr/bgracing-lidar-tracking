"""Row formats, statistics and the live-checker comparers (E2.3 formatting, E2.5 and E2.6 negative controls)."""
import json
from types import SimpleNamespace as NS

import pytest

from processor import records as rec


def det(x=1.0, y=2.0, z=-0.5, dx=4.5, dy=1.8, dz=1.4, n=120):
    return NS(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, n_points=n)


def result(raw=100, roi=90, voxel=80, ng=40, cl=30, found=3, kept=1, status="ok"):
    return NS(stages={"raw": raw, "roi": roi, "voxel": voxel, "non_ground": ng, "clustered": cl},
              clusters_found=found, clusters_kept=kept, ground_status=status)


def test_row_formats_are_fixed():
    assert rec.detection_rows(7, 123, [det()]) == ["7,123,1.0000,2.0000,-0.5000,4.5000,1.8000,1.4000,120"]
    assert rec.frames_row(7, 123, result()) == "7,123,100,90,80,40,30,3,1,ok"
    assert rec.csv_text(rec.DETECTION_HEADER, ["a"]).splitlines() == [rec.DETECTION_HEADER, "a"]
    assert rec.csv_text("h", ["a", "b"], pass_no=2) == "pass,h\n2,a\n2,b\n"


def test_nearest_rank_p95():
    assert rec.p95_nearest_rank(list(range(1, 101))) == 95
    assert rec.p95_nearest_rank(list(range(1, 155))) == 147  # ceil(0.95 * 154) = 147th value
    assert rec.p95_nearest_rank([5.0]) == 5.0 and rec.p95_nearest_rank([]) is None


def test_time_stats():
    s = rec.time_stats([10.0, 20.0, 30.0, 40.0], "scope")
    assert s["mean"] == 25.0 and s["max"] == 40.0 and s["p95"] == 40.0 and s["scope"] == "scope"
    assert rec.time_stats([], "x")["n"] == 0


def test_canonical_json_is_stable_and_rounds():
    a = rec.canonical_json({"b": 1.123456789, "a": [0.1 + 0.2]})
    assert a == rec.canonical_json({"a": [0.30000000000000004], "b": 1.12345678}) and json.loads(a) == {"a": [0.3], "b": 1.1235}


def test_atomic_write_leaves_no_temp_file(tmp_path):
    rec.atomic_write(tmp_path / "sub" / "m.json", "{}")
    assert (tmp_path / "sub" / "m.json").read_text() == "{}" and not list((tmp_path / "sub").glob("*.tmp"))


# ---------------------------------------------------------------- live files and parity (E2.6)

def write_live(tmp_path, frames):
    """frames: [(pass, frame, stamp, result, [det])] -> live-format files."""
    det_lines, frame_lines = ["pass," + rec.DETECTION_HEADER], ["pass," + rec.FRAMES_HEADER]
    for p, f, s, r, ds in frames:
        frame_lines.append(f"{p},{rec.frames_row(f, s, r)}")
        det_lines += [f"{p},{row}" for row in rec.detection_rows(f, s, ds)]
    (tmp_path / "d.csv").write_text("\n".join(det_lines) + "\n")
    (tmp_path / "f.csv").write_text("\n".join(frame_lines) + "\n")
    return rec.read_live_rows(tmp_path / "d.csv", tmp_path / "f.csv")


def write_offline(tmp_path, frames):
    (tmp_path / "od.csv").write_text(rec.csv_text(rec.DETECTION_HEADER, [r for f, s, _, ds in frames for r in rec.detection_rows(f, s, ds)]))
    (tmp_path / "of.csv").write_text(rec.csv_text(rec.FRAMES_HEADER, [rec.frames_row(f, s, r) for f, s, r, _ in frames]))
    return rec.read_offline_rows(tmp_path / "od.csv", tmp_path / "of.csv")


FRAMES = [(0, 1000, result(kept=2), [det(10, 1), det(20, -2)]), (1, 2000, result(kept=0, cl=0), []), (2, 3000, result(kept=1), [det(5, 5)])]


def test_parity_ok_for_a_subset_with_two_passes(tmp_path):
    offline = write_offline(tmp_path, FRAMES)
    live = write_live(tmp_path, [(0, f, s, r, d) for f, s, r, d in FRAMES[:2]] + [(1, f, s, r, d) for f, s, r, d in FRAMES])
    out = rec.compare_parity(live, offline, min_frames=3)
    assert out == {"compared": 5, "distinct_frames": 3, "mismatched": [], "ok": True}


def test_parity_control_one_changed_row_is_flagged(tmp_path):
    offline = write_offline(tmp_path, FRAMES)
    bad = [(0, 0, 1000, result(kept=2), [det(10, 1), det(20.0001, -2)])]
    live = write_live(tmp_path, bad + [(0, 1, 2000, result(kept=0, cl=0), [])])
    out = rec.compare_parity(live, offline)
    assert not out["ok"] and out["mismatched"] == [{"pass": 0, "frame": 0}]


def test_parity_control_missing_and_extra_detection_and_stage_difference(tmp_path):
    offline = write_offline(tmp_path, FRAMES)
    for bad in ([(0, 0, 1000, result(kept=2), [det(10, 1)])],  # a detection missing
                [(0, 0, 1000, result(kept=2), [det(10, 1), det(20, -2), det(30, 0)])],  # an extra one
                [(0, 0, 1000, result(kept=2, ng=41), [det(10, 1), det(20, -2)])],  # stage counts differ
                [(0, 0, 9999, result(kept=2), [det(10, 1), det(20, -2)])]):  # stamp differs
        assert not rec.compare_parity(write_live(tmp_path, bad), offline)["ok"]


def test_parity_needs_enough_frames(tmp_path):
    offline = write_offline(tmp_path, FRAMES)
    live = write_live(tmp_path, [(0, f, s, r, d) for f, s, r, d in FRAMES])
    assert rec.compare_parity(live, offline, min_frames=3)["ok"] and not rec.compare_parity(live, offline, min_frames=4)["ok"]
    assert not rec.compare_parity({}, offline)["ok"]


# ---------------------------------------------------------------- marker checker (E2.5)

def arr(stamp, boxes, first=rec.DELETEALL, texts=None):
    texts = boxes if texts is None else texts
    return {"stamp_ns": stamp, "first_action": first, "n_line_list": boxes, "n_text": texts, "n_total": 1 + boxes + texts}


def live_for(tmp_path):
    return write_live(tmp_path, [(0, f, s, r, d) for f, s, r, d in FRAMES] + [(1, f, s, r, d) for f, s, r, d in FRAMES])


def test_markers_ok_including_an_empty_frame(tmp_path):
    out = rec.compare_markers([arr(1000, 2), arr(2000, 0), arr(3000, 1), arr(1000, 2)], live_for(tmp_path))
    assert out["ok"] and out["arrays"] == 4


@pytest.mark.parametrize("bad,why", [
    (arr(1000, 2, first=0), "DELETEALL"),               # fabricated: no DELETEALL first
    (arr(1000, 3), "3 boxes but 2 detections"),         # an extra box
    (arr(1000, 1), "1 boxes but 2 detections"),         # a missing box
    (arr(2000, 1), "1 boxes but 0 detections"),         # a stale box on an empty frame
    (arr(1000, 2, texts=1), "label per detection"),     # a missing label
    (arr(4242, 0), "not processed live"),               # an array for a stamp that was never processed
])
def test_marker_checker_flags_bad_arrays(tmp_path, bad, why):
    out = rec.compare_markers([arr(3000, 1), bad], live_for(tmp_path))
    assert not out["ok"] and len(out["problems"]) == 1 and why in out["problems"][0]["problem"]


def test_marker_checker_needs_at_least_one_array(tmp_path):
    assert not rec.compare_markers([], live_for(tmp_path))["ok"]


def test_marker_checker_flags_a_stamp_with_different_counts_across_passes(tmp_path):
    live = write_live(tmp_path, [(0, 0, 1000, result(), [det()]), (1, 0, 1000, result(), [det(), det(9, 9)])])
    assert "differ across passes" in rec.compare_markers([arr(1000, 1)], live)["problems"][0]["problem"]
