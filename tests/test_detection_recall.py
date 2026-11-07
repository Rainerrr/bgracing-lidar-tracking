"""Offline detection recall and determinism, on the real drive (154 frames). Fails, never skips, without ./data."""
import dataclasses
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from core import kitti
from processor import matching
from scripts import run_offline  # noqa: F401  (import check; run() is used through the module below)

from conftest import DATA_DIR

FILES = ("detections.csv", "frames.csv", "detection_recall.json")
RECALL_BAR = 0.8
CAR_VAN_BOX_FRAMES = 520  # Car 248 + Van 272 in the evaluation region (counted from the tracklets before any detection code ran)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def runs(tmp_path_factory, drive, params):
    out = {}
    for name, p in (("run1", params), ("run2", params), ("other_seed", dataclasses.replace(params, seed=params.seed + 1))):
        out[name] = run_offline.run(p, tmp_path_factory.mktemp(name), data_dir=DATA_DIR)
    return out


@pytest.fixture(scope="module")
def gt(tracklets):
    return kitti.gt_boxes(tracklets)


def test_car_van_recall_meets_the_bar(runs):
    cv = runs["run1"]["recall"]["car_van"]
    assert cv["total"] == CAR_VAN_BOX_FRAMES, "the denominator is the 520 Car+Van box-frames in the evaluation region"
    assert cv["recall"] >= RECALL_BAR, f"Car+Van recall {cv['recall']} < {RECALL_BAR}"
    r = runs["run1"]["recall"]
    print("\nCAR_VAN", json.dumps(cv))
    print("PER_CLASS", json.dumps(r["per_class"]))
    print("PER_TRACKLET", json.dumps(r["per_tracklet"]))
    print("CENTER_ERROR", json.dumps(r["center_error_m"]))
    print("UNMATCHED", json.dumps(r["unmatched_detections_in_region"]))


def test_recall_control_a_sideways_shift_collapses_recall(runs, gt, params):
    shifted = {f: tuple(NS(x=d.x, y=d.y + 3.0, z=d.z) for d in ds) for f, ds in runs["run1"]["dets_by_frame"].items()}
    assert matching.score_detections(shifted, gt, params)["car_van"]["recall"] < 0.4


def test_recall_control_an_empty_detection_set_scores_zero(gt, params):
    assert matching.score_detections({}, gt, params)["car_van"] == {"matched": 0, "total": CAR_VAN_BOX_FRAMES, "recall": 0.0}


def test_recall_control_a_three_frame_offset_lowers_recall(runs, gt, params):
    base = runs["run1"]["recall"]["car_van"]["recall"]
    dets = runs["run1"]["dets_by_frame"]
    late = {f + 3: ds for f, ds in dets.items()}  # the stamp-to-frame mapping is part of what is being tested
    assert matching.score_detections(late, gt, params)["car_van"]["recall"] < base


def test_two_runs_are_byte_identical(runs):
    a, b = runs["run1"]["out_dir"], runs["run2"]["out_dir"]
    for name in FILES:
        assert sha(a / name) == sha(b / name), name
    print("\n" + "\n".join(f"{name} sha256 {sha(a / name)}" for name in FILES))


def test_determinism_control_a_different_seed_changes_the_output(runs):
    a, c = runs["run1"]["out_dir"], runs["other_seed"]["out_dir"]
    assert sha(a / "detection_recall.json") != sha(c / "detection_recall.json"), "the seed is recorded in the summary"
    print("\nseed 42 vs 43: detections.csv", sha(a / "detections.csv")[:12], sha(c / "detections.csv")[:12],
          "| frames.csv", sha(a / "frames.csv")[:12], sha(c / "frames.csv")[:12])


def test_every_frame_has_a_stage_row_and_valid_status(runs):
    frames = (runs["run1"]["out_dir"] / "frames.csv").read_text().splitlines()[1:]
    assert len(frames) == 154
    assert {line.split(",")[-1] for line in frames} <= {"ok", "no_plane", "too_few_points"}
    for line in frames:
        _, _, raw, roi, voxel, ng, cl, *_ = line.split(",")
        assert int(raw) >= int(roi) >= int(voxel) >= int(ng) >= int(cl)


def test_hashed_files_carry_no_timing(runs):
    for name in FILES:
        assert "time" not in (runs["run1"]["out_dir"] / name).read_text().lower().replace("stamp", "")
