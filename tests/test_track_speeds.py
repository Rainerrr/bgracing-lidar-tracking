"""E3.2 parked speeds and E3.3 moving speeds on the real drive (offline, 154 frames), with their negative
controls and determinism. Bars from phase-3 FEATURE_SPEC D1/D2, defined in processor.track_scoring.
Fails, never skips, without ./data."""
import dataclasses
import hashlib

import pytest

from processor import track_scoring
from scripts import run_offline

from conftest import DATA_DIR

HASHED = ("tracks.csv", "tracking_speeds.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def runs(tmp_path_factory, drive, params):
    variants = {
        "on": (params, 0),
        "on_again": (params, 0),
        "other_seed": (dataclasses.replace(params, seed=params.seed + 1), 0),
        "no_ego": (dataclasses.replace(params, ego_compensation=False), 0),
    }
    return {name: run_offline.run(p, tmp_path_factory.mktemp(name), data_dir=DATA_DIR) for name, (p, _) in variants.items()}


def test_the_parked_and_moving_sets_are_the_e1_2_fixture():
    from test_pose_check import MOVING, PARKED
    assert track_scoring.PARKED == PARKED and track_scoring.MOVING == tuple(sorted(MOVING))


# ---------------------------------------------------------------- E3.2

def test_parked_tracklets_read_near_zero(runs):
    pk = runs["on"]["speeds"]["parked"]
    table = {tid: (v["median_speed"], v["matched_frames"]) for tid, v in pk["tracklets"].items()}
    assert len(table) == 10 and all(n > 0 for _, n in table.values()), table
    assert pk["median_of_medians"] <= 0.5, table
    assert all(m <= 0.7 for m, _ in table.values()), table
    assert pk["pass"]


def test_control_without_ego_compensation_parked_tracklets_read_the_ego_speed(runs):
    sp = runs["no_ego"]["speeds"]
    assert sp["tracking_frame"] == "velodyne"
    table = {tid: v["median_speed"] for tid, v in sp["parked"]["tracklets"].items()}
    assert all(m is not None and m >= 2.5 for m in table.values()), table
    assert not sp["parked"]["pass"]


# ---------------------------------------------------------------- E3.3

def test_moving_objects_read_their_speed(runs):
    mv = runs["on"]["speeds"]["moving"]["objects"]
    errors = {tid: (o["median_abs_error"], o["matched_frames"]) for tid, o in mv.items()}
    assert set(errors) == {"0", "1"} and all(n > 0 for _, n in errors.values()), errors
    assert all(e <= 0.5 for e, _ in errors.values()), errors
    assert mv["1"]["small_sample"] is True


def test_control_without_ego_compensation_moving_objects_fail_the_bar(runs):
    """Amended during the build (Nov 6): the spec's control, a 2-frame pose shift, does not fail (a constant pose
    lag at near-constant ego velocity offsets positions, not velocities: van error 0.215 m/s,
    EVIDENCE/e3.2-e3.3-speeds.txt). Without ego compensation both objects read their speed relative to the ego."""
    mv = runs["no_ego"]["speeds"]["moving"]
    errors = {tid: o["median_abs_error"] for tid, o in mv["objects"].items()}
    assert all(e is not None and e > 0.5 for e in errors.values()), errors
    assert not mv["pass"]


# ---------------------------------------------------------------- determinism and Phase 2 unchanged

def test_two_runs_are_byte_identical(runs):
    for f in HASHED:
        assert sha(runs["on"]["out_dir"] / f) == sha(runs["on_again"]["out_dir"] / f), f


def test_determinism_control_a_different_seed_changes_the_tracks(runs):
    assert sha(runs["on"]["out_dir"] / "tracks.csv") != sha(runs["other_seed"]["out_dir"] / "tracks.csv")


def test_tracks_csv_follows_the_contract(runs):
    lines = (runs["on"]["out_dir"] / "tracks.csv").read_text().splitlines()
    assert lines[0] == "pass,frame,stamp_ns,track_id,x,y,vx,vy,speed,state,l,w,h"
    rows = [line.split(",") for line in lines[1:]]
    assert rows and all(r[0] == "0" and r[9] in ("moving", "static") for r in rows)
    keys = [(int(r[1]), int(r[3])) for r in rows]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)  # sorted by frame then ID, one row per track



# ---------------------------------------------------------------- E3.4 replay on real data

def test_replay_equals_the_offline_run_and_control_a_removed_frame_differs(runs, params):
    """The replay E3.4 uses reproduces the full offline run, and is sensitive to one missing frame."""
    rows = runs["on"]["track_rows"]
    assert run_offline.replay(params, {0: list(range(154))}, DATA_DIR)[0] == rows
    without_60 = [f for f in range(154) if f != 60]
    assert run_offline.replay(params, {0: without_60}, DATA_DIR)[0] != [r for r in rows if r.split(",")[1] != "60"]


# ---------------------------------------------------------------- report tags (offline PNG only)

def test_report_tags_name_the_moving_tracklets_and_no_parked_one(runs):
    tags = runs["on"]["gt_tags"]
    tagged = {t.split()[1] for t in tags.values()}  # "#0", "#1", ...
    assert {"#0", "#1"} <= tagged, tags  # the van and the cyclist are always found
    assert not tagged & {f"#{i}" for i in track_scoring.PARKED}, tags
    assert all(t.startswith("GT #") for t in tags.values())
