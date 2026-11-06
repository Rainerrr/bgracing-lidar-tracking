"""Phase 4 on the real drive (offline, 154 frames): E4.2 consistency, E4.4 ablation gate, E4.6 cross-check with the
Phase 3 speed summary. Uses the session-scoped `runs` fixture of tests/conftest.py. Fails, never skips, without ./data."""
import dataclasses
import math
import re
from pathlib import Path

import pytest

from core import kitti, poses
from processor import evaluation, track_scoring

from conftest import SAMPLES_DIR

N_FRAMES = 154
CAR_VAN_BOX_FRAMES = 520  # the denominator of the Phase 2 recall


@pytest.fixture(scope="module")
def inputs(drive, tracklets):
    return kitti.gt_boxes(tracklets), poses.world_velo_poses(drive.oxts, drive.T_velo_imu), drive.stamps_ns


def evaluate_run(run, params, inputs):
    gt, T_wv, stamps = inputs
    p = params if run["metrics"]["params"]["ego_compensation"] else dataclasses.replace(params, ego_compensation=False)
    return evaluation.evaluate(run["track_rows"], run["dets_by_frame"], gt, T_wv, stamps, p, N_FRAMES)


@pytest.fixture(scope="module")
def ev(runs, params, inputs):
    return {"on": evaluate_run(runs["on"], params, inputs), "no_ego": evaluate_run(runs["no_ego"], params, inputs)}


def walk(obj):
    """Every number in a nested result."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


# ---------------------------------------------------------------- E4.2 consistency

def test_the_static_tracklets_are_the_e1_2_parked_fixture(inputs):
    gt, T_wv, stamps = inputs
    assert evaluation.static_tracklets(gt, T_wv, stamps) == set(track_scoring.PARKED)


@pytest.mark.parametrize("run", ["on", "no_ego"])
def test_matched_plus_missed_equals_the_ground_truth_in_the_region(ev, run):
    rec = ev[run]["recall"]
    assert rec["overall"]["num"] + rec["missed"] == rec["overall"]["den"]
    for cls, c in rec["per_class"].items():
        assert c["num"] + c["missed"] == c["den"], cls
    assert sum(c["den"] for c in rec["per_class"].values()) == rec["overall"]["den"]
    assert rec["car_van"]["den"] == CAR_VAN_BOX_FRAMES
    assert set(rec["per_class"]) == {"Car", "Cyclist", "Pedestrian", "Van"}
    assert rec["per_class"]["Pedestrian"]["small_sample"] and rec["per_class"]["Cyclist"]["small_sample"]
    assert not rec["per_class"]["Car"]["small_sample"] and not rec["per_class"]["Van"]["small_sample"]


@pytest.mark.parametrize("run", ["on", "no_ego"])
def test_track_counts_add_up(ev, run):
    e = ev[run]
    pr = e["produced"]
    assert pr["labelled_tracks"] + pr["unlabelled_tracks"] == pr["tracks_in_region"] <= pr["tracks"]
    st = e["stillness"]
    assert st["unlabelled"]["tracks"] == pr["unlabelled_tracks"]
    for part in st.values():
        assert part["scored"] + part["excluded"] == part["tracks"]
    assert e["coverage"]["num"] + e["unlabelled_track_frames"] == e["coverage"]["den"]
    assert len(e["tracklets"]) == 15  # every tracklet has box-frames in the region, matched or not


@pytest.mark.parametrize("run", ["on", "no_ego"])
def test_every_number_is_finite_and_every_missing_one_names_its_zero_denominator(ev, run):
    e = ev[run]
    assert all(math.isfinite(x) for x in walk(e))

    def dicts(o):
        if isinstance(o, dict):
            if "value" in o:
                yield o
            for v in o.values():
                yield from dicts(v)

    for d in dicts(e):
        if d["value"] is None:
            assert d["den_name"], d  # n/a is only ever printed together with the name of its denominator


def test_the_compensated_run_has_the_expected_shape(ev):
    e = ev["on"]
    assert e["tracking_frame"] == "world"
    assert e["recall"]["car_van"]["value"] >= 0.8  # Phase 2's bar on the same frames; tracks follow detections
    assert 0 < e["coverage"]["value"] < 0.5  # most tracks have no KITTI label: coverage is a lower bound (D14)
    assert e["stillness"]["unlabelled"]["scored"] > 50  # enough unlabelled tracks to say something (D15)


# ---------------------------------------------------------------- E4.4 ablation gate (D1)

def test_without_ego_compensation_static_velocity_error_is_at_least_three_times_larger(ev):
    abl = evaluation.ablation(ev["on"], ev["no_ego"])
    assert ev["no_ego"]["tracking_frame"] == "velodyne"
    assert abl["bar"] == 3.0 and abl["ratio"] is not None and abl["ratio"] >= 3.0, abl
    assert abl["pass"]


def test_ablation_control_a_run_compared_with_itself_fails_the_gate(ev):
    assert not evaluation.ablation(ev["on"], ev["on"])["pass"]


def test_the_uncompensated_run_labels_parked_objects_moving(ev):
    """Context for the ablation row (not gated): without the ego pose parked cars read as moving."""
    assert ev["no_ego"]["labels"]["flicker"]["value"] > 0.9
    assert ev["on"]["labels"]["flicker"]["value"] < 0.1


# ---------------------------------------------------------------- E4.6 cross-check with Phase 3

def cross_check(e, speeds):
    """Mismatches between an evaluation and the Phase 3 speed summary of the same run (same matching, warm-up)."""
    bad = []
    for tid, v in speeds["parked"]["tracklets"].items():
        t = e["tracklets"].get(int(tid))
        if t is None or t["scored_frames"] != v["matched_frames"]:
            bad.append(("parked frames", tid))
            continue
        if v["matched_frames"] and (round(t["median_speed"], 4) != v["median_speed"]
                                    or round(t["share_moving"]["value"], 4) != v["share_moving"]):
            bad.append(("parked speed or share_moving", tid))
    for tid, v in speeds["moving"]["objects"].items():
        t = e["tracklets"].get(int(tid))
        if t is None or round(t["median_abs_error"], 4) != v["median_abs_error"]:
            bad.append(("moving error", tid))
    return bad


@pytest.mark.parametrize("run", ["on", "no_ego"])
def test_speeds_equal_the_phase_3_summary(runs, ev, run):
    """The van and cyclist median speed error and each parked tracklet's median speed and moving share are the
    numbers in tracking_speeds.json (same matching, same warm-up, same tracks)."""
    assert cross_check(ev[run], runs[run]["speeds"]) == []


def test_cross_check_control_a_different_warm_up_makes_it_fail(runs, params, inputs):
    other = evaluate_run(runs["on"], dataclasses.replace(params, warmup_frames=params.warmup_frames + 1), inputs)
    assert cross_check(other, runs["on"]["speeds"]) != []


def test_parked_flicker_is_the_pooled_share_moving_of_the_parked_tracklets(ev):
    e = ev["on"]
    parked = [t for t in e["tracklets"].values() if t["static"]]
    moving = sum(t["share_moving"]["num"] for t in parked)
    scored = sum(t["share_moving"]["den"] for t in parked)
    assert e["labels"]["flicker"]["num"] == moving and e["labels"]["flicker"]["den"] == scored


# ---------------------------------------------------------------- E4.1 determinism, E4.2b structure (D13)

HEADINGS = ["## " + h for h in evaluation.PARTS]


def render_pair(ev_on, ev_off, params):
    return evaluation.render(ev_on, ev_off, evaluation.ablation(ev_on, ev_off), evaluation.render_meta(params))


@pytest.fixture(scope="module")
def text(ev, params):
    return render_pair(ev["on"], ev["no_ego"], params)


def test_two_runs_render_byte_identical_files(runs, ev, params, inputs, text):
    again = render_pair(evaluate_run(runs["on_again"], params, inputs), ev["no_ego"], params)
    assert again == text


def test_render_control_a_different_warm_up_changes_the_file(runs, params, inputs, ev, text):
    other = dataclasses.replace(params, warmup_frames=params.warmup_frames + 1)
    changed = render_pair(evaluate_run(runs["on"], other, inputs), ev["no_ego"], other)
    assert changed != text


def test_the_file_has_no_time_host_or_platform_text(text):
    low = text.lower()
    for word in ("timestamp", "platform", "python 3", "hostname", "ms)", "seconds"):
        assert word not in low, word
    # the only architecture text is the fixed claim about byte identity, the same on every machine
    assert low.count("arm64") == 1 and "verified on arm64 and on x86_64, the latter on a github actions runner" in low
    assert "None" not in text and "nan" not in low.replace("n/a", "")


def test_the_six_parts_come_in_order_each_with_a_reading_note(text):
    lines = text.splitlines()
    at = [lines.index(h) for h in HEADINGS]
    assert at == sorted(at)
    for i in at:
        following = [ln for ln in lines[i + 1:] if ln.strip()][0]
        assert following.startswith(evaluation.READ_PREFIX) and len(following) > len(evaluation.READ_PREFIX) + 40
        assert len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", following[len(evaluation.READ_PREFIX):])) <= 3  # D13: 1-3 sentences


def test_no_number_is_called_a_false_positive_and_part_3_says_consistency_not_correctness(text):
    low = text.lower()
    assert "false positive" not in low and "false-positive" not in low
    part3 = text.split(HEADINGS[3])[1].split(HEADINGS[4])[0]
    assert "consistency, not correctness" in part3
    assert "Excluded (< 10 frames)" in part3  # survivorship is printed beside the result
    assert "probably" in part3 and "nothing here classifies them" in part3  # D15: no claim of what the tracks are
    assert "does not assume that unmatched objects are fixed" in part3 and "including the unmatched frames of tracks" in part3
    assert "directly comparable" in part3  # object size biases the box-center drift of the two rows
    assert "main result" in text.split(HEADINGS[4])[1].split(HEADINGS[5])[0]  # the gate passed on the real drive


def test_every_row_that_is_not_gated_says_so(text):
    assert text.count("no bar by design") >= 10
    gated = [ln for ln in text.splitlines() if "**gated" in ln]
    assert len(gated) == 2  # the velocity RMSE row and the ratio row of part 4
    assert "**Gate (D1): PASS.**" in text


def test_the_summary_quotes_the_phase_3_bars_from_the_same_run(text):
    summary = text.split(HEADINGS[0])[1].split(HEADINGS[1])[0]
    assert "gated in Phase 3 (E3.2)" in summary and "gated in Phase 3 (E3.3)" in summary


# ---------------------------------------------------------------- E4.5 committed samples match a fresh run (D4)

def sample_mismatch(name, fresh, samples_dir=SAMPLES_DIR):
    """None if docs/samples/<name> equals the fresh bytes, else the reason. A missing sample is a mismatch."""
    path = Path(samples_dir) / name
    if not path.is_file():
        return f"{name}: sample not found at {path} (scripts/verify.sh mounts docs/samples read-only at /samples)"
    if path.read_bytes() != fresh:
        return (f"{name}: differs from a fresh run. If you changed the code, regenerate it and copy it to docs/samples/. "
                "Byte identity is verified on arm64 only: on another CPU a small numeric difference is possible and is not "
                "by itself a code defect (see the README, Evaluation)")
    return None


@pytest.fixture(scope="module")
def fresh(runs, text):
    return {"evaluation.md": text.encode(),
            "tracking_speeds.json": (runs["on"]["out_dir"] / "tracking_speeds.json").read_bytes(),
            "tracking_speeds_no_ego.json": (runs["no_ego"]["out_dir"] / "tracking_speeds.json").read_bytes(),
            "detection_recall.json": (runs["on"]["out_dir"] / "detection_recall.json").read_bytes()}


@pytest.mark.parametrize("name", ["evaluation.md", "tracking_speeds.json", "tracking_speeds_no_ego.json", "detection_recall.json"])
def test_the_committed_sample_equals_a_fresh_run(fresh, name):
    assert sample_mismatch(name, fresh[name]) is None


def test_sample_control_one_changed_byte_is_flagged(fresh, tmp_path):
    changed = bytearray(fresh["evaluation.md"])
    changed[len(changed) // 2] ^= 1
    (tmp_path / "evaluation.md").write_bytes(bytes(changed))
    assert "differs from a fresh run" in sample_mismatch("evaluation.md", fresh["evaluation.md"], tmp_path)


def test_sample_control_a_missing_sample_fails_and_names_the_mount(fresh, tmp_path):
    msg = sample_mismatch("evaluation.md", fresh["evaluation.md"], tmp_path)
    assert msg is not None and "/samples" in msg and "verify.sh" in msg
