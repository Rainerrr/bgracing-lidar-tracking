"""The evaluation metric functions on hand-built cases (no data, no ROS). Identity poses unless stated.

Scene: GT tracklets are static boxes in front of the car (x = 10 m, inside the evaluation region), tracked by
confirmed tracks at the same place. warmup_frames is 5 (config), so frames 0-4 of a track are warm-up.
"""
import dataclasses

import numpy as np
import pytest

from core.kitti import GtBox
from processor import evaluation

N = 20
STEP_NS = 100_000_000


def row(f, tid, x, y, vx=0.0, vy=0.0, state="static"):
    return f"0,{f},{f * STEP_NS},{tid},{x},{y},{vx},{vy},{np.hypot(vx, vy)},{state},4,2,1.5"


def gt(i, cls, x, y, frames=range(N), z=-1.0):
    return [GtBox(f, i, cls, x, y, z, 4.0, 1.8, 1.5, 0.0) for f in frames]


def still(tid, x, y, frames=range(N), **kw):
    return [row(f, tid, x, y, **kw) for f in frames]


def run(tracks, gts, params, n=N, T=None, dets=None):
    T_wv = T if T is not None else np.tile(np.eye(4), (n, 1, 1))
    return evaluation.evaluate(tracks, dets or {}, gts, T_wv, np.arange(n) * STEP_NS, params, n)


@pytest.fixture
def car():
    return gt(0, "Car", 10.0, 0.0)


# ---------------------------------------------------------------- perfect match, miss, false track

def test_perfect_match(params, car):
    e = run(still(1, 10.0, 0.0), car, params)
    assert e["recall"]["overall"]["value"] == 1.0 and e["recall"]["missed"] == 0
    assert e["coverage"]["value"] == 1.0
    assert e["center_error"]["overall"]["value"] == 0.0
    assert e["id_switches"]["total"] == 0
    assert e["velocity"]["rmse_static"]["value"] == 0.0 and e["velocity"]["rmse_static"]["n"] == N - params.warmup_frames
    assert e["labels"]["static_accuracy"]["value"] == 1.0 and e["labels"]["flicker"]["value"] == 0.0


def test_one_miss_lowers_recall_by_one_over_n_and_leaves_coverage(params, car):
    e = run([r for r in still(1, 10.0, 0.0) if not r.startswith("0,10,")], car, params)
    assert e["recall"]["overall"]["num"] == N - 1 and e["recall"]["overall"]["den"] == N
    assert e["recall"]["overall"]["value"] == pytest.approx(1 - 1 / N) and e["recall"]["missed"] == 1
    assert e["coverage"]["value"] == 1.0


def test_one_false_track_inside_the_region_lowers_coverage_by_one_over_n(params, car):
    e = run(still(1, 10.0, 0.0) + [row(3, 2, 20.0, 5.0)], car, params)
    assert (e["coverage"]["num"], e["coverage"]["den"]) == (N, N + 1)
    assert e["unlabelled_track_frames"] == 1 and e["produced"]["unlabelled_tracks"] == 1


def test_a_false_moving_track_raises_the_false_moving_rate_but_only_after_warm_up(params, car):
    base = still(1, 10.0, 0.0)
    # frame 3 is warm-up for track 2 (its first frame is 3): not counted; frame 10 is post-warm-up
    e = run(base + [row(3, 2, 20.0, 5.0, state="moving")] + [row(f, 2, 20.0, 5.0, state="moving") for f in range(4, 20)],
            car, params)
    fm = e["labels"]["false_moving"]
    assert fm["den"] == 20 - 3 - params.warmup_frames + 0 and fm["value"] == 1.0  # rows 8..19 of track 2: frame - 3 >= 5
    calm = run(base + still(2, 20.0, 5.0), car, params)
    assert calm["labels"]["false_moving"]["value"] == 0.0


def test_a_false_track_outside_the_region_changes_nothing(params, car):
    e = run(still(1, 10.0, 0.0) + [row(3, 2, -30.0, 0.0), row(4, 2, 10.0, 30.0)], car, params)
    assert (e["coverage"]["num"], e["coverage"]["den"]) == (N, N)
    assert e["produced"]["tracks"] == 2 and e["produced"]["tracks_in_region"] == 1


# ---------------------------------------------------------------- ID switches

def test_one_relabelled_track_id_is_one_switch(params, car):
    rows = still(1, 10.0, 0.0, range(10)) + still(3, 10.0, 0.0, range(10, N))
    e = run(rows, car, params)
    assert e["id_switches"]["total"] == 1 and e["id_switches"]["per_tracklet"][0] == 1


def test_a_switch_across_an_unmatched_gap_still_counts(params, car):
    rows = still(1, 10.0, 0.0, range(8)) + still(3, 10.0, 0.0, range(12, N))
    e = run(rows, car, params)
    assert e["id_switches"]["total"] == 1  # frames 8-11 are unmatched; the memory is kept across them


def test_a_gap_without_a_new_id_is_not_a_switch(params, car):
    e = run(still(1, 10.0, 0.0, range(8)) + still(1, 10.0, 0.0, range(12, N)), car, params)
    assert e["id_switches"]["total"] == 0


def test_two_tracklets_with_two_tracks_are_never_a_switch(params, car):
    e = run(still(1, 10.0, 0.0) + still(2, 10.0, 8.0), car + gt(1, "Car", 10.0, 8.0), params)
    assert e["id_switches"]["total"] == 0


# ---------------------------------------------------------------- velocity, warm-up scope, flicker

def test_a_one_metre_per_second_bias_gives_an_rmse_of_one(params, car):
    e = run(still(1, 10.0, 0.0, vx=1.0), car, params)
    assert e["velocity"]["rmse_static"]["value"] == pytest.approx(1.0)
    assert e["velocity"]["rmse_moving"]["value"] is None  # no moving GT in the scene: n/a, not 0


def test_warm_up_frames_are_excluded_from_velocity_and_labels_but_not_from_recall_or_coverage(params, car):
    rows = [row(f, 1, 10.0, 0.0, vx=5.0 if f < params.warmup_frames else 0.0,
                state="moving" if f < params.warmup_frames else "static") for f in range(N)]
    e = run(rows, car, params)
    assert e["velocity"]["rmse_static"]["value"] == 0.0 and e["labels"]["flicker"]["value"] == 0.0
    assert e["recall"]["overall"]["num"] == N and e["coverage"]["den"] == N  # warm-up frames still count


def test_flicker_and_static_accuracy_are_complements(params, car):
    rows = [row(f, 1, 10.0, 0.0, state="moving" if f % 4 == 0 else "static") for f in range(N)]
    e = run(rows, car, params)
    lab = e["labels"]
    assert lab["flicker"]["value"] > 0
    assert lab["static_accuracy"]["value"] + lab["flicker"]["value"] == pytest.approx(1.0)


def test_moving_label_recall_counts_moving_gt_only(params):
    # a GT car driving at 5 m/s along x, tracked with the right velocity but labelled moving half the time
    moving = [GtBox(f, 0, "Car", 10.0 + 0.5 * f, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0) for f in range(N)]
    rows = [row(f, 1, 10.0 + 0.5 * f, 0.0, vx=5.0, state="moving" if f % 2 == 0 else "static") for f in range(N)]
    e = run(rows, moving, params)
    assert e["gt"]["static_tracklets"] == []
    assert 0 < e["labels"]["moving_label_recall"]["value"] < 1
    assert e["labels"]["static_accuracy"]["value"] is None and e["velocity"]["rmse_moving"]["value"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- frames

def test_tracks_in_the_world_frame_map_back_to_the_velodyne_region(params, car):
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, 0, 3] = 100.0  # the sensor sits at world x = 100
    inside = still(1, 110.0, 0.0)  # = velodyne x 10: matches the car
    far = [row(3, 2, 210.0, 0.0)]  # = velodyne x 110: outside the region
    e = run(inside + far, car, params, T=T)
    assert (e["coverage"]["num"], e["coverage"]["den"]) == (N, N) and e["recall"]["overall"]["value"] == 1.0
    assert e["tracking_frame"] == "world"


def test_without_ego_compensation_tracks_are_already_in_velodyne_and_gt_velocity_is_rotated(params, car):
    off = dataclasses.replace(params, ego_compensation=False)
    e = run(still(1, 10.0, 0.0), car, off)
    assert e["tracking_frame"] == "velodyne" and e["recall"]["overall"]["value"] == 1.0
    # the tracker in the velodyne frame sees a parked car streaming past at 5 m/s: that is the error the uncompensated velocity RMSE measures
    rows = still(1, 10.0, 0.0, vx=-5.0)
    assert run(rows, car, off)["velocity"]["rmse_static"]["value"] == pytest.approx(5.0)


def test_without_ego_compensation_the_gt_velocity_is_rotated_into_velodyne_axes(params):
    """The ego is turned 90 degrees (velodyne x = world y). A GT car moving 5 m/s east in the world frame streams
    along -y in the velodyne axes; a tracker reading exactly that has zero error only if GT is rotated into velodyne axes."""
    off = dataclasses.replace(params, ego_compensation=False)
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, :2, :2] = [[0.0, -1.0], [1.0, 0.0]]  # R = Rz(+90): p_world = R p_velodyne
    car_y = [GtBox(f, 0, "Car", 10.0, 5.0 - 0.5 * f, -1.0, 4.0, 1.8, 1.5, 0.0) for f in range(N)]
    rows = [row(f, 1, 10.0, 5.0 - 0.5 * f, vx=0.0, vy=-5.0, state="moving") for f in range(N)]
    e = run(rows, car_y, off, T=T)
    assert e["gt"]["static_tracklets"] == [] and e["velocity"]["rmse_moving"]["n"] == N - off.warmup_frames
    assert e["velocity"]["rmse_moving"]["value"] == pytest.approx(0.0, abs=1e-6)
    wrong = [row(f, 1, 10.0, 5.0 - 0.5 * f, vx=5.0, vy=0.0, state="moving") for f in range(N)]  # world axes
    assert run(wrong, car_y, off, T=T)["velocity"]["rmse_moving"]["value"] == pytest.approx(np.hypot(5.0, 5.0), abs=1e-6)


# ---------------------------------------------------------------- world stillness

def test_world_stillness_counts_scored_and_excluded_unlabelled_tracks(params, car):
    far_still = still(2, 20.0, 10.0, range(15))  # post-warm-up rows: frames 5-14 = 10, scored
    drifting = [row(f, 3, 20.0 if f < 10 else 22.0, -10.0) for f in range(15)]  # mean 21 m after warm-up, spread 1.0
    short = still(4, 25.0, 10.0, range(14))  # 9 post-warm-up rows: excluded, and counted as excluded
    once = [row(3, 5, 10.0, 0.0, state="static")]  # one frame matched to the car: labelled, not unlabelled
    # track 1 skips frame 3, so the car is matched there by track 5 (two tracks on one box would share the match)
    e = run(still(1, 10.0, 0.0, [f for f in range(N) if f != 3]) + far_still + drifting + short + once, car, params)
    s = e["stillness"]["unlabelled"]
    assert (s["tracks"], s["scored"], s["excluded"]) == (3, 2, 1)
    assert s["median_spread"] == pytest.approx(0.5)  # medians of {0.0, 1.0}
    assert s["median_speed"] == 0.0
    assert e["produced"]["labelled_tracks"] == 2 and e["produced"]["unlabelled_tracks"] == 3
    # both labelled tracks match the parked car; track 5 has one row, so only track 1 is scored in the reference
    ref = e["stillness"]["parked"]
    assert (ref["tracks"], ref["scored"], ref["excluded"]) == (2, 1, 1) and ref["median_spread"] == 0.0


def test_a_track_matched_in_a_single_frame_is_labelled(params, car):
    e = run(still(1, 10.0, 0.0, range(1)), car, params)
    assert e["produced"]["labelled_tracks"] == 1 and e["produced"]["unlabelled_tracks"] == 0


# ---------------------------------------------------------------- degenerate input

def test_no_tracks_give_zero_recall_and_na_elsewhere_without_a_crash(params, car):
    e = run([], car, params)
    assert e["recall"]["overall"]["value"] == 0.0 and e["recall"]["missed"] == N
    assert e["coverage"]["value"] is None and e["coverage"]["den_name"]
    assert e["velocity"]["rmse_static"]["value"] is None and e["labels"]["false_moving"]["value"] is None
    assert e["center_error"]["overall"]["value"] is None and e["id_switches"]["total"] == 0
    assert e["produced"]["tracks"] == 0 and e["stillness"]["unlabelled"]["tracks"] == 0


def test_a_second_pass_is_rejected(params, car):
    with pytest.raises(ValueError, match="one offline pass"):
        run(["1," + still(1, 10.0, 0.0)[0].split(",", 1)[1]], car, params)


def test_missing_ground_truth_is_rejected(params):
    with pytest.raises(ValueError, match="ground truth"):
        run(still(1, 10.0, 0.0), [], params)


def test_per_class_rows_flag_small_samples_and_count_missed(params, car):
    ped = gt(1, "Pedestrian", 10.0, 8.0)
    e = run(still(1, 10.0, 0.0), car + ped, params)
    pc = e["recall"]["per_class"]
    assert pc["Pedestrian"]["small_sample"] and pc["Pedestrian"]["missed"] == N and pc["Pedestrian"]["value"] == 0.0
    assert pc["Car"]["num"] == N
    assert e["recall"]["overall"]["den"] == pc["Car"]["den"] + pc["Pedestrian"]["den"]
    for c in pc.values():
        assert c["num"] + c["missed"] == c["den"]  # consistency: matched + missed = GT box-frames in the region


# ---------------------------------------------------------------- the ablation gate

def _eval_with(rmse):
    return {"velocity": {"rmse_static": {"value": rmse}}}


def test_ablation_gate_passes_at_exactly_three_times_and_fails_below():
    assert evaluation.ablation(_eval_with(0.5), _eval_with(1.5))["pass"]  # exactly 3.0 in binary floating point
    assert not evaluation.ablation(_eval_with(0.5), _eval_with(1.49))["pass"]


def test_ablation_control_the_same_run_twice_fails_the_gate():
    abl = evaluation.ablation(_eval_with(0.4), _eval_with(0.4))
    assert abl["ratio"] == pytest.approx(1.0) and not abl["pass"]


def test_ablation_with_no_measurable_rmse_fails_instead_of_dividing_by_zero():
    assert not evaluation.ablation(_eval_with(None), _eval_with(5.0))["pass"]
    assert not evaluation.ablation(_eval_with(0.0), _eval_with(5.0))["pass"]


# ---------------------------------------------------------------- rendering on degenerate input

def test_an_empty_run_renders_a_complete_file_with_named_zero_denominators(params, car):
    off = dataclasses.replace(params, ego_compensation=False)
    on_e, off_e = run([], car, params), run([], car, off)
    abl = evaluation.ablation(on_e, off_e)
    text = evaluation.render(on_e, off_e, abl, evaluation.render_meta(params))
    assert [h for h in text.splitlines() if h.startswith("## ")] == ["## " + h for h in evaluation.PARTS]
    assert "**Gate: FAIL.**" in text
    assert "n/a (confirmed track-frames in the region = 0)" in text
    assert "None" not in text and "nan" not in text.lower().replace("n/a", "")
    assert text == evaluation.render(on_e, off_e, abl, evaluation.render_meta(params))  # still deterministic


# ---------------------------------------------------------------- edge cases

def test_the_track_to_velodyne_mapping_uses_the_transpose_of_the_rotation(params, car):
    """Ego turned 90 degrees at world (100, 0): p_world = Rz(90) p_velodyne + t. The car at velodyne (10, 0) is at
    world (100, 10); world (100, -10) is velodyne (-10, 0), behind the car, outside the front half. A mapping that
    applied R instead of R^T would put that track at velodyne (10, 0) and count it."""
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, :2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    T[:, 0, 3] = 100.0
    e = run(still(1, 100.0, 10.0) + [row(3, 2, 100.0, -10.0)], car, params, T=T)
    assert (e["coverage"]["num"], e["coverage"]["den"]) == (N, N)
    assert e["produced"]["tracks"] == 2 and e["produced"]["tracks_in_region"] == 1


def test_the_result_does_not_depend_on_the_row_order(params, car):
    rows = still(1, 10.0, 0.0) + still(2, 20.0, 5.0)
    a, b = run(rows, car, params), run(list(reversed(rows)), car, params)
    assert a["labels"] == b["labels"] and a["stillness"] == b["stillness"] and a["coverage"] == b["coverage"]


def test_the_parked_summary_is_n_a_unless_all_parked_tracklets_were_scored():
    parked = [3, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    full = {"gt": {"static_tracklets": parked, "moving_tracklets": []},
            "tracklets": {i: {"median_speed": 0.2, "median_abs_error": 0.1} for i in parked}}
    mom, worst, _ = evaluation._speed_summary(full)
    assert mom == 0.2 and worst == 0.2
    partial = {"gt": {"static_tracklets": parked, "moving_tracklets": []},
               "tracklets": {3: {"median_speed": 0.2, "median_abs_error": None}}}
    assert evaluation._speed_summary(partial)[:2] == (None, None)  # one tracklet must not stand in for ten
    none = {"gt": {"static_tracklets": [], "moving_tracklets": []}, "tracklets": {}}
    assert evaluation._speed_summary(none)[:2] == (None, None)  # no parked tracklet: n/a, never a value


def test_the_part_4_note_does_not_claim_the_result_when_the_gate_fails(params, car):
    off = dataclasses.replace(params, ego_compensation=False)
    on_e, off_e = run([], car, params), run([], car, off)
    text = evaluation.render(on_e, off_e, evaluation.ablation(on_e, off_e), evaluation.render_meta(params))
    part4 = text.split("## " + evaluation.PARTS[4])[1].split("## " + evaluation.PARTS[5])[0]
    assert "The gate failed in this run" in part4 and "main result" not in part4


# ---------------------------------------------------------------- drive-specific text comes from the drive

def _render(on_e, off_e, params, ego_speed=None):
    return evaluation.render(on_e, off_e, evaluation.ablation(on_e, off_e), evaluation.render_meta(params, ego_speed))


def test_no_moving_tracklet_renders_a_named_n_a_row(params, car):
    off = dataclasses.replace(params, ego_compensation=False)
    on_e, off_e = run(still(1, 10.0, 0.0), car, params), run(still(1, 10.0, 0.0), car, off)
    assert on_e["gt"]["moving_tracklets"] == []
    text = _render(on_e, off_e, params)
    assert "| Moving tracklets: none meets the rule | n/a (moving tracklets = 0) |" in text
    assert "below 0.5 m/s (the 1 parked tracklet); every" in text


def test_the_moving_row_names_the_selected_tracklets_and_their_class(params, car):
    walker = [GtBox(f, 4, "Cyclist", 10.0 + 0.2 * f, 5.0, -1.0, 1.8, 0.6, 1.7, 0.0) for f in range(N)]  # 2 m/s, 20 frames
    off = dataclasses.replace(params, ego_compensation=False)
    on_e, off_e = run([], car + walker, params), run([], car + walker, off)
    assert on_e["gt"]["moving_tracklets"] == [4]
    assert "| Moving tracklets #4 Cyclist: median speed error (m/s) |" in _render(on_e, off_e, params)


def test_the_ego_speed_range_comes_from_the_oxts_and_is_left_out_when_unknown(params, car):
    oxts = np.zeros((3, 30))
    oxts[:, 6], oxts[:, 7] = [0.0, 3.0, 0.0], [0.0, 4.0, 6.5]  # north, east: 0, 5 and 6.5 m/s
    assert evaluation.ego_speed_range(oxts) == (0.0, 6.5)
    off = dataclasses.replace(params, ego_compensation=False)
    on_e, off_e = run([], car, params), run([], car, off)
    assert "at about the ego speed (0.0 to 6.5 m/s), whether" in _render(on_e, off_e, params, (0.0, 6.5))
    assert "at about the ego speed, whether" in _render(on_e, off_e, params)


def test_no_parked_tracklet_is_said_in_words(params):
    mover = [GtBox(f, 2, "Car", 10.0 + 0.2 * f, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0) for f in range(N)]
    off = dataclasses.replace(params, ego_compensation=False)
    text = _render(run([], mover, params), run([], mover, off), params)
    assert "below 0.5 m/s (no parked tracklets); every" in text and "None" not in text


def test_a_parked_tracklet_that_was_never_scored_is_named_as_the_reason_for_n_a(params, car):
    """Two parked tracklets, one tracked and one never matched: the summary says which reason, not 'zero tracklets'."""
    other = gt(5, "Car", 30.0, 8.0)
    off = dataclasses.replace(params, ego_compensation=False)
    on_e, off_e = run(still(1, 10.0, 0.0), car + other, params), run(still(1, 10.0, 0.0), car + other, off)
    assert on_e["gt"]["static_tracklets"] == [0, 5]
    text = _render(on_e, off_e, params)
    assert "| n/a (1 of 2 parked tracklets not scored) |" in text and "parked tracklets = 0" not in text


def test_the_printed_command_reproduces_the_drive_it_describes(params):
    other = dataclasses.replace(params, drive="2011_09_26_drive_0011")
    assert evaluation.render_meta(params, None, params.drive)["command"] == evaluation.COMMAND  # the config's drive: unchanged
    assert evaluation.render_meta(params)["command"] == evaluation.COMMAND
    cmd = evaluation.render_meta(other, None, params.drive)["command"]
    assert cmd == evaluation.COMMAND + " --set drive=2011_09_26_drive_0011"


def test_another_drive_does_not_claim_the_x86_check(params, car):
    off = dataclasses.replace(params, ego_compensation=False)
    on_e, off_e = run([], car, params), run([], car, off)
    other = dataclasses.replace(params, drive="2011_09_26_drive_0011")
    text = evaluation.render(on_e, off_e, evaluation.ablation(on_e, off_e), evaluation.render_meta(other, None, params.drive))
    assert "verified on arm64; not checked on x86_64" in text and "GitHub Actions" not in text
