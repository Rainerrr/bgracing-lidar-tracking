"""The parked and moving tracklet rules and the speed scores on empty sets, on hand-built ground truth (no data, no ROS).

Identity poses, 10 Hz. GT boxes in front of the car (x >= 10 m) are inside the evaluation region; x = -10 m is behind
it, outside the region.
"""
import json
import warnings

import numpy as np
import pytest

from core.kitti import GtBox
from processor import track_scoring as ts

STEP_NS = 100_000_000


def boxes(i, x0, speed, frames, cls="Car", y=0.0):
    """A tracklet moving along x at `speed` m/s over `frames`."""
    return [GtBox(f, i, cls, x0 + speed * f * STEP_NS * 1e-9, y, -1.0, 4.0, 1.8, 1.5, 0.0) for f in frames]


def row(f, tid, x, y=0.0, speed=0.0, state="static"):
    return f"0,{f},{f * STEP_NS},{tid},{x},{y},{speed},0.0,{speed},{state},4,2,1.5"


def score(gt, rows=(), n=40, p=None):
    return ts.score(list(rows), gt, np.tile(np.eye(4), (n, 1, 1)), np.arange(n) * STEP_NS, p)


# ---------------------------------------------------------------- the rules

def test_the_parked_rule_is_a_net_speed_below_half_a_metre_per_second_and_one_frame_in_the_region():
    net = {1: 0.0, 2: 0.49, 3: 0.5, 4: 3.0, 5: 0.0}
    counts = {1: 1, 2: 50, 3: 50, 4: 50, 5: 0}
    assert ts.parked_tracklets(net, counts) == (1, 2)  # 5 is never in the region, 3 and 4 move


def test_a_parked_tracklet_never_in_the_region_does_not_fail_the_parked_test(params):
    gt = boxes(0, 10.0, 0.0, range(30)) + boxes(2, -10.0, 0.0, range(30))  # #2 is parked behind the car
    s = score(gt, [row(f, 7, 10.0) for f in range(30)], p=params)
    assert list(s["parked"]["tracklets"]) == ["0"] and s["parked"]["pass"] is True


def test_the_moving_rule_needs_the_net_speed_and_twenty_box_frames_in_the_region():
    net = {1: 0.49, 2: 0.5, 3: 0.5, 4: 6.0, 5: 6.0}
    counts = {1: 100, 2: 20, 3: 19, 4: 20, 5: 0}
    assert ts.MOVING_MIN_BOX_FRAMES == 20
    assert ts.moving_tracklets(net, counts) == (2, 4)  # 19 box-frames or a parked net speed are left out
    assert ts.moving_tracklets(net, {}) == ()


def test_only_box_frames_inside_the_evaluation_region_count(params):
    gt = boxes(1, 10.0, 0.0, range(15)) + boxes(1, -10.0, 0.0, range(15, 30))
    assert ts.region_box_frames(gt, params) == {1: 15}


@pytest.mark.parametrize("n_frames, selected", [(19, False), (20, True)])
def test_score_selects_a_moving_tracklet_from_twenty_in_region_box_frames(params, n_frames, selected):
    gt = boxes(0, 10.0, 0.0, range(30)) + boxes(1, 10.0, 2.0, range(n_frames))
    s = score(gt, [row(f, 7, 10.0) for f in range(30)], p=params)
    assert list(s["parked"]["tracklets"]) == ["0"]
    assert ("1" in s["moving"]["objects"]) is selected


# ---------------------------------------------------------------- empty sets never pass

def test_no_moving_tracklet_fails_the_moving_test_instead_of_passing_it(params):
    s = score(boxes(0, 10.0, 0.0, range(30)), [row(f, 7, 10.0) for f in range(30)], p=params)
    assert s["moving"]["objects"] == {} and s["moving"]["pass"] is False
    assert s["parked"]["pass"] is True  # the parked car is measured still, so only the empty set fails


def test_no_parked_tracklet_fails_the_parked_test_with_null_values(params):
    gt = boxes(1, 10.0, 2.0, range(30))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no numpy "mean of empty slice"
        s = score(gt, p=params)
    pk = s["parked"]
    assert pk["tracklets"] == {} and pk["median_of_medians"] is None and pk["pass"] is False
    assert pk["worst"] == {"id": None, "median_speed": None}
    json.dumps(s)  # still serialisable as tracking_speeds.json
