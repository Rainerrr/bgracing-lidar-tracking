"""E2.2 matching function on hand-built cases (shared with Phase 4)."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core.kitti import GtBox
from processor import matching as mt


def gt(frame, id, cls, x, y, z=-0.9):
    return GtBox(frame, id, cls, x, y, z, 4.0, 1.8, 1.5, 0.0)


def det(x, y, z=-0.9):
    return NS(x=x, y=y, z=z)


def test_perfect_match():
    pairs = mt.match([(10, 0), (20, 5)], [(20.1, 5.0), (10.0, 0.2)])
    assert sorted((g, d) for g, d, _ in pairs) == [(0, 1), (1, 0)]


def test_one_miss_and_one_extra():
    pairs = mt.match([(10, 0), (20, 5)], [(10.2, 0.0)])
    assert [(g, d) for g, d, _ in pairs] == [(0, 0)]
    assert mt.match([(10, 0)], [(10, 0), (30, 9)]) == [(0, 0, 0.0)]


def test_pair_beyond_the_gate_is_unmatched_and_the_gate_is_inclusive():
    assert mt.match([(0, 0)], [(2.01, 0)]) == []
    assert len(mt.match([(0, 0)], [(2.0, 0)])) == 1


def test_hungarian_differs_from_greedy_nearest_neighbor():
    # g0's nearest detection is d0 (1.0 vs 1.05), but g1 can only use d0 (1.9; d1 is 3.95 away):
    # greedy in GT order takes d0 for g0 and strands g1 (1 pair); the optimal assignment finds 2
    gts, dets = [(0.0, 0.0), (2.9, 0.0)], [(1.0, 0.0), (-1.05, 0.0)]
    pairs = mt.match(gts, dets)
    assert len(pairs) == 2 and sorted((g, d) for g, d, _ in pairs) == [(0, 1), (1, 0)]
    greedy = {}
    for gi, g in enumerate(gts):
        d = min((i for i in range(len(dets)) if i not in greedy.values()), key=lambda i: np.hypot(dets[i][0] - g[0], dets[i][1] - g[1]))
        if np.hypot(dets[d][0] - g[0], dets[d][1] - g[1]) <= 2.0:
            greedy[gi] = d
    assert len(greedy) < 2


def test_gt_row_without_a_detection_inside_the_gate_does_not_raise():
    assert mt.match([(0, 0), (50, 50)], [(0.5, 0)]) == [(0, 0, 0.5)]
    assert mt.match([(0, 0)], [(50, 50), (60, 60)]) == []
    assert mt.match([], [(0, 0)]) == [] and mt.match([(0, 0)], []) == []


def test_eval_region(params):
    xs = np.array([10.0, -5.0, 1.0, 45.0, 10.0, 10.0])
    ys = np.array([0.0, 0.0, 0.5, 0.0, 25.0, 0.0])
    zs = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 3.0])
    assert mt.in_eval_region(xs, ys, zs, params).tolist() == [True, False, False, False, False, False]


def test_score_perfect_and_empty(params):
    boxes = [gt(0, 0, "Car", 10, 0), gt(0, 1, "Van", 20, 4), gt(1, 0, "Car", 10.5, 0), gt(1, 1, "Van", 20.5, 4)]
    dets = {0: [det(10, 0), det(20, 4)], 1: [det(10.5, 0), det(20.5, 4)]}
    s = mt.score_detections(dets, boxes, params)
    assert s["car_van"] == {"matched": 4, "total": 4, "recall": 1.0}
    assert s["per_class"]["Car"]["tracklets"] == 1 and s["per_class"]["Car"]["small_sample"]
    assert s["unmatched_detections_in_region"]["total"] == 0
    s0 = mt.score_detections({}, boxes, params)
    assert s0["car_van"]["recall"] == 0.0 and s0["car_van"]["total"] == 4


def test_score_counts_one_miss_and_extra_detections(params):
    boxes = [gt(0, 0, "Car", 10, 0), gt(0, 1, "Car", 20, 4)]
    s = mt.score_detections({0: [det(10, 0), det(30, -8)]}, boxes, params)
    assert s["car_van"]["matched"] == 1 and s["car_van"]["total"] == 2 and s["car_van"]["recall"] == 0.5
    assert s["unmatched_detections_in_region"]["total"] == 1
    assert s["per_tracklet"]["0"]["recall"] == 1.0 and s["per_tracklet"]["1"]["recall"] == 0.0


def test_a_pedestrian_nearer_to_the_detection_takes_the_match(params):
    boxes = [gt(0, 0, "Car", 10, 0), gt(0, 1, "Pedestrian", 10.3, 0)]
    s = mt.score_detections({0: [det(10.2, 0)]}, boxes, params)
    assert s["car_van"]["matched"] == 0, "the detection belongs to the nearer pedestrian, not to the car"
    assert s["per_class"]["Pedestrian"]["matched"] == 1


def test_gt_outside_the_region_is_not_counted_and_missing_gt_is_reported(params):
    s = mt.score_detections({}, [gt(0, 0, "Car", -10, 0), gt(0, 1, "Car", 60, 0)], params)
    assert s == {"ground_truth": "missing"}
    s = mt.score_detections({0: [det(10, 0)]}, [gt(0, 0, "Car", -10, 0), gt(0, 1, "Car", 10, 0)], params)
    assert s["car_van"]["total"] == 1


def test_detections_outside_the_region_are_not_counted_as_unmatched(params):
    s = mt.score_detections({0: [det(10, 0), det(-10, 0), det(70, 0)]}, [gt(0, 0, "Car", 10, 0)], params)
    assert s["unmatched_detections_in_region"]["total"] == 0
