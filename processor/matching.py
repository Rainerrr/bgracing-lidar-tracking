"""Ground-truth matching and the detection recall summary. ROS-free (numpy, scipy). Reused by the evaluation.

Per frame: one-to-one Hungarian assignment on 2D center distance with a 2.0 m gate. Ground truth of
every class in the evaluation region takes part in the assignment, so a detection on a pedestrian
cannot be credited to a car. Detections are not region-filtered for recall.
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

EVAL_GATE_M = 2.0
CAR_VAN = ("Car", "Van")
SMALL_SAMPLE_TRACKLETS = 3  # classes with fewer tracklets are flagged as small samples
_BIG = 1e6  # gated pairs get a large finite cost: linear_sum_assignment raises on infeasible inf matrices


def in_eval_region(x, y, z, p):
    """Evaluation region: the processing ROI, minus the ego box, restricted to the front half (x >= 0)."""
    x, y, z = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)
    inside = ((x >= max(p.roi_x[0], 0.0)) & (x <= p.roi_x[1]) & (y >= p.roi_y[0]) & (y <= p.roi_y[1])
              & (z >= p.roi_z[0]) & (z <= p.roi_z[1]))
    return inside & ~((np.abs(x) < p.ego_box[0]) & (np.abs(y) < p.ego_box[1]))


def match(gt_xy, det_xy, gate=EVAL_GATE_M):
    """Optimal one-to-one matching. Returns [(gt_index, det_index, distance)] with distance <= gate."""
    gt_xy, det_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2), np.asarray(det_xy, dtype=np.float64).reshape(-1, 2)
    if len(gt_xy) == 0 or len(det_xy) == 0:
        return []
    dist = np.hypot(gt_xy[:, None, 0] - det_xy[None, :, 0], gt_xy[:, None, 1] - det_xy[None, :, 1])
    rows, cols = linear_sum_assignment(np.where(dist <= gate, dist, _BIG))
    return [(int(r), int(c), float(dist[r, c])) for r, c in zip(rows, cols) if dist[r, c] <= gate]


def _r(x, nd=4):
    return round(float(x), nd)


def score_detections(dets_by_frame, gt_boxes, p, n_frames=None):
    """Recall summary of detections against ground-truth boxes (core.kitti.GtBox rows).

    dets_by_frame: {frame: sequence of objects with .x and .y}. Frames without detections may be absent.
    """
    gt_by_frame = defaultdict(list)
    for b in gt_boxes:
        if bool(in_eval_region(b.x, b.y, b.z, p)):
            gt_by_frame[b.frame].append(b)

    matched = defaultdict(int)
    total = defaultdict(int)
    tracklet = {}  # id -> [cls, matched, total]
    errors = []
    unmatched_per_frame = []
    frames = sorted(set(gt_by_frame) | set(dets_by_frame) | set(range(n_frames or 0)))
    for f in frames:
        gts, dets = gt_by_frame.get(f, []), list(dets_by_frame.get(f, ()))
        pairs = match([(g.x, g.y) for g in gts], [(d.x, d.y) for d in dets])
        hit = {gi for gi, _, _ in pairs}
        for gi, g in enumerate(gts):
            total[g.cls] += 1
            matched[g.cls] += gi in hit
            t = tracklet.setdefault(g.id, [g.cls, 0, 0])
            t[1] += gi in hit
            t[2] += 1
        errors += [d for _, _, d in pairs]
        in_region = [d for d in dets if bool(in_eval_region(d.x, d.y, d.z, p))]
        unmatched_per_frame.append(len(in_region) - sum(1 for _, di, _ in pairs if bool(
            in_eval_region(dets[di].x, dets[di].y, dets[di].z, p))))

    if not gt_by_frame:
        return {"ground_truth": "missing"}

    cv_total = sum(total.get(c, 0) for c in CAR_VAN)
    cv_matched = sum(matched.get(c, 0) for c in CAR_VAN)
    per_class = {}
    for cls in sorted(total):
        ids = sorted(i for i, t in tracklet.items() if t[0] == cls)
        per_class[cls] = {"matched": matched[cls], "total": total[cls], "recall": _r(matched[cls] / total[cls]),
                          "tracklets": len(ids), "small_sample": len(ids) < SMALL_SAMPLE_TRACKLETS}
    return {
        "ground_truth": "present",
        "gate_m": EVAL_GATE_M,
        "car_van": {"matched": cv_matched, "total": cv_total, "recall": _r(cv_matched / cv_total) if cv_total else 0.0},
        "per_class": per_class,
        "per_tracklet": {str(i): {"cls": t[0], "matched": t[1], "total": t[2], "recall": _r(t[1] / t[2])}
                         for i, t in sorted(tracklet.items()) if t[0] in CAR_VAN},
        "center_error_m": {"mean": _r(np.mean(errors)) if errors else None,
                           "p90": _r(np.percentile(errors, 90)) if errors else None, "n": len(errors)},
        "unmatched_detections_in_region": {
            "total": int(sum(unmatched_per_frame)),
            "per_frame_mean": _r(np.mean(unmatched_per_frame)) if unmatched_per_frame else 0.0,
            "per_frame_max": int(max(unmatched_per_frame)) if unmatched_per_frame else 0,
            "note": "detections in the evaluation region with no ground-truth match; not necessarily false",
        },
    }
