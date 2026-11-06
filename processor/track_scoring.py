"""Speed scoring of tracks against KITTI tracklets (E3.2, E3.3; reused by Phase 4). ROS-free.

Method (phase-3 FEATURE_SPEC D1, D2 and the engineering decisions):
- GT velocity: centered difference of the tracklet's world-frame centers over +-GT_HALF_WINDOW frames (1 s),
  shortened at the tracklet's ends. Differencing over +-1 frame makes parked cars read 0.24-0.45 m/s from
  label jitter alone (EVIDENCE/pre-spec-speeds.txt).
- Per frame, confirmed tracks are matched one-to-one to the GT boxes in the evaluation region (processing ROI
  and x >= 0, tested on the GT center in velodyne) with matching.match (Hungarian, 2.0 m gate), on centers in
  the tracking frame: world, or velodyne when ego compensation is off.
- Results are aggregated by tracklet, never by track ID, so an ID switch doesn't split a tracklet.
- A track's first warmup_frames frames after confirmation are excluded. A track appears in tracks.csv from
  the frame it is confirmed, so its first row is its confirmation frame.
"""
from collections import defaultdict

import numpy as np

from core import poses
from processor import matching

GT_HALF_WINDOW = 5
PARKED = (3, 5, 6, 7, 8, 9, 10, 11, 13, 14)  # the E1.2 fixture: world net speed < 0.5 m/s
MOVING = (0, 1)  # lead van, cyclist
PARKED_BAR_MEDIAN = 0.5  # D1: median over the parked tracklets of each tracklet's median speed (m/s)
PARKED_BAR_EACH = 0.7  # D1: every parked tracklet's median speed; = moving_speed
MOVING_BAR = 0.5  # D2: per object, median |filtered speed - GT speed| (m/s)
SMALL_SAMPLE_CLASSES = ("Pedestrian", "Cyclist")


def parse_track_rows(rows):
    """tracks.csv data lines (no header) -> list of dicts."""
    out = []
    for line in rows:
        f = line.split(",")
        out.append({"pass": int(f[0]), "frame": int(f[1]), "track_id": int(f[3]), "x": float(f[4]),
                    "y": float(f[5]), "vx": float(f[6]), "vy": float(f[7]), "speed": float(f[8]), "state": f[9]})
    return out


def gt_world_velocity(gt_boxes, T_wv, stamps_ns, half=GT_HALF_WINDOW):
    """{(tracklet id, frame): (vx, vy)} from world-frame centers, centered difference over +-half frames."""
    t = np.asarray(stamps_ns, dtype=np.int64)
    centers = defaultdict(dict)
    for b in gt_boxes:
        centers[b.id][b.frame] = poses.transform_points(T_wv[b.frame], (b.x, b.y, b.z))[:2]
    vel = {}
    for tid, by_frame in centers.items():
        fs = sorted(by_frame)
        for i, f in enumerate(fs):
            lo, hi = fs[max(0, i - half)], fs[min(len(fs) - 1, i + half)]
            if hi == lo:
                continue
            dt = (t[hi] - t[lo]) * 1e-9
            vel[(tid, f)] = tuple((by_frame[hi] - by_frame[lo]) / dt)
    return vel


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _matches(rows, gt_boxes, T_wv, p):
    """Per-frame one-to-one matches of track rows to GT boxes in the evaluation region: yields (frame, row, box)."""
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)
    gt_by_frame = defaultdict(list)
    for b in gt_boxes:
        if bool(matching.in_eval_region(b.x, b.y, b.z, p)):
            gt_by_frame[b.frame].append(b)
    for f in sorted(set(gt_by_frame) & set(by_frame)):
        gts, trs = gt_by_frame[f], by_frame[f]
        if p.ego_compensation:
            gxy = [tuple(poses.transform_points(T_wv[f], (g.x, g.y, g.z))[:2]) for g in gts]
        else:
            gxy = [(g.x, g.y) for g in gts]
        for gi, ti, _ in matching.match(gxy, [(r["x"], r["y"]) for r in trs]):
            yield f, trs[ti], gts[gi]


def gt_net_speed(gt_boxes, T_wv, stamps_ns):
    """{tracklet id: world net speed (first-to-last displacement / time), m/s} (the E1.2 definition)."""
    t = np.asarray(stamps_ns, dtype=np.int64)
    frames = defaultdict(dict)
    for b in gt_boxes:
        frames[b.id][b.frame] = poses.transform_points(T_wv[b.frame], (b.x, b.y, b.z))[:2]
    net = {}
    for i, by_frame in frames.items():
        f0, f1 = min(by_frame), max(by_frame)
        net[i] = float(np.hypot(*(by_frame[f1] - by_frame[f0])) / ((t[f1] - t[f0]) * 1e-9)) if f1 > f0 else 0.0
    return net


def gt_tags(track_rows, gt_boxes, T_wv, stamps_ns, p, min_net_speed=0.5):
    """{(pass, track_id): "GT #id Class"} for tracks whose most frequent GT match is a moving tracklet
    (world net speed >= min_net_speed, the complement of the static definition). For the offline report only:
    the labels come from the ground truth; the processor classifies nothing."""
    if not gt_boxes:
        return {}
    cls = {b.id: b.cls for b in gt_boxes}
    net = gt_net_speed(gt_boxes, T_wv, stamps_ns)
    counts = defaultdict(lambda: defaultdict(int))
    for _, r, g in _matches(parse_track_rows(track_rows), gt_boxes, T_wv, p):
        counts[(r["pass"], r["track_id"])][g.id] += 1
    tags = {}
    for key, c in counts.items():
        gid = min(c, key=lambda i: (-c[i], i))  # most matched frames; ties to the lower tracklet id
        if net.get(gid, 0.0) >= min_net_speed:
            tags[key] = f"GT #{gid} {cls[gid]}"
    return tags


def score(track_rows, gt_boxes, T_wv, stamps_ns, p):
    """Speed summary (deterministic dict). track_rows: tracks.csv data lines of one pass."""
    if not gt_boxes:
        return {"ground_truth": "missing"}
    rows = parse_track_rows(track_rows)
    first_frame = {}
    for r in rows:
        first_frame.setdefault((r["pass"], r["track_id"]), r["frame"])
    gt_vel = gt_world_velocity(gt_boxes, T_wv, stamps_ns)
    cls = {b.id: b.cls for b in gt_boxes}

    samples = defaultdict(list)  # tracklet id -> [(track speed, gt speed, state)]
    for f, r, g in _matches(rows, gt_boxes, T_wv, p):
        if f - first_frame[(r["pass"], r["track_id"])] < p.warmup_frames:
            continue
        v = gt_vel.get((g.id, f))
        samples[g.id].append((r["speed"], None if v is None else float(np.hypot(*v)), r["state"]))

    parked = {}
    for tid in PARKED:
        s = samples.get(tid, [])
        sp = [x[0] for x in s]
        parked[str(tid)] = {"cls": cls.get(tid), "matched_frames": len(s),
                            "median_speed": _r(np.median(sp)) if sp else None,
                            "p90_speed": _r(np.percentile(sp, 90)) if sp else None,
                            "share_moving": _r(sum(x[2] == "moving" for x in s) / len(s)) if s else None}
    medians = [v["median_speed"] for v in parked.values()]
    known = [m for m in medians if m is not None]
    mom = float(np.median(known)) if len(known) == len(medians) else None
    worst = max(parked.items(), key=lambda kv: -1 if kv[1]["median_speed"] is None else kv[1]["median_speed"])
    parked_ok = (mom is not None and mom <= PARKED_BAR_MEDIAN
                 and all(m <= PARKED_BAR_EACH for m in known))

    moving = {}
    for tid in MOVING:
        err = [abs(x[0] - x[1]) for x in samples.get(tid, []) if x[1] is not None]
        moving[str(tid)] = {"cls": cls.get(tid), "matched_frames": len(err),
                            "median_abs_error": _r(np.median(err)) if err else None,
                            "p90_abs_error": _r(np.percentile(err, 90)) if err else None,
                            "median_gt_speed": _r(np.median([x[1] for x in samples[tid] if x[1] is not None])) if err else None,
                            "share_moving": _r(sum(x[2] == "moving" for x in samples[tid]) / len(samples[tid])) if samples.get(tid) else None,
                            "small_sample": cls.get(tid) in SMALL_SAMPLE_CLASSES}
    moving_ok = all(v["median_abs_error"] is not None and v["median_abs_error"] <= MOVING_BAR for v in moving.values())

    return {
        "ground_truth": "present",
        "tracking_frame": "world" if p.ego_compensation else "velodyne",
        "method": {"gt_velocity": f"centered difference of world-frame GT centers over +-{GT_HALF_WINDOW} frames",
                   "matching": f"per frame, Hungarian on 2D centers, gate {matching.EVAL_GATE_M} m, evaluation region",
                   "warmup_frames": p.warmup_frames, "aggregation": "by tracklet"},
        "parked": {"tracklets": parked, "median_of_medians": _r(mom),
                   "worst": {"id": int(worst[0]), "median_speed": worst[1]["median_speed"]},
                   "bar_median_of_medians": PARKED_BAR_MEDIAN, "bar_each": PARKED_BAR_EACH, "pass": bool(parked_ok)},
        "moving": {"objects": moving, "bar_median_abs_error": MOVING_BAR, "pass": bool(moving_ok)},
        "tracks": {"rows": len(rows), "confirmed": len(first_frame)},
    }
