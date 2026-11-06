"""Phase 4 evaluation metrics of one offline run against the KITTI tracklets. ROS-free (numpy, scipy via matching).

Definitions are fixed in docs/spec.md (Evaluation, Metrics) and the phase-4 FEATURE_SPEC (D4-D16):
- The unit is the GT box-frame. Matching is track_scoring._matches, unchanged: per frame, one-to-one Hungarian on
  2D centers, gate 2.0 m, ground truth of every class in the evaluation region against all confirmed tracks.
  That keeps every number that overlaps Phase 3 (tracking_speeds.json) identical.
- A GT object is static if its tracklet's world net speed is < 0.5 m/s (the E1.2 / PARKED definition).
- The first warmup_frames frames after a track's confirmation are excluded from velocity RMSE, static accuracy,
  moving-label recall, the false-moving rate and world stillness. They are not excluded from recall, label
  coverage, center error, ID switches or lifetimes.
- A track is in the region if its center, mapped back to velodyne, is in the evaluation region on x and y
  (tracks.csv has no z, so z never excludes a track; the sensor height is used for the mapping).
- Every ratio is a dict {value, num/n, den_name}; value is None (printed n/a) when its denominator is zero.
"""
from collections import defaultdict

import numpy as np

from core import poses
from processor import matching, track_scoring

STATIC_NET_SPEED = 0.5  # m/s, world net speed of a tracklet below which it is static (D7)
ABLATION_RATIO_BAR = 3.0  # D1: no-ego static-object velocity RMSE must be at least this multiple of the compensated one
MIN_STILLNESS_FRAMES = 10  # D15: post-warm-up frames a track needs to be scored for world stillness (1 s)


def _rate(num, den, den_name):
    return {"value": (num / den) if den else None, "num": int(num), "den": int(den), "den_name": den_name}


def _stat(values, den_name, q=None):
    """Median (and an optional percentile q) of a list; value None when empty."""
    if not len(values):
        return {"value": None, "n": 0, "den_name": den_name, "pq": None}
    return {"value": float(np.median(values)), "n": len(values), "den_name": den_name,
            "pq": None if q is None else float(np.percentile(values, q))}


def _mean(values):
    return {"value": float(np.mean(values)) if len(values) else None, "n": len(values), "den_name": "matched pairs"}


def _rmse(sum_sq, n, den_name):
    return {"value": float(np.sqrt(sum_sq / n)) if n else None, "n": int(n), "den_name": den_name}


def static_tracklets(gt_boxes, T_wv, stamps_ns):
    """Ids of the static GT tracklets (D7): world net speed < 0.5 m/s."""
    return {i for i, v in track_scoring.gt_net_speed(gt_boxes, T_wv, stamps_ns).items() if v < STATIC_NET_SPEED}


def _track_xy_velodyne(r, T_wv, ego):
    """A track row's center in the velodyne frame, xy only (the sensor height stands in for the missing z)."""
    if not ego:
        return r["x"], r["y"]
    T = T_wv[r["frame"]]
    d = np.array([r["x"] - T[0, 3], r["y"] - T[1, 3], 0.0])
    p = T[:3, :3].T @ d
    return float(p[0]), float(p[1])


def evaluate(track_rows, dets_by_frame, gt_boxes, T_wv, stamps_ns, params, n_frames):
    """All metrics of one run (one offline pass). track_rows: tracks.csv data lines. Returns a plain dict."""
    p = params
    ego = p.ego_compensation
    if not gt_boxes:
        raise ValueError("evaluation needs ground truth (no tracklet boxes)")
    rows = sorted(track_scoring.parse_track_rows(track_rows), key=lambda r: (r["frame"], r["track_id"]))
    if any(r["pass"] != 0 for r in rows):
        raise ValueError("evaluation expects one offline pass (pass 0)")

    z_mid = float(np.mean(p.roi_z))
    by_track, first_frame = defaultdict(list), {}
    for r in rows:
        by_track[r["track_id"]].append(r)
        first_frame.setdefault(r["track_id"], r["frame"])
        vx, vy = _track_xy_velodyne(r, T_wv, ego)
        r["in_region"] = bool(matching.in_eval_region(vx, vy, z_mid, p))
        r["warm"] = r["frame"] - first_frame[r["track_id"]] >= p.warmup_frames

    static = static_tracklets(gt_boxes, T_wv, stamps_ns)
    cls_of = {b.id: b.cls for b in gt_boxes}
    gt_region = [b for b in gt_boxes if bool(matching.in_eval_region(b.x, b.y, b.z, p))]
    gt_keys = {(b.frame, b.id) for b in gt_region}

    matches = list(track_scoring._matches(rows, gt_boxes, T_wv, p))  # private on purpose: unchanged Phase 3 matching
    matched_gt = {(f, g.id) for f, _, g in matches}
    matched_rows = {(f, r["track_id"]) for f, r, _ in matches}
    matched_track_ids = {r["track_id"] for _, r, _ in matches}
    assert matched_gt <= gt_keys

    # ------------------------------------------------------------ recall (D5)
    per_class, per_tracklet = {}, {}
    for b in gt_region:
        c = per_class.setdefault(b.cls, {"total": 0, "matched": 0, "ids": set()})
        c["total"] += 1
        c["ids"].add(b.id)
        t = per_tracklet.setdefault(b.id, {"cls": b.cls, "static": b.id in static, "total": 0, "matched": 0})
        t["total"] += 1
    for _, g in matched_gt:
        per_class[cls_of[g]]["matched"] += 1
        per_tracklet[g]["matched"] += 1
    missed = {c: v["total"] - v["matched"] for c, v in per_class.items()}
    recall_by_class = {
        c: {**_rate(v["matched"], v["total"], f"{c} GT box-frames"), "missed": missed[c], "tracklets": len(v["ids"]),
            "small_sample": len(v["ids"]) < matching.SMALL_SAMPLE_TRACKLETS}
        for c, v in sorted(per_class.items())}
    cv_m = sum(per_class.get(c, {"matched": 0})["matched"] for c in matching.CAR_VAN)
    cv_t = sum(per_class.get(c, {"total": 0})["total"] for c in matching.CAR_VAN)
    recall = {"overall": _rate(len(matched_gt), len(gt_keys), "GT box-frames in the region"),
              "missed": len(gt_keys - matched_gt),
              "car_van": _rate(cv_m, cv_t, "Car+Van GT box-frames"), "per_class": recall_by_class}

    # ------------------------------------------------------------ label coverage (D14), unlabelled tracks
    region_rows = [r for r in rows if r["in_region"]]
    matched_region_rows = [r for r in region_rows if (r["frame"], r["track_id"]) in matched_rows]
    coverage = _rate(len(matched_region_rows), len(region_rows), "confirmed track-frames in the region")
    region_tracks = {r["track_id"] for r in region_rows}
    labelled = region_tracks & matched_track_ids
    unlabelled = region_tracks - matched_track_ids

    # ------------------------------------------------------------ center error, ID switches
    def gt_xy(f, g):
        return (poses.transform_points(T_wv[f], (g.x, g.y, g.z))[:2] if ego else np.array([g.x, g.y]))

    errs, errs_by_class = [], defaultdict(list)
    for f, r, g in matches:
        e = float(np.hypot(r["x"] - gt_xy(f, g)[0], r["y"] - gt_xy(f, g)[1]))
        errs.append(e)
        errs_by_class[g.cls].append(e)
    center_error = {"overall": _mean(errs), "per_class": {c: _mean(errs_by_class.get(c, [])) for c in sorted(per_class)}}

    seq = defaultdict(list)  # tracklet id -> [(frame, track_id)]
    for f, r, g in matches:
        seq[g.id].append((f, r["track_id"]))
    id_sw = {}
    for gid, items in seq.items():
        ids = [tid for _, tid in sorted(items)]
        id_sw[gid] = sum(a != b for a, b in zip(ids, ids[1:]))

    # ------------------------------------------------------------ velocity, labels (warm-up scope, D8)
    gt_vel = track_scoring.gt_world_velocity(gt_boxes, T_wv, stamps_ns)
    ss = {"static": 0.0, "moving": 0.0}
    nn = {"static": 0, "moving": 0}
    cls_ss, cls_n = defaultdict(float), defaultdict(int)
    label = {"static_n": 0, "static_ok": 0, "moving_n": 0, "moving_ok": 0}
    samples = defaultdict(list)  # tracklet id -> [(speed, gt speed or None, state)]
    for f, r, g in matches:
        if not r["warm"]:
            continue
        v = gt_vel.get((g.id, f))
        samples[g.id].append((r["speed"], None if v is None else float(np.hypot(*v)), r["state"]))
        kind = "static" if g.id in static else "moving"
        label[kind + "_n"] += 1
        label[kind + "_ok"] += r["state"] == kind
        if v is None:
            continue
        gv = np.array([v[0], v[1], 0.0])
        if not ego:
            gv = T_wv[f][:3, :3].T @ gv  # D10: world GT velocity in velodyne axes, no ego subtraction
        d2 = float((r["vx"] - gv[0]) ** 2 + (r["vy"] - gv[1]) ** 2)
        ss[kind] += d2
        nn[kind] += 1
        cls_ss[g.cls] += d2
        cls_n[g.cls] += 1
    velocity = {"rmse_static": _rmse(ss["static"], nn["static"], "matched static post-warm-up frames"),
                "rmse_moving": _rmse(ss["moving"], nn["moving"], "matched moving post-warm-up frames"),
                "per_class": {c: _rmse(cls_ss[c], cls_n[c], f"matched {c} post-warm-up frames") for c in sorted(cls_n)}}
    labels = {"static_accuracy": _rate(label["static_ok"], label["static_n"], "matched static-GT post-warm-up frames"),
              "flicker": _rate(label["static_n"] - label["static_ok"], label["static_n"],
                               "matched static-GT post-warm-up frames"),
              "moving_label_recall": _rate(label["moving_ok"], label["moving_n"], "matched moving-GT post-warm-up frames")}

    unmatched_warm = [r for r in region_rows if r["warm"] and (r["frame"], r["track_id"]) not in matched_rows]
    labels["false_moving"] = _rate(sum(r["state"] == "moving" for r in unmatched_warm), len(unmatched_warm),
                                   "unmatched in-region post-warm-up track-frames")

    # ------------------------------------------------------------ per-tracklet table
    tracklets = {}
    for tid, t in sorted(per_tracklet.items()):
        s = samples.get(tid, [])
        err = [abs(x[0] - x[1]) for x in s if x[1] is not None]
        tracklets[tid] = {**t, "recall": _rate(t["matched"], t["total"], f"#{tid} GT box-frames"),
                          "scored_frames": len(s),
                          "median_speed": _stat([x[0] for x in s], "scored frames")["value"],
                          "median_abs_error": _stat(err, "scored frames with GT velocity")["value"],
                          "share_moving": _rate(sum(x[2] == "moving" for x in s), len(s), "scored frames"),
                          "id_switches": id_sw.get(tid, 0)}

    # ------------------------------------------------------------ world stillness (D15), lifetimes (part 1)
    best = {}  # track id -> the tracklet it matches most often (ties: lower id), as track_scoring.gt_tags
    counts = defaultdict(lambda: defaultdict(int))
    for _, r, g in matches:
        counts[r["track_id"]][g.id] += 1
    for tid, c in counts.items():
        best[tid] = min(c, key=lambda i: (-c[i], i))
    parked_ref = {tid for tid, gid in best.items() if gid in static}

    def stillness(track_ids):
        speeds, spreads, excluded = [], [], 0
        for tid in sorted(track_ids):
            warm = [r for r in by_track[tid] if r["warm"]]
            if len(warm) < MIN_STILLNESS_FRAMES:
                excluded += 1
                continue
            xy = np.array([[r["x"], r["y"]] for r in warm])
            speeds.append(float(np.median([r["speed"] for r in warm])))
            spreads.append(float(np.max(np.hypot(*(xy - xy.mean(axis=0)).T))))
        return {"tracks": len(track_ids), "scored": len(speeds), "excluded": excluded,
                "median_speed": _stat(speeds, "scored tracks")["value"],
                "median_spread": _stat(spreads, "scored tracks")["value"]}

    def lifetimes(track_ids):
        return _stat([len(by_track[t]) for t in track_ids], "tracks", q=90)

    moving_tracks = sum(1 for tid, rs in by_track.items() if sum(r["state"] == "moving" for r in rs) * 2 >= len(rs))
    per_frame = np.bincount([r["frame"] for r in rows], minlength=max(n_frames, 1)) if rows else np.zeros(max(n_frames, 1), int)
    dets_per_frame = [len(dets_by_frame.get(f, ())) for f in range(n_frames)]

    return {
        "frames": n_frames,
        "tracking_frame": "world" if ego else "velodyne",
        "gt": {"box_frames": len(gt_keys), "tracklets": len(per_tracklet), "static_tracklets": sorted(static)},
        "produced": {
            "detections_per_frame": {"mean": float(np.mean(dets_per_frame)) if dets_per_frame else None,
                                     "max": max(dets_per_frame) if dets_per_frame else None},
            "tracks": len(by_track), "tracks_in_region": len(region_tracks),
            "labelled_tracks": len(labelled), "unlabelled_tracks": len(unlabelled),
            "confirmed_per_frame": {"mean": float(per_frame.mean()), "max": int(per_frame.max())},
            "track_frames": len(rows), "moving_track_frame_share": _rate(sum(r["state"] == "moving" for r in rows), len(rows), "track-frames"),
            "moving_tracks": moving_tracks,
            "lifetime_frames": {"all": lifetimes(sorted(by_track)), "labelled": lifetimes(sorted(labelled)),
                                "unlabelled": lifetimes(sorted(unlabelled))},
        },
        "recall": recall, "coverage": coverage,
        "unlabelled_track_frames": len(region_rows) - len(matched_region_rows),
        "center_error": center_error, "velocity": velocity, "labels": labels,
        "id_switches": {"total": int(sum(id_sw.values())), "per_tracklet": {int(k): v for k, v in sorted(id_sw.items())}},
        "tracklets": tracklets,
        "stillness": {"unlabelled": stillness(unlabelled), "parked": stillness(parked_ref)},
    }


def ablation(on, off):
    """D1: the no-ego static-object velocity RMSE must be >= ABLATION_RATIO_BAR times the compensated one."""
    a, b = on["velocity"]["rmse_static"]["value"], off["velocity"]["rmse_static"]["value"]
    ratio = (b / a) if (a and b is not None) else None
    return {"rmse_on": a, "rmse_off": b, "ratio": ratio, "bar": ABLATION_RATIO_BAR,
            "pass": bool(ratio is not None and ratio >= ABLATION_RATIO_BAR)}


# ---------------------------------------------------------------- rendering (E4.1, E4.2b)
# Fixed formats (:.3f, integers), sorted classes and tracklets, no timestamp, timing, host or platform text:
# the same inputs always render the same bytes.

PARTS = ("0. Summary", "1. What the pipeline produced", "2. Accuracy against KITTI labels", "3. Physical consistency",
         "4. Ablation: ego compensation off", "5. Method and reproducibility")
READ_PREFIX = "*How to read this:* "
NOTES = (
    "The headline numbers, each pointing to its part. One result is gated: removing ego compensation must at least "
    "triple the velocity error of parked cars. Everything else is reported, with no pass/fail bar by design, because "
    "bars are not set after the results are known.",
    "What the tracker output over the drive, before anything judges it: detections and tracks per frame, how long "
    "tracks live, how many have a KITTI label and how many read as moving. Read this first to know the scale of "
    "everything below; none of it is a quality claim.",
    "KITTI labels only cars, vans, pedestrians and cyclists, and only in front of the car. These numbers judge the "
    "tracker on those objects alone, so they say nothing about poles, signs or walls. The share of tracks explained "
    "by a label is therefore a lower bound on how many tracks are real objects, not an error rate.",
    "Most tracks have no KITTI label, which annotates only cars, vans, pedestrians and cyclists; many are probably "
    "poles, signs, trunks, bushes or walls, but nothing here classifies them. Without labels they can still be judged "
    "by physics: once ego motion is removed, a fixed object must stay still, like a labelled parked car. This shows "
    "consistency, not correctness: a still track could equally be a wall segment or leftover ground.",
    "The same drive tracked in the sensor frame, without the ego pose. Every fixed object should then appear to move "
    "at about the ego speed (3.1 to 6.5 m/s), whether KITTI labels it or not.",
    "How every number was computed, and the one command that regenerates this file byte for byte on the same CPU "
    "architecture.",
)
NO_BAR = "no bar by design"
NOTE4_PASS = "This is the project's main result: it shows that ego compensation, not luck, makes parked objects read still."
NOTE4_FAIL = "The gate failed in this run, so it does not show that ego compensation makes parked objects read still."
COMMAND = "docker compose run --rm --no-deps processor python3 /app/scripts/evaluate.py"


def render_meta(params):
    """The run description that goes into the file (no time, host or platform)."""
    return {"drive": params.drive, "seed": params.seed, "warmup_frames": params.warmup_frames, "command": COMMAND}


def _na(d):
    return f"n/a ({d['den_name']} = 0)"


def _v(d, nd=3):
    return _na(d) if d["value"] is None else f"{d['value']:.{nd}f}"


def _vn(d, nd=3):
    return _na(d) if d["value"] is None else f"{d['value']:.{nd}f} (n = {d['n']})"


def _rate_s(d):
    return _na(d) if d["value"] is None else f"{d['value']:.3f} ({d['num']} / {d['den']})"


def _x(v, name, nd=3):
    return f"n/a ({name} = 0)" if v is None else f"{v:.{nd}f}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _speed_summary(on):
    """The Phase 3 bars quoted from the compensated run: median of the parked medians, and the van and cyclist errors."""
    parked = [on["tracklets"].get(i, {}).get("median_speed") for i in track_scoring.PARKED]
    if any(m is None for m in parked):  # as track_scoring.score: no summary unless all ten tracklets were scored
        parked = []
    mom = float(np.median(parked)) if parked else None
    err = {i: on["tracklets"].get(i, {}).get("median_abs_error") for i in track_scoring.MOVING}
    return mom, (max(parked) if parked else None), err


def render(on, off, abl, meta):
    """The six-part evaluation.md text. on / off: evaluate() results of the compensated and the no-ego run."""
    L = [f"# Evaluation: {meta['drive']}", "",
         f"Offline run over all {on['frames']} frames, once with ego compensation and once without, scored against the KITTI "
         f"tracklets. Regenerate with `{meta['command']}`. Seed {meta['seed']}, association gate {matching.EVAL_GATE_M} m, "
         f"warm-up {meta['warmup_frames']} frames after a track's confirmation.", ""]

    def head(i):
        note = NOTES[i] + (" " + (NOTE4_PASS if abl["pass"] else NOTE4_FAIL) if i == 4 else "")
        L.extend([f"## {PARTS[i]}", "", READ_PREFIX + note, ""])

    rec, vel, lab = on["recall"], on["velocity"], on["labels"]
    mom, worst, merr = _speed_summary(on)
    st_u, st_p = on["stillness"]["unlabelled"], on["stillness"]["parked"]

    # ---- 0
    head(0)
    L += [f"**Gate (D1): {'PASS' if abl['pass'] else 'FAIL'}.** Static-object velocity RMSE is {_x(abl['rmse_on'], 'compensated RMSE')} m/s "
          f"with ego compensation and {_x(abl['rmse_off'], 'uncompensated RMSE')} m/s without: "
          f"a ratio of {_x(abl['ratio'], 'ratio', 2)}, against a bar of at least {abl['bar']:.1f} (part 4).", ""]
    L += _table(["Result", "Value", "Judged by", "Part"], [
        ["Static-object velocity RMSE, ego compensation off vs on", f"{_x(abl['ratio'], 'ratio', 2)}x", f"gated, bar {abl['bar']:.1f}x", 4],
        ["Parked tracklets: median of per-tracklet median speed (m/s)", _x(mom, "parked tracklets"),
         f"gated in Phase 3 (E3.2), bar {track_scoring.PARKED_BAR_MEDIAN}; worst tracklet {_x(worst, 'parked tracklets')}, bar {track_scoring.PARKED_BAR_EACH}", 2],
        ["Lead van and cyclist: median speed error (m/s)", " / ".join(_x(merr[i], "scored frames") for i in track_scoring.MOVING),
         f"gated in Phase 3 (E3.3), bar {track_scoring.MOVING_BAR}", 2],
        ["Car+Van recall (box-frames)", _rate_s(rec["car_van"]), NO_BAR, 2],
        ["Label coverage: share of in-region track-frames explained by a label", _rate_s(on["coverage"]), f"{NO_BAR}; a lower bound", 2],
        ["Static accuracy / parked flicker", f"{_v(lab['static_accuracy'])} / {_v(lab['flicker'])}", NO_BAR, 2],
        ["Moving-label recall", _rate_s(lab["moving_label_recall"]), NO_BAR, 2],
        ["ID switches", on["id_switches"]["total"], NO_BAR, 2],
        ["Unlabelled tracks: median speed (m/s) / position spread (m)", f"{_x(st_u['median_speed'], 'scored tracks')} / {_x(st_u['median_spread'], 'scored tracks')}",
         f"{NO_BAR}; parked-car reference {_x(st_p['median_speed'], 'scored tracks')} / {_x(st_p['median_spread'], 'scored tracks')}", 3],
        ["False-moving rate", _rate_s(lab["false_moving"]), NO_BAR, 3],
    ])
    L.append("")

    # ---- 1
    pr = on["produced"]
    head(1)
    lt = pr["lifetime_frames"]
    L += _table(["Quantity", "Value"], [
        ["Frames", on["frames"]],
        ["Detections per frame, mean / max", f"{pr['detections_per_frame']['mean']:.1f} / {pr['detections_per_frame']['max']}"],
        ["Confirmed tracks over the drive", pr["tracks"]],
        ["Confirmed tracks per frame, mean / max", f"{pr['confirmed_per_frame']['mean']:.1f} / {pr['confirmed_per_frame']['max']}"],
        ["Tracks with a frame in the evaluation region", pr["tracks_in_region"]],
        ["... of which matched to a KITTI tracklet in some frame (labelled)", pr["labelled_tracks"]],
        ["... of which never matched (unlabelled)", pr["unlabelled_tracks"]],
        ["Track-frames labelled `moving`", _rate_s(pr["moving_track_frame_share"])],
        ["Tracks labelled `moving` in at least half of their frames", pr["moving_tracks"]],
        ["Track lifetime in frames, median / p90, all tracks", f"{_x(lt['all']['value'], 'tracks', 1)} / {_x(lt['all']['pq'], 'tracks', 1)}"],
        ["Track lifetime, labelled tracks", f"{_x(lt['labelled']['value'], 'labelled tracks', 1)} / {_x(lt['labelled']['pq'], 'labelled tracks', 1)}"],
        ["Track lifetime, unlabelled tracks", f"{_x(lt['unlabelled']['value'], 'unlabelled tracks', 1)} / {_x(lt['unlabelled']['pq'], 'unlabelled tracks', 1)}"],
    ])
    L.append("")

    # ---- 2
    head(2)
    rows = []
    for cls, c in rec["per_class"].items():
        rows.append([cls, c["tracklets"], c["den"], c["num"], c["missed"], _v(c), _vn(on["center_error"]["per_class"][cls]),
                     _vn(vel["per_class"][cls]) if cls in vel["per_class"] else "n/a (scored frames = 0)",
                     "small sample" if c["small_sample"] else ""])
    ov = rec["overall"]
    rows.append(["**Overall**", on["gt"]["tracklets"], ov["den"], ov["num"], rec["missed"], _v(ov), _vn(on["center_error"]["overall"]),
                 "static / moving below", ""])
    L += ["**By class** (GT box-frames in the evaluation region; recall, center error and velocity RMSE: no bar by design)", ""]
    L += _table(["Class", "Tracklets", "GT box-frames", "Matched", "Missed", "Recall", "Center error (m)", "Velocity RMSE (m/s)", "Note"], rows)
    cv = rec["car_van"]
    L += ["", f"Car+Van recall: {_rate_s(cv)} (the Phase 2 detection recall used the same 520 box-frames).", ""]
    L += _table(["Metric", "Value", "Judged by"], [
        ["Label coverage (precision against labels): matched / in-region track-frames", _rate_s(on["coverage"]),
         f"{NO_BAR}; a lower bound, {on['unlabelled_track_frames']} track-frames have no label"],
        ["Velocity RMSE, static GT (m/s)", _vn(vel["rmse_static"]), f"{NO_BAR}; the ablation (part 4) gates it"],
        ["Velocity RMSE, moving GT (m/s)", _vn(vel["rmse_moving"]), NO_BAR],
        ["Static accuracy: static GT frames labelled `static`", _rate_s(lab["static_accuracy"]), NO_BAR],
        ["Parked flicker: static GT frames labelled `moving` (= 1 - static accuracy)", _rate_s(lab["flicker"]), NO_BAR],
        ["Moving-label recall: moving GT frames labelled `moving`", _rate_s(lab["moving_label_recall"]),
         f"{NO_BAR}; walkers read `static` by design (D10)"],
        ["ID switches, all tracklets", on["id_switches"]["total"], NO_BAR],
    ])
    trs = []
    for tid, t in on["tracklets"].items():
        trs.append([tid, t["cls"], "static" if t["static"] else "moving", t["total"], t["matched"], _v(t["recall"]), t["id_switches"],
                    t["scored_frames"], _x(t["median_speed"], "scored frames"), _x(t["median_abs_error"], "scored frames"),
                    _v(t["share_moving"])])
    L += ["", "**By tracklet** (frames after the warm-up: speed in m/s, |speed error| against the +-5-frame GT velocity, share labelled `moving`)", ""]
    L += _table(["Tracklet", "Class", "GT is", "GT box-frames", "Matched", "Recall", "ID switches", "Scored frames", "Median speed",
                 "Median abs. error", "Share moving"], trs)
    L.append("")

    # ---- 3
    head(3)
    L += _table(["Tracks", "Count", "Scored", "Excluded (< 10 frames)", "Median speed (m/s)", "Median position spread (m)"], [
        ["Unlabelled (never matched)", st_u["tracks"], st_u["scored"], st_u["excluded"], _x(st_u["median_speed"], "scored tracks"),
         _x(st_u["median_spread"], "scored tracks")],
        ["Reference: matched to a parked tracklet", st_p["tracks"], st_p["scored"], st_p["excluded"], _x(st_p["median_speed"], "scored tracks"),
         _x(st_p["median_spread"], "scored tracks")],
    ])
    L += ["", f"Only tracks with at least {MIN_STILLNESS_FRAMES} frames after the warm-up are scored; shorter ones are counted as excluded, not hidden. "
          f"Speed is the per-track median; spread is the largest distance from the track's mean world position. "
          f"The box center of a large object drifts more than that of a small one, so the two rows are not directly comparable. No bar by design.", ""]
    L += _table(["Metric", "Value", "Judged by"], [
        ["False-moving rate: unmatched in-region track-frames after the warm-up labelled `moving`", _rate_s(lab["false_moving"]),
         f"{NO_BAR}; the baseline for the Phase 6 fixes"]])
    L += ["", "This counts every in-region track-frame that matches no tracklet, including the unmatched frames of tracks that do match "
          "a tracklet in other frames (for example when a moving vehicle is briefly missed). It does not assume that unmatched objects are fixed.", ""]

    # ---- 4
    head(4)
    po = off["stillness"]
    L += _table(["Metric", "Ego compensation on (world)", "Ego compensation off (velodyne)", "Judged by"], [
        ["Static-object velocity RMSE (m/s)", _vn(on["velocity"]["rmse_static"]), _vn(off["velocity"]["rmse_static"]), "**gated**"],
        ["Ratio off / on", f"{_x(abl['ratio'], 'ratio', 2)}x", "", f"**gated: bar {abl['bar']:.1f}x, {'PASS' if abl['pass'] else 'FAIL'}**"],
        ["Moving-object velocity RMSE (m/s)", _vn(on["velocity"]["rmse_moving"]), _vn(off["velocity"]["rmse_moving"]), NO_BAR],
        ["Parked flicker", _rate_s(on["labels"]["flicker"]), _rate_s(off["labels"]["flicker"]), NO_BAR],
        ["Moving-label recall", _rate_s(on["labels"]["moving_label_recall"]), _rate_s(off["labels"]["moving_label_recall"]), NO_BAR],
        ["False-moving rate", _rate_s(on["labels"]["false_moving"]), _rate_s(off["labels"]["false_moving"]), NO_BAR],
        ["Car+Van recall", _rate_s(on["recall"]["car_van"]), _rate_s(off["recall"]["car_van"]), NO_BAR],
        ["ID switches", on["id_switches"]["total"], off["id_switches"]["total"], NO_BAR],
        ["Unlabelled tracks: median speed (m/s)", _x(st_u["median_speed"], "scored tracks"), _x(po["unlabelled"]["median_speed"], "scored tracks"), NO_BAR],
        ["Unlabelled tracks: median position spread (m)", _x(st_u["median_spread"], "scored tracks"), _x(po["unlabelled"]["median_spread"], "scored tracks"), NO_BAR],
        ["Parked-car reference: median speed (m/s)", _x(st_p["median_speed"], "scored tracks"), _x(po["parked"]["median_speed"], "scored tracks"), NO_BAR],
    ])
    L += ["", "The off run tracks in the velodyne frame, so its positions move with the car and its velocities are measured in the sensor axes; "
          "the ground-truth world velocity is rotated into those axes (rotation only, no ego-velocity subtraction). "
          "The two runs match different sets of frames (the counts n differ), so the ratio compares RMSE over different frames.", ""]

    # ---- 5
    head(5)
    L += [
        f"- **Matching:** per frame, one-to-one Hungarian assignment on 2D center distance, gate {matching.EVAL_GATE_M} m; ground truth of every class in the evaluation region against all confirmed tracks (the Phase 3 matching, unchanged).",
        "- **Evaluation region:** the processing ROI minus the ego box, front half only (x >= 0 in the velodyne frame). Ground truth is tested on its velodyne center; a track is mapped back to velodyne and tested on x and y (tracks.csv has no z).",
        "- **Unit:** one ground-truth box-frame. Recall is matched / GT box-frames in the region; matched + missed equals that count.",
        f"- **Static:** a tracklet whose world net speed is below {STATIC_NET_SPEED} m/s (the 10 parked tracklets); every other tracklet is moving.",
        f"- **Warm-up:** the first {meta['warmup_frames']} frames after a track's confirmation are excluded from velocity RMSE, static accuracy, moving-label recall, the false-moving rate and world stillness. They are not excluded from recall, label coverage, center error, ID switches or lifetimes.",
        "- **Velocity:** the track's filtered velocity against the world-frame GT velocity, a centered difference over +-5 frames; RMSE is the root of the mean squared vector difference.",
        "- **ID switch:** per tracklet, a matched frame whose track ID differs from the previous matched track ID; frames without a match do not reset it.",
        f"- **Unlabelled track:** a track with at least one frame in the region that is never matched to a tracklet. World stillness scores tracks with at least {MIN_STILLNESS_FRAMES} post-warm-up frames.",
        "- **Zero denominators** print `n/a` with the name of the empty denominator, never 0.",
        f"- **Gate:** only the ablation ratio (at least {ABLATION_RATIO_BAR:.1f}) is a pass/fail bar, fixed before the first Phase 4 run.",
        f"- **Reproduce:** `{meta['command']}`. Two runs give byte-identical files on the same CPU architecture (verified on arm64 and on x86_64, the latter on a GitHub Actions runner).",
    ]
    return "\n".join(L) + "\n"
