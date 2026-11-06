"""Row formats, file writers and comparers shared by the live node, the offline runner and the live checker.

ROS-free. One writer for the detection and stage files is what makes the live/offline parity check
(E2.6) meaningful: both sides format rows with the same code. Hashed files carry no timing.
"""
import json
import math
import os
import platform
from pathlib import Path

DETECTION_HEADER = "frame,stamp_ns,x,y,z,dx,dy,dz,n_points"
FRAMES_HEADER = "frame,stamp_ns,raw,roi,voxel,non_ground,clustered,clusters_found,clusters_kept,ground_status"
DELETEALL = 3  # visualization_msgs/Marker.DELETEALL
TRACK_HEADER = "pass,frame,stamp_ns,track_id,x,y,vx,vy,speed,state,l,w,h"  # frozen at the end of Phase 3


# ---------------------------------------------------------------- rows

def detection_rows(frame, stamp_ns, detections):
    return [f"{frame},{stamp_ns},{d.x:.4f},{d.y:.4f},{d.z:.4f},{d.dx:.4f},{d.dy:.4f},{d.dz:.4f},{d.n_points}"
            for d in detections]


def frames_row(frame, stamp_ns, result):
    s = result.stages
    return (f"{frame},{stamp_ns},{s['raw']},{s['roi']},{s['voxel']},{s['non_ground']},{s['clustered']},"
            f"{result.clusters_found},{result.clusters_kept},{result.ground_status}")


def track_rows(pass_no, frame, stamp_ns, tracks):
    """tracks.csv rows (contract in docs/spec.md): confirmed tracks of one frame, sorted by track_id.

    x, y, vx, vy in the tracking frame (world, or velodyne with ego compensation off); l, w, h are the last
    matched detection's extents along the velodyne axes. The same function formats live and offline rows.
    """
    return [f"{pass_no},{frame},{stamp_ns},{t.track_id},{t.x:.4f},{t.y:.4f},{t.vx:.4f},{t.vy:.4f},{t.speed:.4f},"
            f"{t.state},{t.l:.4f},{t.w:.4f},{t.h:.4f}" for t in sorted(tracks, key=lambda t: t.track_id)]


def csv_text(header, rows, pass_no=None):
    """Header plus rows, \\n line endings. With pass_no the rows get a leading 'pass' column (live files)."""
    if pass_no is None:
        lines = [header, *rows]
    else:
        lines = ["pass," + header, *[f"{pass_no},{r}" for r in rows]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- statistics

def p95_nearest_rank(values):
    v = sorted(values)
    if not v:
        return None
    return v[max(math.ceil(0.95 * len(v)) - 1, 0)]


def time_stats(ms, scope):
    if not ms:
        return {"scope": scope, "mean": None, "p50": None, "p95": None, "max": None, "n": 0}
    v = sorted(ms)
    return {"scope": scope, "n": len(v), "mean": round(sum(v) / len(v), 3), "p50": round(v[(len(v) - 1) // 2], 3),
            "p95": round(p95_nearest_rank(v), 3), "max": round(v[-1], 3)}


def platform_info():
    import numpy, scipy, sklearn
    return {"machine": platform.machine(), "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


# ---------------------------------------------------------------- files

def _round_floats(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, nd) for v in obj]
    return obj


def canonical_json(obj):
    """Sorted keys, rounded floats: byte-identical for equal content."""
    return json.dumps(_round_floats(obj), indent=1, sort_keys=True) + "\n"


def atomic_write(path, text):
    """Write via a temp file and os.replace, so a reader (or a SIGKILL) never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- comparers (live checker)

def read_live_rows(detections_csv, frames_csv):
    """{(pass, frame): (frames_row_text, [detection_row_text, ...])} from the live files (pass column first)."""
    out = {}
    for line in Path(frames_csv).read_text().splitlines()[1:]:
        pass_no, rest = line.split(",", 1)
        out[(int(pass_no), int(rest.split(",", 1)[0]))] = (rest, [])
    for line in Path(detections_csv).read_text().splitlines()[1:]:
        pass_no, rest = line.split(",", 1)
        out[(int(pass_no), int(rest.split(",", 1)[0]))][1].append(rest)
    return out


def read_offline_rows(detections_csv, frames_csv):
    """{frame: (frames_row_text, [detection_row_text, ...])} from the offline files (no pass column)."""
    out = {}
    for line in Path(frames_csv).read_text().splitlines()[1:]:
        out[int(line.split(",", 1)[0])] = (line, [])
    for line in Path(detections_csv).read_text().splitlines()[1:]:
        out[int(line.split(",", 1)[0])][1].append(line)
    return out


def compare_parity(live, offline, min_frames=0):
    """E2.6: every live (pass, frame) must equal the offline rows of that frame, byte for byte.

    Returns {"compared": pairs, "distinct_frames": n, "mismatched": [...], "ok": bool}; ok needs 0 mismatches
    and at least min_frames distinct dataset frames compared.
    """
    mismatched = []
    for (pass_no, frame), (frames_text, det_rows) in sorted(live.items()):
        ref = offline.get(frame)
        if ref is None or ref[0] != frames_text or ref[1] != det_rows:
            mismatched.append({"pass": pass_no, "frame": frame})
    n = len(live)
    distinct = len({frame for _, frame in live})
    return {"compared": n, "distinct_frames": distinct, "mismatched": mismatched,
            "ok": not mismatched and n > 0 and distinct >= min_frames}


def compare_markers(arrays, live):
    """E2.5: every recorded MarkerArray starts with DELETEALL, and its LINE_LIST count equals the live
    detection count for its stamp (one text label per box, nothing else).

    arrays: [{"stamp_ns", "first_action", "n_line_list", "n_text", "n_total"}] in arrival order.
    live: output of read_live_rows. Counts are matched by stamp: detections are deterministic, so a
    stamp has the same count in every pass (checked, a stamp with two counts is reported).
    """
    by_stamp = {}
    for (_, _), (frames_text, rows) in live.items():
        stamp = int(frames_text.split(",")[1])
        by_stamp.setdefault(stamp, set()).add(len(rows))
    problems = []
    for i, a in enumerate(arrays):
        counts = by_stamp.get(a["stamp_ns"])
        if a["first_action"] != DELETEALL:
            problems.append({"array": i, "stamp_ns": a["stamp_ns"], "problem": "does not begin with DELETEALL"})
        elif counts is None:
            problems.append({"array": i, "stamp_ns": a["stamp_ns"], "problem": "stamp was not processed live"})
        elif len(counts) > 1:
            problems.append({"array": i, "stamp_ns": a["stamp_ns"], "problem": f"live counts differ across passes: {sorted(counts)}"})
        elif a["n_line_list"] != next(iter(counts)):
            problems.append({"array": i, "stamp_ns": a["stamp_ns"],
                             "problem": f"{a['n_line_list']} boxes but {next(iter(counts))} detections"})
        elif a["n_text"] != a["n_line_list"] or a["n_total"] != 1 + a["n_line_list"] + a["n_text"]:
            problems.append({"array": i, "stamp_ns": a["stamp_ns"], "problem": "markers are not DELETEALL + box + label per detection"})
    return {"arrays": len(arrays), "problems": problems, "ok": not problems and len(arrays) > 0}


# ---------------------------------------------------------------- Phase 3 comparers (live checker)

def read_live_tracks(tracks_csv):
    """{(pass, frame): [tracks.csv row text, ...]} (rows keep their pass column)."""
    out = {}
    for line in Path(tracks_csv).read_text().splitlines()[1:]:
        p, f = line.split(",", 2)[:2]
        out.setdefault((int(p), int(f)), []).append(line)
    return out


def tracks_by_pass(processed, live_tracks):
    """{pass: [rows in processing order]} for the processed (pass, frame) list."""
    out = {}
    for p, f in processed:
        out.setdefault(p, []).extend(live_tracks.get((p, f), []))
    return out


def frames_by_pass(processed):
    out = {}
    for p, f in processed:
        out.setdefault(p, []).append(f)
    return out


def compare_track_parity(live_by_pass, replay_by_pass):
    """E3.4: per pass, the live tracks.csv rows equal the offline replay of the same frames, byte for byte."""
    mismatched = []
    for p in sorted(set(live_by_pass) | set(replay_by_pass)):
        a, b = live_by_pass.get(p, []), replay_by_pass.get(p, [])
        if a != b:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            mismatched.append({"pass": p, "live_rows": len(a), "replay_rows": len(b), "first_difference_row": first,
                               "live": a[first] if first < len(a) else None, "replay": b[first] if first < len(b) else None})
    rows = sum(len(v) for v in live_by_pass.values())
    return {"passes": len(live_by_pass), "rows": rows, "mismatched": mismatched, "ok": not mismatched and rows > 0}


def check_loop(processed, live_tracks):
    """E3.5: pass changes exactly at a wrap (frame index goes back), by +1; the first processed frame of every
    pass has no confirmed track (a reset tracker needs confirm_hits frames), so no track survives a wrap.

    Assumes confirm_hits >= 2 (it is 3): with 1, a fresh track would be confirmed on the first frame. A stale ID
    later in a pass can't happen without a missed reset, and Tracker.step raises on a backward stamp that was
    not preceded by reset()."""
    problems = []
    for (p0, f0), (p1, f1) in zip(processed, processed[1:]):
        if p1 not in (p0, p0 + 1):
            problems.append({"at": [p1, f1], "problem": f"pass jumps from {p0} to {p1}"})
        elif p1 == p0 + 1 and f1 >= f0:
            problems.append({"at": [p1, f1], "problem": "pass incremented without a wrap"})
        elif p1 == p0 and f1 <= f0:
            problems.append({"at": [p1, f1], "problem": "frame went back without a pass increment"})
    firsts = {}
    for p, f in processed:
        firsts.setdefault(p, f)
    for p, f in sorted(firsts.items()):
        if live_tracks.get((p, f)):
            problems.append({"at": [p, f], "problem": f"{len(live_tracks[(p, f)])} tracks on the first frame of the pass"})
    return {"passes": len(firsts), "wraps": len(firsts) - 1, "problems": problems, "ok": not problems and bool(processed)}


def compare_track_markers(arrays, live_frames, live_tracks):
    """Track markers (Phase 3): every array starts with DELETEALL carrying the pass; boxes and labels: one per
    confirmed track of that (pass, stamp); arrows: one per moving track.

    arrays: [{"topic", "pass", "stamp_ns", "first_action", "n_line_list", "n_text", "n_arrow", "n_total"}].
    live_frames: read_live_rows output (gives (pass, frame) -> stamp); live_tracks: read_live_tracks output.
    """
    counts = {}
    for (p, f), (frames_text, _) in live_frames.items():
        rows = live_tracks.get((p, f), [])
        counts[(p, int(frames_text.split(",")[1]))] = (len(rows), sum(r.split(",")[9] == "moving" for r in rows))
    want = {"/perception/bounding_boxes": ("n_line_list", 0), "/perception/track_labels": ("n_text", 0),
            "/perception/obstacle_vectors": ("n_arrow", 1)}
    problems, per_topic = [], {}
    for i, a in enumerate(arrays):
        per_topic[a["topic"]] = per_topic.get(a["topic"], 0) + 1
        c = counts.get((a["pass"], a["stamp_ns"]))
        field, which = want[a["topic"]]
        if a["first_action"] != DELETEALL:
            problems.append({"array": i, "topic": a["topic"], "problem": "does not begin with DELETEALL"})
        elif c is None:
            problems.append({"array": i, "topic": a["topic"], "problem": f"(pass {a['pass']}, stamp {a['stamp_ns']}) not processed live"})
        elif a[field] != c[which] or a["n_total"] != 1 + c[which]:
            problems.append({"array": i, "topic": a["topic"], "stamp_ns": a["stamp_ns"],
                             "problem": f"{a[field]} markers ({a['n_total'] - 1} in all) but {c[which]} live tracks"})
    return {"arrays": len(arrays), "per_topic": per_topic, "problems": problems,
            "ok": not problems and len(per_topic) == 3}
