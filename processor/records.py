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


# ---------------------------------------------------------------- rows

def detection_rows(frame, stamp_ns, detections):
    return [f"{frame},{stamp_ns},{d.x:.4f},{d.y:.4f},{d.z:.4f},{d.dx:.4f},{d.dy:.4f},{d.dz:.4f},{d.n_points}"
            for d in detections]


def frames_row(frame, stamp_ns, result):
    s = result.stages
    return (f"{frame},{stamp_ns},{s['raw']},{s['roi']},{s['voxel']},{s['non_ground']},{s['clustered']},"
            f"{result.clusters_found},{result.clusters_kept},{result.ground_status}")


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
