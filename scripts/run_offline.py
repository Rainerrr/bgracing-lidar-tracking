#!/usr/bin/env python3
"""Offline runner: the perception core and the tracker over every sweep of the drive, no ROS and no publisher.

    docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py
    docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py --set ego_compensation=false

Writes to /output/offline (/output/offline/no_ego with ego compensation off; never the live output folder). For a
drive other than the config file's (--set drive=...), the default is /output/<drive>/offline instead:
  detections.csv, frames.csv       deterministic per-frame rows (what the live node also writes)
  detection_recall.json            deterministic recall summary against the tracklets
  tracks.csv                       confirmed tracks per frame (columns as in the README; pass is 0 offline)
  tracking_speeds.json             deterministic speed summary (parked and moving) against the tracklets
  obstacle_tracking_report.png     top-down map of the tracks (byte-stable per architecture)
  metrics.json                     timing, platform, seed, parameters, track counts (not byte-stable, never hashed)

Poses: each frame's world -> velodyne pose goes through core.poses.carried_pose, the same float path the live
node takes from the odom message, so live and offline tracks can be compared byte for byte.
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

from core import kitti, poses
from processor import config, matching, perception, records, report, track_scoring, tracking

DEFAULT_CONFIG = "/config/tracker.yaml"
DEFAULT_OUT = "/output/offline"
DEFAULT_OUT_NO_EGO = "/output/offline/no_ego"
TIME_SCOPE = "detect() = pipeline steps 2-7 (ROI to detections); .bin reads and file writes excluded"
TRACK_SCOPE = "to_world() + Tracker.step() = pipeline steps 7 (world transform) to 11"


def default_out_dir(params, config_drive):
    """/output/offline (or offline/no_ego), under /output/<drive>/ when the drive is not the config file's."""
    root = config.output_root(params.drive, config_drive)
    return str(root / "offline" if params.ego_compensation else root / "offline" / "no_ego")


def carried_poses(drive):
    """T_world_velo per frame as the odom message carries it (see the module docstring)."""
    return np.array([poses.carried_pose(T) for T in poses.world_velo_poses(drive.oxts, drive.T_velo_imu)])


def tracking_pose(T_carried, frame, params):
    """The pose the tracker uses: identity with ego compensation off (the tracker then runs in velodyne)."""
    return T_carried[frame] if params.ego_compensation else np.eye(4)


def replay(params, frames_by_pass, data_dir=None):
    """Track replay: the tracker fed exactly the given frames, a fresh state per pass. {pass: [tracks.csv rows]}."""
    drive = kitti.load_drive(data_dir or params.data_dir, params.drive)
    T_c = carried_poses(drive)
    out = {}
    for pass_no, frames in sorted(frames_by_pass.items()):
        tr, rows = tracking.Tracker(params), []
        for f in frames:
            ns = int(drive.stamps_ns[f])
            dets = perception.detect(kitti.load_sweep(drive.sweep_paths[f]), ns, params).detections
            tracks = tr.step(f, ns, tracking.to_world(dets, tracking_pose(T_c, f, params)))
            rows += records.track_rows(pass_no, f, ns, tracks or [])
        out[pass_no] = rows
    return out


def run(params, out_dir, data_dir=None):
    """Run every frame, write the files into out_dir, return a dict with the in-memory results."""
    out_dir = Path(out_dir)
    if out_dir.resolve() == Path(params.output_dir).resolve():
        raise config.ConfigError(f"--out-dir {out_dir} is the live output folder; offline runs never overwrite live outputs")
    drive = kitti.load_drive(data_dir or params.data_dir, params.drive)
    T_c = carried_poses(drive)
    tracker = tracking.Tracker(params)

    det_rows, frame_rows, track_rows, ms, track_ms, dets_by_frame = [], [], [], [], [], {}
    confirmed_per_frame = []
    status_counts, stage_sums = {}, dict.fromkeys(perception.STAGES, 0)
    for frame, path in enumerate(drive.sweep_paths):
        points = kitti.load_sweep(path)
        stamp_ns = int(drive.stamps_ns[frame])
        t0 = time.perf_counter_ns()
        result = perception.detect(points, stamp_ns, params)
        t1 = time.perf_counter_ns()
        tracks = tracker.step(frame, stamp_ns, tracking.to_world(result.detections,
                                                                 tracking_pose(T_c, frame, params)))
        t2 = time.perf_counter_ns()
        ms.append((t1 - t0) / 1e6)
        track_ms.append((t2 - t1) / 1e6)
        dets_by_frame[frame] = result.detections
        det_rows += records.detection_rows(frame, stamp_ns, result.detections)
        frame_rows.append(records.frames_row(frame, stamp_ns, result))
        track_rows += records.track_rows(0, frame, stamp_ns, tracks)
        confirmed_per_frame.append(len(tracks))
        status_counts[result.ground_status] = status_counts.get(result.ground_status, 0) + 1
        for k in stage_sums:
            stage_sums[k] += result.stages[k]

    try:
        gt = kitti.gt_boxes(kitti.load_tracklets(drive.tracklet_path))
        recall = matching.score_detections(dets_by_frame, gt, params, n_frames=drive.n_frames)
        speeds = track_scoring.score(track_rows, gt, poses.world_velo_poses(drive.oxts, drive.T_velo_imu),
                                     drive.stamps_ns, params)
        tags = track_scoring.gt_tags(track_rows, gt, poses.world_velo_poses(drive.oxts, drive.T_velo_imu),
                                     drive.stamps_ns, params)
    except kitti.MissingDataError as exc:
        print(f"run_offline: ground truth missing ({exc}); writing detections and tracks without scores", file=sys.stderr)
        recall = {"ground_truth": "missing"}
        speeds = {"ground_truth": "missing"}
        tags = None

    n = drive.n_frames
    recall_doc = {"drive": params.drive, "frames": n, "seed": params.seed, "params": dataclasses.asdict(params), **recall}
    speeds_doc = {"drive": params.drive, "frames": n, "seed": params.seed, "params": dataclasses.asdict(params), **speeds}
    metrics = {
        "mode": "offline", "seed": params.seed, "params": dataclasses.asdict(params), "platform": records.platform_info(),
        "frames_expected": n, "frames_processed": n, "frames_dropped": 0, "first_frame_seen": 0,
        "unknown_stamps": 0, "passes": 1,
        "time_ms": records.time_stats(ms, TIME_SCOPE),
        "tracker_ms": records.time_stats(track_ms, TRACK_SCOPE),
        "stage_means": {k: round(v / n, 1) for k, v in stage_sums.items()},
        "ground_status_counts": status_counts,
        "recall_car_van": recall.get("car_van"),
        "tracks": {"confirmed": tracker.confirmed_total, "deleted": tracker.deleted_total,
                   "duplicate_stamps": tracker.duplicates,
                   "confirmed_per_frame_mean": round(sum(confirmed_per_frame) / n, 2),
                   "confirmed_per_frame_max": max(confirmed_per_frame)},
        "speeds_pass": {"parked": speeds.get("parked", {}).get("pass"), "moving": speeds.get("moving", {}).get("pass")},
    }
    records.atomic_write(out_dir / "detections.csv", records.csv_text(records.DETECTION_HEADER, det_rows))
    records.atomic_write(out_dir / "frames.csv", records.csv_text(records.FRAMES_HEADER, frame_rows))
    records.atomic_write(out_dir / "detection_recall.json", records.canonical_json(recall_doc))
    records.atomic_write(out_dir / "tracks.csv", records.csv_text(records.TRACK_HEADER, track_rows))
    records.atomic_write(out_dir / "tracking_speeds.json", records.canonical_json(speeds_doc))
    records.atomic_write(out_dir / "metrics.json", records.canonical_json(metrics))
    ego_xy = T_c[:, :2, 3] if params.ego_compensation else np.zeros((1, 2))
    report.render_report(out_dir / "obstacle_tracking_report.png", track_rows, ego_xy,
                         "world" if params.ego_compensation else "velodyne",
                         "" if params.ego_compensation else " (ego compensation OFF: ablation)", tags=tags)
    return {"out_dir": out_dir, "recall": recall, "speeds": speeds, "metrics": metrics, "gt_tags": tags,
            "dets_by_frame": dets_by_frame, "track_rows": track_rows, "time_ms": ms, "tracker_ms": track_ms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default=None, help=f"default {DEFAULT_OUT} ({DEFAULT_OUT_NO_EGO} with ego compensation off); "
                                                   "under /output/<drive>/ for a drive other than the config file's")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set ego_compensation=false (repeatable)")
    ap.add_argument("--data-dir", default=None, help="override data_dir from the config (tests)")
    args = ap.parse_args(argv)
    try:
        base = config.load_params(args.config)
        params = config.apply_overrides(base, args.set)
        out_dir = args.out_dir or default_out_dir(params, base.drive)
        result = run(params, out_dir, args.data_dir)
    except (config.ConfigError, kitti.DataError) as exc:
        print(f"run_offline: {exc}", file=sys.stderr)
        return 1
    m, r = result["metrics"], result["recall"]
    t = m["time_ms"]
    print(f"wrote {out_dir}: {m['frames_processed']} frames, time p95 {t['p95']} ms (mean {t['mean']}, max {t['max']})")
    if r.get("ground_truth") == "present":
        cv = r["car_van"]
        print(f"Car+Van recall {cv['recall']:.4f} ({cv['matched']}/{cv['total']})")
    sp = result["speeds"]
    tk = m["tracks"]
    print(f"tracks: {tk['confirmed']} confirmed, {tk['confirmed_per_frame_mean']} per frame (max {tk['confirmed_per_frame_max']}); "
          f"tracker p95 {m['tracker_ms']['p95']} ms")
    if sp.get("ground_truth") == "present":
        pk, mv = sp["parked"], sp["moving"]
        print(f"parked ({sp['tracking_frame']}): median of medians {pk['median_of_medians']} m/s, worst #{pk['worst']['id']} "
              f"{pk['worst']['median_speed']} m/s -> {'PASS' if pk['pass'] else 'FAIL'} (bars {pk['bar_median_of_medians']} / {pk['bar_each']})")
        for tid, o in mv["objects"].items():
            print(f"moving #{tid} {o['cls']}: median |error| {o['median_abs_error']} m/s over {o['matched_frames']} frames "
                  f"(GT median {o['median_gt_speed']} m/s)")
        print(f"moving -> {'PASS' if mv['pass'] else 'FAIL'} (bar {mv['bar_median_abs_error']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
