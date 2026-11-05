#!/usr/bin/env python3
"""Offline runner: the perception core over every sweep of the drive, no ROS and no publisher.

    docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py

Writes to /output/offline (never the live output folder):
  detections.csv, frames.csv       deterministic per-frame rows (what the live node also writes)
  detection_recall.json            deterministic recall summary against the tracklets
  metrics.json                     timing, platform, seed and parameters (not byte-stable, never hashed)
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

from core import kitti
from processor import config, matching, perception, records

DEFAULT_CONFIG = "/config/tracker.yaml"
DEFAULT_OUT = "/output/offline"
TIME_SCOPE = "detect() = pipeline steps 2-7 (ROI to detections); .bin reads and file writes excluded"


def run(params, out_dir, data_dir=None):
    """Run every frame, write the four files into out_dir, return a dict with the in-memory results."""
    out_dir = Path(out_dir)
    if out_dir.resolve() == Path(params.output_dir).resolve():
        raise config.ConfigError(f"--out-dir {out_dir} is the live output folder; offline runs never overwrite live outputs")
    drive = kitti.load_drive(data_dir or params.data_dir, params.drive)

    det_rows, frame_rows, ms, dets_by_frame = [], [], [], {}
    status_counts, stage_sums = {}, dict.fromkeys(perception.STAGES, 0)
    for frame, path in enumerate(drive.sweep_paths):
        points = kitti.load_sweep(path)
        stamp_ns = int(drive.stamps_ns[frame])
        t0 = time.perf_counter_ns()
        result = perception.detect(points, stamp_ns, params)
        ms.append((time.perf_counter_ns() - t0) / 1e6)
        dets_by_frame[frame] = result.detections
        det_rows += records.detection_rows(frame, stamp_ns, result.detections)
        frame_rows.append(records.frames_row(frame, stamp_ns, result))
        status_counts[result.ground_status] = status_counts.get(result.ground_status, 0) + 1
        for k in stage_sums:
            stage_sums[k] += result.stages[k]

    try:
        gt = kitti.gt_boxes(kitti.load_tracklets(drive.tracklet_path))
        recall = matching.score_detections(dets_by_frame, gt, params, n_frames=drive.n_frames)
    except kitti.MissingDataError as exc:
        print(f"run_offline: ground truth missing ({exc}); writing detections without recall", file=sys.stderr)
        recall = {"ground_truth": "missing"}

    n = drive.n_frames
    recall_doc = {"drive": params.drive, "frames": n, "seed": params.seed, "params": dataclasses.asdict(params), **recall}
    metrics = {
        "mode": "offline", "seed": params.seed, "params": dataclasses.asdict(params), "platform": records.platform_info(),
        "frames_expected": n, "frames_processed": n, "frames_dropped": 0, "first_frame_seen": 0,
        "unknown_stamps": 0, "passes": 1,
        "time_ms": records.time_stats(ms, TIME_SCOPE),
        "stage_means": {k: round(v / n, 1) for k, v in stage_sums.items()},
        "ground_status_counts": status_counts,
        "recall_car_van": recall.get("car_van"),
    }
    records.atomic_write(out_dir / "detections.csv", records.csv_text(records.DETECTION_HEADER, det_rows))
    records.atomic_write(out_dir / "frames.csv", records.csv_text(records.FRAMES_HEADER, frame_rows))
    records.atomic_write(out_dir / "detection_recall.json", records.canonical_json(recall_doc))
    records.atomic_write(out_dir / "metrics.json", records.canonical_json(metrics))
    return {"out_dir": out_dir, "recall": recall, "metrics": metrics, "dets_by_frame": dets_by_frame, "time_ms": ms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--data-dir", default=None, help="override data_dir from the config (tests)")
    args = ap.parse_args(argv)
    try:
        result = run(config.load_params(args.config), args.out_dir, args.data_dir)
    except (config.ConfigError, kitti.DataError) as exc:
        print(f"run_offline: {exc}", file=sys.stderr)
        return 1
    m, r = result["metrics"], result["recall"]
    t = m["time_ms"]
    print(f"wrote {args.out_dir}: {m['frames_processed']} frames, time p95 {t['p95']} ms (mean {t['mean']}, max {t['max']})")
    if r.get("ground_truth") == "present":
        cv = r["car_van"]
        print(f"Car+Van recall {cv['recall']:.4f} ({cv['matched']}/{cv['total']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
