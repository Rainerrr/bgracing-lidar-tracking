#!/usr/bin/env python3
"""Dataset statistics for the README: writes dataset_stats.json and dataset_stats.png.

Runs without ROS and without the publisher:

    docker compose run --rm --no-deps processor python3 /app/scripts/dataset_stats.py

The JSON is the checked artifact: sorted keys, fixed histogram edges and rounded
floats, so two runs give byte-identical files. If tracklet_labels.xml is missing,
the stats still run and say "ground_truth": "missing".

Phase 2 adds the per-stage point counts, computed by the same perception core the live node uses.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core import kitti, poses  # noqa: E402
from processor import config, perception  # noqa: E402

DEFAULT_DRIVE = "2011_09_26_drive_0005"
DEFAULT_CONFIG = "/config/tracker.yaml"
STAGE_LABELS = {"raw": "raw", "roi": "after ROI", "voxel": "after voxel", "non_ground": "after ground removal",
                "clustered": "in kept clusters"}
RANGE_EDGES = np.arange(0.0, 100.0 + 1e-9, 2.0)  # metres from the sensor (3D); data max is 80 m
REFLECTANCE_EDGES = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
CLASS_ORDER = ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc")
LINE_STYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "D", "v", "x", "+", "*")


def r(x, nd=4):
    return round(float(x), nd)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def compute_stats(data_dir, drive_name, params):
    drive = kitti.load_drive(data_dir, drive_name)
    n = drive.n_frames
    stamps_s = (drive.stamps_ns - drive.stamps_ns[0]) / 1e9
    periods = np.diff(stamps_s)

    counts = np.zeros(n, dtype=np.int64)
    stage_series = {k: np.zeros(n, dtype=np.int64) for k in perception.STAGES}
    ground_status = {}
    range_hist = np.zeros(len(RANGE_EDGES) - 1, dtype=np.int64)
    refl_hist = np.zeros(len(REFLECTANCE_EDGES) - 1, dtype=np.int64)
    beyond = 0
    for i, path in enumerate(drive.sweep_paths):
        pts = kitti.load_sweep(path)
        counts[i] = len(pts)
        result = perception.detect(pts, int(drive.stamps_ns[i]), params)  # same code as the live node
        for k in perception.STAGES:
            stage_series[k][i] = result.stages[k]
        ground_status[result.ground_status] = ground_status.get(result.ground_status, 0) + 1
        rng = np.linalg.norm(pts[:, :3].astype(np.float64), axis=1)
        range_hist += np.histogram(rng, bins=RANGE_EDGES)[0]
        beyond += int(np.count_nonzero(rng > RANGE_EDGES[-1]))
        refl_hist += np.histogram(pts[:, 3], bins=REFLECTANCE_EDGES)[0]

    speed = np.hypot(drive.oxts[:, poses.VN], drive.oxts[:, poses.VE])

    date_dir = drive.root.parent
    inputs = {str(p.relative_to(date_dir)): sha256(p) for p in (
        drive.root / "velodyne_points" / "timestamps.txt", date_dir / "calib_imu_to_velo.txt", drive.tracklet_path)}

    stats = {
        "drive": drive_name,
        "generated_by": "scripts/dataset_stats.py",
        "inputs_sha256": inputs,
        "frames": n,
        "duration_s": r(stamps_s[-1], 3),
        "mean_rate_hz": r((n - 1) / stamps_s[-1], 3) if n > 1 else None,
        "sensor_period_s": {"min": r(periods.min()), "max": r(periods.max())} if n > 1 else None,
        "points_per_frame": {"mean": r(counts.mean(), 1), "min": int(counts.min()), "max": int(counts.max()),
                             "series": counts.tolist()},
        "range_hist": {"unit": "m", "edges": [r(e, 2) for e in RANGE_EDGES], "counts": range_hist.tolist(),
                       "beyond_last_edge": beyond},
        "reflectance_hist": {"edges": [r(e, 2) for e in REFLECTANCE_EDGES], "counts": refl_hist.tolist()},
        "ego_speed_mps": {"source": "OXTS hypot(vn, ve)", "min": r(speed.min(), 3), "mean": r(speed.mean(), 3),
                          "max": r(speed.max(), 3), "series": [r(s, 3) for s in speed]},
        "time_s": [r(t, 4) for t in stamps_s],
        "stage_points": {k: {"mean": r(v.mean(), 1), "min": int(v.min()), "max": int(v.max()),
                             "share_of_raw": r(v.sum() / counts.sum(), 4), "series": v.tolist()}
                         for k, v in stage_series.items()},
        "ground_status_counts": dict(sorted(ground_status.items())),
        "perception_seed": params.seed,
    }

    try:
        tracklets = kitti.load_tracklets(drive.tracklet_path)
    except kitti.MissingDataError as exc:
        print(f"dataset_stats: ground truth missing ({exc}); writing the other statistics", file=sys.stderr)
        stats.update(ground_truth="missing", gt_objects_per_frame=None, gt_tracklets_by_class=None)
        return stats

    classes = sorted({t.cls for t in tracklets}, key=lambda c: (CLASS_ORDER.index(c) if c in CLASS_ORDER else 99, c))
    per_frame = {c: [0] * n for c in classes}
    for box in kitti.gt_boxes(tracklets):
        if 0 <= box.frame < n:
            per_frame[box.cls][box.frame] += 1
    stats.update(
        ground_truth="present",
        gt_objects_per_frame=per_frame,
        gt_tracklets_by_class={c: sum(t.cls == c for t in tracklets) for c in classes},
        gt_poses_total=sum(len(t.poses) for t in tracklets),
    )
    return stats


def build_figure(stats):
    """Six labeled panels. Every panel has a title; every chart has axis labels with units."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    (ax_sum, ax_pts, ax_rng), (ax_refl, ax_speed, ax_gt) = axes
    t = np.array(stats["time_s"])

    ax_sum.set_title(f"Drive summary: {stats['drive']}")
    ax_sum.axis("off")
    period = stats["sensor_period_s"]
    ppf, speed = stats["points_per_frame"], stats["ego_speed_mps"]
    lines = [
        f"Frames (sweeps)   {stats['frames']}",
        f"Duration          {stats['duration_s']:.2f} s",
        f"Mean sensor rate  {stats['mean_rate_hz']:.2f} Hz",
        f"Sensor period     {period['min'] * 1e3:.1f}-{period['max'] * 1e3:.1f} ms" if period else "Sensor period     n/a",
        f"Points per frame  mean {ppf['mean']:,.0f}",
        f"                  min {ppf['min']:,}, max {ppf['max']:,}",
        f"Ego speed         {speed['min']:.1f}-{speed['max']:.1f} m/s, mean {speed['mean']:.1f}",
        "Points kept (share of raw):",
        *[f"  {STAGE_LABELS[k]:<21}{v['share_of_raw'] * 100:5.1f} %" for k, v in stats["stage_points"].items()],
    ]
    if stats["ground_truth"] == "present":
        lines.append(f"GT tracklets      {sum(stats['gt_tracklets_by_class'].values())}, "
                     f"{stats['gt_poses_total']} poses")
        lines += [f"                  {c}: {k}" for c, k in stats["gt_tracklets_by_class"].items()]
    else:
        lines += ["Ground truth      MISSING", "                  (tracklet_labels.xml not found)"]
    text = ax_sum.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace",
                       transform=ax_sum.transAxes, linespacing=1.5)
    text.set_in_layout(False)  # keep the text panel from squeezing the charts

    ax_pts.set_title("Points per frame after each pipeline stage")
    for k, (stage, info) in enumerate(stats["stage_points"].items()):
        ax_pts.plot(np.arange(stats["frames"]), np.maximum(info["series"], 1), LINE_STYLES[k % 4], marker=MARKERS[k % 8],
                    markevery=20, markersize=4, label=STAGE_LABELS[stage])
    ax_pts.set_yscale("log")
    ax_pts.set_ylim(2e3, 1.5e6)  # headroom above the data for the legend
    ax_pts.legend(loc="upper center", ncol=2, fontsize=8, title="stage (line style)", title_fontsize=8)
    ax_pts.set_xlabel("frame (index, ≈0.103 s apart)")
    ax_pts.set_ylabel("points (count, log scale)")

    edges = np.array(stats["range_hist"]["edges"])
    ax_rng.set_title("Range histogram (all frames)")
    ax_rng.bar(edges[:-1], np.array(stats["range_hist"]["counts"]) / 1e6, width=np.diff(edges), align="edge",
               color="0.45", edgecolor="0.15")
    ax_rng.set_xlabel("distance from sensor (m)")
    ax_rng.set_ylabel("points (millions)")

    edges = np.array(stats["reflectance_hist"]["edges"])
    ax_refl.set_title("Reflectance histogram (all frames)")
    ax_refl.bar(edges[:-1], np.array(stats["reflectance_hist"]["counts"]) / 1e6, width=np.diff(edges),
                align="edge", color="0.45", edgecolor="0.15")
    ax_refl.set_xlabel("reflectance (unitless, 0-1)")
    ax_refl.set_ylabel("points (millions)")

    ax_speed.set_title("Ego speed over the drive (OXTS)")
    ax_speed.plot(t, stats["ego_speed_mps"]["series"], "-", color="0.2")
    ax_speed.set_xlabel("time since first sweep (s)")
    ax_speed.set_ylabel("speed (m/s)")

    ax_gt.set_title("Ground-truth objects per frame")
    ax_gt.set_xlabel("frame (index, ≈0.103 s apart)")
    ax_gt.set_ylabel("objects (count)")
    if stats["ground_truth"] == "present":
        for k, (cls, series) in enumerate(stats["gt_objects_per_frame"].items()):
            n_trk = stats["gt_tracklets_by_class"][cls]
            label = f"{cls} ({n_trk} tracklet{'' if n_trk == 1 else 's'})"
            ax_gt.plot(np.arange(stats["frames"]), series, LINE_STYLES[k % 4], marker=MARKERS[k % 8], markevery=15,
                       label=label)
        ax_gt.legend(loc="upper left", fontsize=9, title="class (line style)", title_fontsize=9)
    else:
        ax_gt.text(0.5, 0.5, "Ground truth missing:\ntracklet_labels.xml not found", ha="center", va="center",
                   fontsize=12, transform=ax_gt.transAxes)

    fig.suptitle(f"KITTI raw {stats['drive']} (synced + rectified): dataset statistics", fontsize=13)
    fig.tight_layout()
    return fig


def write_outputs(stats, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    fig = build_figure(stats)
    fig.savefig(out_dir / "dataset_stats.png", dpi=100, metadata={"Software": None})
    plt.close(fig)
    return out_dir / "dataset_stats.json", out_dir / "dataset_stats.png"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--drive", default=DEFAULT_DRIVE)
    ap.add_argument("--out-dir", default="/output")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="perception parameters (per-stage counts)")
    args = ap.parse_args(argv)
    try:
        stats = compute_stats(args.data_dir, args.drive, config.load_params(args.config))
    except (kitti.DataError, config.ConfigError) as exc:
        print(f"dataset_stats: {exc}", file=sys.stderr)
        return 1
    for path in write_outputs(stats, args.out_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
