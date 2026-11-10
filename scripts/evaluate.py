#!/usr/bin/env python3
"""Evaluation: the offline runner with and without ego compensation, scored against the KITTI tracklets.

    docker compose run --rm --no-deps processor python3 /app/scripts/evaluate.py

No publisher and no Foxglove. Runs scripts/run_offline.py in-process twice (its normal outputs go to
/output/offline and /output/offline/no_ego), evaluates the returned tracks in memory (processor/evaluation.py)
and writes /output/evaluation.md atomically. For a drive other than the config file's (--set drive=...), all of
these default to /output/<drive>/ instead. Two runs give byte-identical files on the same CPU architecture.

Exit codes: 0 on success, also when the ablation gate fails (the file is the evidence; verify enforces the gate);
1 on a missing drive or tracklet file (naming scripts/download_kitti.sh), a bad config or --set, or an unwritable
output path. A missing tracklet file stops before anything is run or written.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

from core import kitti, poses
from processor import config, evaluation, records
from scripts import run_offline

DEFAULT_OUT = "/output/evaluation.md"
DEFAULT_OFFLINE = "/output/offline"


def run_evaluation(params, data_dir=None, offline_dir=DEFAULT_OFFLINE, log=print, config_drive=None):
    """Both runs, both evaluations, the ablation and the rendered text. Returns a dict."""
    drive = kitti.load_drive(data_dir or params.data_dir, params.drive)
    gt = kitti.gt_boxes(kitti.load_tracklets(drive.tracklet_path))  # before any run: a missing file stops here
    T_wv = poses.world_velo_poses(drive.oxts, drive.T_velo_imu)
    results = {}
    for i, (name, ego) in enumerate((("on", True), ("off", False)), 1):
        p = dataclasses.replace(params, ego_compensation=ego)
        log(f"run {i}/2: ego compensation {name}")
        out = Path(offline_dir) if ego else Path(offline_dir) / "no_ego"
        r = run_offline.run(p, out, data_dir)
        results[name] = evaluation.evaluate(r["track_rows"], r["dets_by_frame"], gt, T_wv, drive.stamps_ns, p, drive.n_frames)
    abl = evaluation.ablation(results["on"], results["off"])
    text = evaluation.render(results["on"], results["off"], abl, evaluation.render_meta(params, evaluation.ego_speed_range(drive.oxts), config_drive))
    return {"text": text, "ablation": abl, "on": results["on"], "off": results["off"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=run_offline.DEFAULT_CONFIG)
    ap.add_argument("--out", default=None, help=f"the evaluation file (default {DEFAULT_OUT}; /output/<drive>/evaluation.md "
                                                 "for a drive other than the config file's)")
    ap.add_argument("--offline-dir", default=None,
                    help=f"where the two offline runs write (default {DEFAULT_OFFLINE}, or /output/<drive>/offline; "
                         "the no-ego run goes to <dir>/no_ego)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (repeatable)")
    ap.add_argument("--data-dir", default=None, help="override data_dir from the config (tests)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(DEFAULT_OUT)
    try:
        base = config.load_params(args.config)
        params = config.apply_overrides(base, args.set)
        root = config.output_root(params.drive, base.drive)
        out = Path(args.out) if args.out else root / Path(DEFAULT_OUT).name
        offline_dir = args.offline_dir or str(root / Path(DEFAULT_OFFLINE).name)
        out.parent.mkdir(parents=True, exist_ok=True)  # an unwritable path fails now, before the two runs
        res = run_evaluation(params, args.data_dir, offline_dir, config_drive=base.drive)
        records.atomic_write(out, res["text"])
    except (config.ConfigError, kitti.DataError, ValueError) as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"evaluate: cannot write {out}: {exc}", file=sys.stderr)
        return 1
    a = res["ablation"]
    ratio = "n/a" if a["ratio"] is None else f"{a['ratio']:.2f}x"
    print(f"wrote {out}")
    print(f"gate: static-object velocity RMSE off/on = {ratio} (bar {a['bar']:.1f}x) -> {'PASS' if a['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
