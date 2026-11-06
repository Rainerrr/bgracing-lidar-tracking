#!/usr/bin/env python3
"""Live checks: markers and E2.6 detection parity (compare), E3.4 track parity and E3.5 loop (tracks), and
E3.6 output files (files). Needs a live run; compare also needs the offline reference.

    # 1. offline reference, then the live pipeline
    docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py
    docker compose up -d
    # 2. record the MarkerArrays of the three track topics (ROS must be sourced)
    docker compose exec -T processor bash -c 'source /opt/ros/jazzy/setup.bash; python3 /app/scripts/live_check.py record --duration 40'
    # 3. stop (the node writes its final files on SIGINT), then compare
    docker compose stop
    docker compose run --rm --no-deps processor python3 /app/scripts/live_check.py compare
    docker compose run --rm --no-deps processor python3 /app/scripts/live_check.py tracks
    docker compose run --rm --no-deps processor python3 /app/scripts/live_check.py files

compare exits 0 only if every recorded array begins with DELETEALL and matches the live tracks of its (pass,
stamp) (boxes and labels: one per confirmed track; arrows: one per moving track), and every live frame equals the
offline detection rows of that frame (E2.6, at least 140 distinct frames).
tracks exits 0 only if, for every live pass, the offline tracker fed exactly the frames the live node processed
gives byte-equal tracks.csv rows (E3.4), and the pass changes exactly at wraps with no track surviving one (E3.5).
files exits 0 only if tracks.csv, metrics.json and the report PNG are complete, parseable and agree (E3.6).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from processor import records

MIN_FRAMES = 140
TOPICS = ("/perception/bounding_boxes", "/perception/obstacle_vectors", "/perception/track_labels")


def record(duration, out_path):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from visualization_msgs.msg import Marker, MarkerArray

    arrays = []

    def on_array(topic, msg):
        kinds = [m.type for m in msg.markers if m.action != Marker.DELETEALL]
        stamp = msg.markers[0].header.stamp if msg.markers else None
        arrays.append({
            "topic": topic,
            "pass": msg.markers[0].id if msg.markers else -1,  # the DELETEALL marker carries the pass
            "frame_id": msg.markers[0].header.frame_id if msg.markers else "",
            "stamp_ns": stamp.sec * 1_000_000_000 + stamp.nanosec if stamp else -1,
            "first_action": msg.markers[0].action if msg.markers else -1,
            "n_line_list": sum(k == Marker.LINE_LIST for k in kinds),
            "n_text": sum(k == Marker.TEXT_VIEW_FACING for k in kinds),
            "n_arrow": sum(k == Marker.ARROW for k in kinds),
            "n_total": len(msg.markers),
        })

    rclpy.init()
    node = Node("live_check_recorder")
    for topic in TOPICS:
        node.create_subscription(MarkerArray, topic, lambda m, t=topic: on_array(t, m),
                                 QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
    end = time.monotonic() + duration
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    records.atomic_write(out_path, json.dumps({"arrays": arrays}, indent=1) + "\n")
    print(f"recorded {len(arrays)} MarkerArrays over {duration:.0f} s -> {out_path}")
    return 0 if arrays else 1


def compare(live_dir, offline_dir, markers_path):
    live_dir, offline_dir = Path(live_dir), Path(offline_dir)
    for f in (live_dir / "detections.csv", live_dir / "frames.csv", live_dir / "tracks.csv", offline_dir / "detections.csv",
              offline_dir / "frames.csv", Path(markers_path)):
        if not f.is_file():
            print(f"live_check: {f} not found (run the offline runner, the live pipeline and 'record' first)", file=sys.stderr)
            return 1
    live = records.read_live_rows(live_dir / "detections.csv", live_dir / "frames.csv")
    offline = records.read_offline_rows(offline_dir / "detections.csv", offline_dir / "frames.csv")
    arrays = json.loads(Path(markers_path).read_text())["arrays"]
    markers = records.compare_track_markers(arrays, live, records.read_live_tracks(live_dir / "tracks.csv"))
    parity = records.compare_parity(live, offline, min_frames=MIN_FRAMES)
    metrics = json.loads((live_dir / "metrics.json").read_text()) if (live_dir / "metrics.json").is_file() else {}
    print(f"markers: {markers['arrays']} arrays checked {markers['per_topic']}, {len(markers['problems'])} problems "
          f"-> {'OK' if markers['ok'] else 'FAIL'}")
    for p in markers["problems"][:10]:
        print("   ", p)
    print(f"E2.6 parity:  {parity['compared']} live (pass, frame) pairs, {parity['distinct_frames']} distinct frames "
          f"(need >= {MIN_FRAMES}), {len(parity['mismatched'])} mismatched -> {'OK' if parity['ok'] else 'FAIL'}")
    for m in parity["mismatched"][:10]:
        print("   ", m)
    if metrics:
        t, c = metrics.get("time_ms", {}), metrics.get("callback_ms", {})
        print(f"live run: processed={metrics.get('frames_processed')} expected={metrics.get('frames_expected')} "
              f"dropped={metrics.get('frames_dropped')} first_frame_seen={metrics.get('first_frame_seen')} "
              f"passes={metrics.get('passes')} detect_p95={t.get('p95')} ms callback_p95={c.get('p95')} ms "
              f"(100 ms period; not gated)")
    consistent = not metrics or metrics.get("frames_processed") == len(live)
    print(f"consistency:  metrics.json frames_processed={metrics.get('frames_processed')} vs {len(live)} rows in frames.csv "
          f"-> {'OK' if consistent else 'FAIL'}")
    return 0 if markers["ok"] and parity["ok"] and consistent else 1


def tracks(live_dir, config_path, data_dir=None):
    """E3.4 replay parity and E3.5 loop (needs the drive: the replay re-runs detection and tracking)."""
    from processor import config
    from scripts import run_offline
    live_dir = Path(live_dir)
    for f in (live_dir / "tracks.csv", live_dir / "metrics.json"):
        if not f.is_file():
            print(f"live_check: {f} not found (run the live pipeline and stop it first)", file=sys.stderr)
            return 1
    metrics = json.loads((live_dir / "metrics.json").read_text())
    processed = [tuple(x) for x in metrics.get("processed_frames", [])]
    live_tracks = records.read_live_tracks(live_dir / "tracks.csv")
    params = config.load_params(config_path)  # the file the live node read (metrics.json holds rounded copies)
    replay = run_offline.replay(params, records.frames_by_pass(processed), data_dir)
    parity = records.compare_track_parity(records.tracks_by_pass(processed, live_tracks), replay)
    loop = records.check_loop(processed, live_tracks)
    by_pass = records.frames_by_pass(processed)
    print(f"E3.4 parity: {parity['passes']} live passes, {parity['rows']} track rows, "
          f"{len(parity['mismatched'])} mismatched passes -> {'OK' if parity['ok'] else 'FAIL'}")
    for m in parity["mismatched"][:5]:
        print("   ", m)
    for p, fs in sorted(by_pass.items()):
        span = fs[-1] - fs[0] + 1
        print(f"    pass {p}: frames {fs[0]}..{fs[-1]}, processed {len(fs)}, gaps {span - len(fs)}")
    print(f"    clean full passes (frames 0..153, no gap): {metrics.get('clean_passes')}; dropped {metrics.get('frames_dropped')}, "
          f"no_odom {metrics.get('no_odom')}, duplicates {metrics.get('duplicate_stamps')} (reported, not gated)")
    print(f"E3.5 loop: {loop['wraps']} wraps, {len(loop['problems'])} problems -> {'OK' if loop['ok'] and loop['wraps'] >= 1 else 'FAIL'}")
    for pr in loop["problems"][:5]:
        print("   ", pr)
    return 0 if parity["ok"] and loop["ok"] and loop["wraps"] >= 1 else 1


def files(live_dir):
    """E3.6: the files a stopped run leaves are complete, parseable and agree with each other."""
    import csv
    live_dir = Path(live_dir)
    problems, rows = [], []
    try:
        metrics = json.loads((live_dir / "metrics.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"E3.6 files: metrics.json unreadable ({exc}) -> FAIL")
        return 1
    try:
        with open(live_dir / "tracks.csv", newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows or ",".join(rows[0]) != records.TRACK_HEADER:
            problems.append("tracks.csv header is not the contract")
        width = len(records.TRACK_HEADER.split(","))
        bad = [i for i, r in enumerate(rows[1:], 2) if len(r) != width or r[9] not in ("moving", "static")]
        if bad:
            problems.append(f"tracks.csv rows malformed at lines {bad[:5]}")
        frames = (live_dir / "frames.csv").read_text().splitlines()[1:]
        if len(frames) != metrics.get("frames_processed"):
            problems.append(f"frames.csv has {len(frames)} rows, metrics.json says {metrics.get('frames_processed')}")
        if len(rows) - 1 != metrics.get("track_rows"):
            problems.append(f"tracks.csv has {len(rows) - 1} rows, metrics.json says {metrics.get('track_rows')}")
        if rows[1:]:
            last = (int(rows[-1][0]), int(rows[-1][1]))
            if list(last) not in metrics.get("processed_frames", []):
                problems.append(f"last tracks.csv row {last} is not a processed frame")
    except OSError as exc:
        problems.append(str(exc))
    png = live_dir / "obstacle_tracking_report.png"
    try:
        import matplotlib.image
        data = png.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            problems.append("report PNG has no PNG signature")
        else:
            img = matplotlib.image.imread(str(png))
            if img.ndim != 3 or img.shape[0] < 100:
                problems.append(f"report PNG decodes to shape {img.shape}")
    except Exception as exc:  # any decoding failure is the finding
        problems.append(f"report PNG unreadable: {exc!r}")
    print(f"E3.6 files: tracks.csv {max(len(rows) - 1, 0)} rows, metrics.json processed "
          f"{metrics.get('frames_processed')}, PNG {png.stat().st_size if png.is_file() else 0} bytes; "
          f"{len(problems)} problems -> {'OK' if not problems else 'FAIL'}")
    for pr in problems:
        print("   ", pr)
    return 0 if not problems else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--duration", type=float, default=40.0)
    r.add_argument("--out", default="/output/live_markers.json")
    c = sub.add_parser("compare")
    c.add_argument("--live-dir", default="/output")
    c.add_argument("--offline-dir", default="/output/offline")
    c.add_argument("--markers", default="/output/live_markers.json")
    t = sub.add_parser("tracks")
    t.add_argument("--live-dir", default="/output")
    t.add_argument("--config", default="/config/tracker.yaml")
    t.add_argument("--data-dir", default=None)
    f = sub.add_parser("files")
    f.add_argument("--live-dir", default="/output")
    args = ap.parse_args(argv)
    if args.cmd == "record":
        return record(args.duration, args.out)
    if args.cmd == "tracks":
        return tracks(args.live_dir, args.config, args.data_dir)
    if args.cmd == "files":
        return files(args.live_dir)
    return compare(args.live_dir, args.offline_dir, args.markers)


if __name__ == "__main__":
    sys.exit(main())
