#!/usr/bin/env python3
"""Live checks for E2.5 (markers) and E2.6 (parity). Needs a live run and an offline reference.

    # 1. offline reference, then the live pipeline
    docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py
    docker compose up -d
    # 2. record the MarkerArrays that /perception/bounding_boxes publishes (ROS must be sourced)
    docker compose exec -T processor bash -c 'source /opt/ros/jazzy/setup.bash; python3 /app/scripts/live_check.py record --duration 40'
    # 3. stop (the node writes its final files on SIGINT), then compare
    docker compose stop
    docker compose run --rm --no-deps processor python3 /app/scripts/live_check.py compare

compare exits 0 only if every recorded array begins with DELETEALL and matches the live detection count for its
stamp (E2.5), and every live frame equals the offline rows of that frame (E2.6, at least 140 distinct frames).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from processor import records

MIN_FRAMES = 140


def record(duration, out_path):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from visualization_msgs.msg import Marker, MarkerArray

    arrays = []

    def on_array(msg):
        kinds = [m.type for m in msg.markers if m.action != Marker.DELETEALL]
        stamp = msg.markers[0].header.stamp if msg.markers else None
        arrays.append({
            "stamp_ns": stamp.sec * 1_000_000_000 + stamp.nanosec if stamp else -1,
            "first_action": msg.markers[0].action if msg.markers else -1,
            "n_line_list": sum(k == Marker.LINE_LIST for k in kinds),
            "n_text": sum(k == Marker.TEXT_VIEW_FACING for k in kinds),
            "n_total": len(msg.markers),
        })

    rclpy.init()
    node = Node("live_check_recorder")
    node.create_subscription(MarkerArray, "/perception/bounding_boxes", on_array,
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
    for f in (live_dir / "detections.csv", live_dir / "frames.csv", offline_dir / "detections.csv",
              offline_dir / "frames.csv", Path(markers_path)):
        if not f.is_file():
            print(f"live_check: {f} not found (run the offline runner, the live pipeline and 'record' first)", file=sys.stderr)
            return 1
    live = records.read_live_rows(live_dir / "detections.csv", live_dir / "frames.csv")
    offline = records.read_offline_rows(offline_dir / "detections.csv", offline_dir / "frames.csv")
    arrays = json.loads(Path(markers_path).read_text())["arrays"]
    markers = records.compare_markers(arrays, live)
    parity = records.compare_parity(live, offline, min_frames=MIN_FRAMES)
    metrics = json.loads((live_dir / "metrics.json").read_text()) if (live_dir / "metrics.json").is_file() else {}
    print(f"E2.5 markers: {markers['arrays']} arrays checked, {len(markers['problems'])} problems -> {'OK' if markers['ok'] else 'FAIL'}")
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
    args = ap.parse_args(argv)
    return record(args.duration, args.out) if args.cmd == "record" else compare(args.live_dir, args.offline_dir, args.markers)


if __name__ == "__main__":
    sys.exit(main())
