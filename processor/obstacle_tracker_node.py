#!/usr/bin/env python3
"""Live detection node (Phase 2): sweep in, boxes out. Replaces the Phase 1 cloud probe.

Subscribes to /sensor/lidar/raw_points (best effort, newest only: a slow frame never builds a backlog),
runs processor.perception.detect (the same code as scripts/run_offline.py), and publishes the filtered
cloud and the box markers in the velodyne frame with the cloud's stamp. It writes the same detection and
stage files as the offline runner (plus a leading 'pass' column) and metrics.json, every report_every_n
processed frames and on shutdown.
"""
import dataclasses
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import MarkerArray

from core import kitti
from processor import config, markers, perception, records

DETECT_SCOPE = "detect() = pipeline steps 2-7 (ROI to detections)"
CALLBACK_SCOPE = "whole cloud callback: decode, detect(), publish, row formatting"
IGNORED_PARAMS = {"use_sim_time", "start_type_description_service"}  # declared by rclpy itself


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def params_from_node(node):
    raw = {}
    for name, param in node.get_parameters_by_prefix("").items():
        if name in IGNORED_PARAMS or name.startswith("qos_overrides"):
            continue
        v = param.value
        raw[name] = list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
    return config.params_from_dict(raw)


class ObstacleTrackerNode(Node):
    def __init__(self):
        super().__init__("obstacle_tracker_node", automatically_declare_parameters_from_overrides=True)
        self.p = params_from_node(self)
        self.drive = kitti.load_drive(self.p.data_dir, self.p.drive)
        self.out = self.p.output_dir

        # One record per processed frame, appended in a single step; every counter, file and statistic is
        # derived from this log. SIGINT can interrupt the callback at any bytecode, so the bookkeeping must
        # not be spread over several updates (that once left metrics.json one frame ahead of the CSVs).
        self.log = []
        self.pass_no, self.last_stamp = 0, None
        self.clouds = self.unknown_stamps = 0

        self.cloud_pub = self.create_publisher(PointCloud2, "/perception/filtered_points", qos_profile_sensor_data)
        self.box_pub = self.create_publisher(MarkerArray, "/perception/bounding_boxes",
                                             QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PointCloud2, "/sensor/lidar/raw_points", self.on_cloud,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(10.0, self.on_wait_timer)
        self.write_outputs()  # a zero-frame metrics.json exists from the start
        self.get_logger().info(
            f"perception ready: {self.drive.n_frames} frames known, seed={self.p.seed}, voxel={self.p.voxel_size} m, "
            f"dbscan eps={self.p.dbscan_eps} min_samples={self.p.dbscan_min_samples}; waiting for the first sweep")

    # ------------------------------------------------------------ callbacks

    def on_wait_timer(self):
        if self.clouds == 0:
            self.get_logger().info("still waiting for the first sweep on /sensor/lidar/raw_points")

    def on_cloud(self, msg):
        t_cb = time.perf_counter_ns()
        self.clouds += 1
        ns = stamp_ns(msg.header.stamp)
        frame = self.drive.frame_of_stamp(ns)
        if frame is None:
            self.unknown_stamps += 1
            self.get_logger().warning(f"stamp {ns} is not in the drive's timestamps; sweep skipped")
            return
        if self.last_stamp is not None and ns < self.last_stamp:
            self.pass_no += 1  # the dataset looped
        self.last_stamp = ns

        points = point_cloud2.read_points_numpy(msg, field_names=["x", "y", "z"])
        t0 = time.perf_counter_ns()
        result = perception.detect(points, ns, self.p)
        detect_ms = (time.perf_counter_ns() - t0) / 1e6

        entry = {"pass": self.pass_no, "frame": frame, "det_rows": records.detection_rows(frame, ns, result.detections),
                 "frame_row": records.frames_row(frame, ns, result), "detect_ms": detect_ms,
                 "callback_ms": None, "status": result.ground_status, "stages": result.stages}
        self.log.append(entry)  # the frame counts as processed from here on

        header = msg.header
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, result.filtered_xyz))
        self.box_pub.publish(markers.box_markers(header, result.detections))
        entry["callback_ms"] = (time.perf_counter_ns() - t_cb) / 1e6

        processed = len(self.log)
        if processed % self.p.report_every_n == 0:
            self.write_outputs()
            m = self.metrics()
            self.get_logger().info(
                f"processed={processed} expected={m['frames_expected']} dropped={m['frames_dropped']} "
                f"detections={len(result.detections)} detect_p95={m['time_ms']['p95']} ms "
                f"callback_p95={m['callback_ms']['p95']} ms ground={m['ground_status_counts']} passes={m['passes']}")

    # ------------------------------------------------------------ outputs

    def metrics(self):
        log = list(self.log)  # a snapshot: the callback may append while this runs
        processed = len(log)
        span = {}
        status_counts, stage_sums = {}, dict.fromkeys(perception.STAGES, 0)
        for e in log:
            lo, hi = span.get(e["pass"], (e["frame"], e["frame"]))
            span[e["pass"]] = (min(lo, e["frame"]), max(hi, e["frame"]))
            status_counts[e["status"]] = status_counts.get(e["status"], 0) + 1
            for k in stage_sums:
                stage_sums[k] += e["stages"][k]
        expected = sum(hi - lo + 1 for lo, hi in span.values())
        n = max(processed, 1)
        return {
            "mode": "live", "seed": self.p.seed, "params": dataclasses.asdict(self.p), "platform": records.platform_info(),
            "frames_expected": expected, "frames_processed": processed, "frames_dropped": expected - processed,
            "first_frame_seen": log[0]["frame"] if log else None, "unknown_stamps": self.unknown_stamps,
            "clouds_received": self.clouds, "passes": len(span),
            "time_ms": records.time_stats([e["detect_ms"] for e in log], DETECT_SCOPE),
            "callback_ms": records.time_stats([e["callback_ms"] for e in log if e["callback_ms"] is not None], CALLBACK_SCOPE),
            "stage_means": {k: round(v / n, 1) for k, v in stage_sums.items()},
            "ground_status_counts": status_counts,
        }

    def write_outputs(self):
        log = list(self.log)
        det = ["pass," + records.DETECTION_HEADER] + [f"{e['pass']},{r}" for e in log for r in e["det_rows"]]
        frm = ["pass," + records.FRAMES_HEADER] + [f"{e['pass']},{e['frame_row']}" for e in log]
        records.atomic_write(f"{self.out}/detections.csv", "\n".join(det) + "\n")
        records.atomic_write(f"{self.out}/frames.csv", "\n".join(frm) + "\n")
        records.atomic_write(f"{self.out}/metrics.json", records.canonical_json(self.metrics()))


def main():
    rclpy.init(args=sys.argv)
    try:
        node = ObstacleTrackerNode()
    except (config.ConfigError, kitti.DataError) as exc:
        print(f"obstacle_tracker_node: {exc}", file=sys.stderr)
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):  # SIGINT from compose's stop_signal
        pass
    node.write_outputs()
    m = node.metrics()
    node.get_logger().info(
        f"FINAL processed={m['frames_processed']} expected={m['frames_expected']} dropped={m['frames_dropped']} "
        f"first_frame_seen={m['first_frame_seen']} unknown_stamps={m['unknown_stamps']} "
        f"detect_p95={m['time_ms']['p95']} ms callback_p95={m['callback_ms']['p95']} ms passes={m['passes']}")
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
