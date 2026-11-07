#!/usr/bin/env python3
"""Live tracking node: sweep and ego pose in, tracked obstacles out.

Subscribes to /sensor/lidar/raw_points (best effort, newest only: a slow frame never builds a backlog) and
/sensor/ego/odom (reliable). Each cloud is paired with the odom message of the same stamp; a cloud without
one is skipped and counted (no_odom), never posed some other way. Per cloud it runs the same code as
scripts/run_offline.py: perception.detect, tracking.to_world, Tracker.step.

Publishes the filtered cloud (velodyne) and, in the tracking frame (world; velodyne with ego compensation off),
/perception/bounding_boxes, /perception/obstacle_vectors and /perception/track_labels, each starting with
DELETEALL. Writes detections.csv, frames.csv, tracks.csv and metrics.json every report_every_n processed frames
and on the stop signal; obstacle_tracking_report.png on the stop signal only (a render takes hundreds of ms).

Time: a backward stamp is the dataset loop (pass + 1, tracker reset); a duplicate stamp is skipped and counted;
a forward gap is handled by the tracker as misses, so a stall never resets it. Wall time is used only for timing.
"""
import dataclasses
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import MarkerArray

from core import kitti, poses
from processor import config, markers, perception, records, report, tracking  # report imports matplotlib now

DETECT_SCOPE = "detect() = pipeline steps 2-7 (ROI to detections)"
TRACK_SCOPE = "to_world() + Tracker.step() = pipeline steps 7 (world transform) to 11"
CALLBACK_SCOPE = "whole cloud callback: decode, detect, track, publish, rows, and the periodic file write"
WRITE_SCOPE = "periodic write_outputs(): every CSV rewritten from the log, plus metrics.json"
IGNORED_PARAMS = {"use_sim_time", "start_type_description_service"}  # declared by rclpy itself
ODOM_CACHE = 64  # stamps; odom normally arrives just before its cloud


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


def clean_passes(processed, n_frames):
    """Passes whose processed frames are exactly 0..n_frames-1."""
    by_pass = {}
    for p, f in processed:
        by_pass.setdefault(p, []).append(f)
    return sorted(p for p, fs in by_pass.items() if fs == list(range(n_frames)))


class ObstacleTrackerNode(Node):
    def __init__(self):
        super().__init__("obstacle_tracker_node", automatically_declare_parameters_from_overrides=True)
        self.p = params_from_node(self)
        self.drive = kitti.load_drive(self.p.data_dir, self.p.drive)
        self.out = self.p.output_dir
        self.track_frame = "world" if self.p.ego_compensation else "velodyne"
        self.tracker = tracking.Tracker(self.p)

        # One record per processed frame, appended in a single step; every counter, file and statistic is
        # derived from this log. SIGINT can interrupt the callback at any bytecode, so the bookkeeping must
        # not be spread over several updates (that once left metrics.json one frame ahead of the CSVs).
        self.log = []
        self.pass_no, self.last_stamp = 0, None
        self.clouds = self.unknown_stamps = self.no_odom = self.duplicate_stamps = 0
        self.odom = OrderedDict()  # stamp_ns -> T_world_velo, bounded; stamps repeat with equal poses each pass
        self.write_ms = []
        self._last_no_odom_log = 0.0

        self.cloud_pub = self.create_publisher(PointCloud2, "/perception/filtered_points", qos_profile_sensor_data)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.box_pub = self.create_publisher(MarkerArray, "/perception/bounding_boxes", reliable)
        self.arrow_pub = self.create_publisher(MarkerArray, "/perception/obstacle_vectors", reliable)
        self.label_pub = self.create_publisher(MarkerArray, "/perception/track_labels", reliable)
        self.ego_pub = self.create_publisher(MarkerArray, "/perception/ego_vehicle", reliable)
        self.create_subscription(Odometry, "/sensor/ego/odom", self.on_odom, reliable)
        self.create_subscription(PointCloud2, "/sensor/lidar/raw_points", self.on_cloud,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(10.0, self.on_wait_timer)
        self.write_outputs()  # a zero-frame metrics.json exists from the start
        self.get_logger().info(
            f"tracking ready: {self.drive.n_frames} frames known, frame={self.track_frame}, "
            f"association={self.p.association}, gate={self.p.gate_m} m, q_accel={self.p.q_accel}, r_pos={self.p.r_pos}, "
            f"moving if speed - {self.p.moving_sigma} sigma > {self.p.moving_speed} m/s; waiting for the first sweep")

    # ------------------------------------------------------------ callbacks

    def on_wait_timer(self):
        if self.clouds == 0:
            self.get_logger().info("still waiting for the first sweep on /sensor/lidar/raw_points")

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odom[stamp_ns(msg.header.stamp)] = poses.pose_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        while len(self.odom) > ODOM_CACHE:
            self.odom.popitem(last=False)

    def on_cloud(self, msg):
        t_cb = time.perf_counter_ns()
        self.clouds += 1
        ns = stamp_ns(msg.header.stamp)
        frame = self.drive.frame_of_stamp(ns)
        if frame is None:
            self.unknown_stamps += 1
            self.get_logger().warning(f"stamp {ns} is not in the drive's timestamps; sweep skipped")
            return
        if self.last_stamp is not None and ns == self.last_stamp:
            self.duplicate_stamps += 1
            return
        T = self.odom.get(ns) if self.p.ego_compensation else np.eye(4)
        if T is None:
            self.no_odom += 1
            now = time.monotonic()
            if now - self._last_no_odom_log > 1.0:
                self._last_no_odom_log = now
                self.get_logger().warning(f"no odom for stamp {ns} (frame {frame}); sweep skipped ({self.no_odom} so far)")
            return
        if self.last_stamp is not None and ns < self.last_stamp:
            self.pass_no += 1  # the dataset looped
            self.tracker.reset()
        self.last_stamp = ns

        points = point_cloud2.read_points_numpy(msg, field_names=["x", "y", "z"])
        t0 = time.perf_counter_ns()
        result = perception.detect(points, ns, self.p)
        t1 = time.perf_counter_ns()
        tracks = self.tracker.step(frame, ns, tracking.to_world(result.detections, T))
        t2 = time.perf_counter_ns()

        entry = {"pass": self.pass_no, "frame": frame, "det_rows": records.detection_rows(frame, ns, result.detections),
                 "frame_row": records.frames_row(frame, ns, result),
                 "track_rows": records.track_rows(self.pass_no, frame, ns, tracks),
                 "detect_ms": (t1 - t0) / 1e6, "track_ms": (t2 - t1) / 1e6, "callback_ms": None,
                 "status": result.ground_status, "stages": result.stages, "n_tracks": len(tracks),
                 "n_moving": sum(t.state == "moving" for t in tracks)}
        self.log.append(entry)  # the frame counts as processed from here on

        header = msg.header
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, result.filtered_xyz))
        th = type(header)()
        th.stamp, th.frame_id = header.stamp, self.track_frame
        boxes, arrows, labels = markers.track_markers(th, tracks, T[:3, :3], self.pass_no)
        self.box_pub.publish(boxes)
        self.arrow_pub.publish(arrows)
        self.label_pub.publish(labels)
        self.ego_pub.publish(markers.ego_vehicle_markers(header))  # velodyne frame: follows the car by itself

        processed = len(self.log)
        if processed % self.p.report_every_n == 0:
            self.write_outputs()
            m = self.metrics()
            self.get_logger().info(
                f"processed={processed} expected={m['frames_expected']} dropped={m['frames_dropped']} "
                f"tracks={len(tracks)} moving={entry['n_moving']} detect_p95={m['time_ms']['p95']} ms "
                f"track_p95={m['tracker_ms']['p95']} ms callback_p95={m['callback_ms']['p95']} ms "
                f"no_odom={self.no_odom} passes={m['passes']}")
        entry["callback_ms"] = (time.perf_counter_ns() - t_cb) / 1e6

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
        frames = [[e["pass"], e["frame"]] for e in log]
        return {
            "mode": "live", "seed": self.p.seed, "params": dataclasses.asdict(self.p), "platform": records.platform_info(),
            "frames_expected": expected, "frames_processed": processed, "frames_dropped": expected - processed,
            "first_frame_seen": log[0]["frame"] if log else None, "unknown_stamps": self.unknown_stamps,
            "no_odom": self.no_odom, "duplicate_stamps": self.duplicate_stamps,
            "clouds_received": self.clouds, "passes": len(span),
            "clean_passes": clean_passes(frames, self.drive.n_frames),
            "processed_frames": frames,
            "time_ms": records.time_stats([e["detect_ms"] for e in log], DETECT_SCOPE),
            "tracker_ms": records.time_stats([e["track_ms"] for e in log], TRACK_SCOPE),
            "callback_ms": records.time_stats([e["callback_ms"] for e in log if e["callback_ms"] is not None], CALLBACK_SCOPE),
            "write_ms": records.time_stats(list(self.write_ms), WRITE_SCOPE),
            "stage_means": {k: round(v / n, 1) for k, v in stage_sums.items()},
            "ground_status_counts": status_counts,
            "track_rows": sum(len(e["track_rows"]) for e in log),
            "tracks": {"confirmed": self.tracker.confirmed_total, "deleted": self.tracker.deleted_total,
                       "confirmed_per_frame_mean": round(sum(e["n_tracks"] for e in log) / n, 2),
                       "moving_per_frame_mean": round(sum(e["n_moving"] for e in log) / n, 2)},
        }

    def write_outputs(self):
        t0 = time.perf_counter_ns()
        log = list(self.log)
        det = ["pass," + records.DETECTION_HEADER] + [f"{e['pass']},{r}" for e in log for r in e["det_rows"]]
        frm = ["pass," + records.FRAMES_HEADER] + [f"{e['pass']},{e['frame_row']}" for e in log]
        trk = [records.TRACK_HEADER] + [r for e in log for r in e["track_rows"]]
        records.atomic_write(f"{self.out}/detections.csv", "\n".join(det) + "\n")
        records.atomic_write(f"{self.out}/frames.csv", "\n".join(frm) + "\n")
        records.atomic_write(f"{self.out}/tracks.csv", "\n".join(trk) + "\n")
        self.write_ms.append((time.perf_counter_ns() - t0) / 1e6)
        records.atomic_write(f"{self.out}/metrics.json", records.canonical_json(self.metrics()))

    def write_report(self):
        """The PNG of the last pass (stop signal only). Ego path from the drive's poses."""
        rows = [r for e in list(self.log) for r in e["track_rows"]]
        ego = (poses.world_velo_poses(self.drive.oxts, self.drive.T_velo_imu)[:, :2, 3]
               if self.p.ego_compensation else np.zeros((1, 2)))
        tmp = f"{self.out}/obstacle_tracking_report.tmp.png"
        report.render_report(tmp, rows, ego, self.track_frame, " (live)")
        os.replace(tmp, f"{self.out}/obstacle_tracking_report.png")


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
    except (KeyboardInterrupt, ExternalShutdownException):  # the stop signal (compose sends SIGINT)
        pass
    node.write_outputs()
    try:
        node.write_report()
    except Exception as exc:  # the CSV and JSON files are already complete; say what failed
        node.get_logger().error(f"report PNG not written: {exc!r}")
    m = node.metrics()
    node.get_logger().info(
        f"FINAL processed={m['frames_processed']} expected={m['frames_expected']} dropped={m['frames_dropped']} "
        f"first_frame_seen={m['first_frame_seen']} unknown_stamps={m['unknown_stamps']} no_odom={m['no_odom']} "
        f"duplicates={m['duplicate_stamps']} passes={m['passes']} clean_passes={m['clean_passes']} "
        f"tracks={m['tracks']['confirmed']} detect_p95={m['time_ms']['p95']} ms track_p95={m['tracker_ms']['p95']} ms "
        f"callback_p95={m['callback_ms']['p95']} ms write_p95={m['write_ms']['p95']} ms")
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
