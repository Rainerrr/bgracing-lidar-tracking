#!/usr/bin/env python3
"""Live check of the publisher's stream (Phase 1). Replaced by obstacle_tracker_node in Phase 2.

For every received cloud it expects exactly one /sensor/ego/odom and one /tf
(world -> velodyne) with the same stamp, and it compares each odom pose and TF
with the pose computed by core from the same data. It logs a summary every
50 clouds and at shutdown; that log is the evidence for E1.3 and E1.5.
"""
import math
import sys
import time
from collections import defaultdict

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage

from core import kitti, poses

SETTLE_S = 1.0  # a cloud is judged once its odom and TF have had this long to arrive
POSE_TOL_M = 1e-3
ROT_TOL_RAD = 1e-6
SUMMARY_EVERY = 50


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def rotation_error_rad(q, q_ref):
    """Angle of the relative rotation between two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q_ref
    x2, y2, z2, w2 = q
    # conj(q_ref) * q
    w = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    v = (w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2,
         w1 * y2 - y1 * w2 - z1 * x2 + x1 * z2,
         w1 * z2 - z1 * w2 - x1 * y2 + y1 * x2)
    return 2.0 * math.atan2(math.sqrt(sum(c * c for c in v)), abs(w))


class StreamCounter:
    """Counts one topic's messages per (pass, stamp); a stamp going backward starts a new pass."""

    def __init__(self):
        self.last = None
        self.pass_no = 0
        self.counts = defaultdict(int)
        self.first_seen = {}

    def add(self, ns):
        if self.last is not None and ns < self.last:
            self.pass_no += 1
        self.last = ns
        key = (self.pass_no, ns)
        self.counts[key] += 1
        self.first_seen.setdefault(key, time.monotonic())
        return key


class CloudProbe(Node):
    def __init__(self):
        super().__init__("cloud_probe")
        self.declare_parameter("data_dir", "/data")
        self.declare_parameter("drive", "2011_09_26_drive_0005")
        self.drive = kitti.load_drive(self.get_parameter("data_dir").value, self.get_parameter("drive").value)
        self.T_wv = poses.world_velo_poses(self.drive.oxts, self.drive.T_velo_imu)
        self.quat = [poses.rot_to_quat_xyzw(T[:3, :3]) for T in self.T_wv]

        self.clouds, self.odoms, self.tfs = StreamCounter(), StreamCounter(), StreamCounter()
        self.n_clouds = 0
        self.judged = self.paired = self.odom_missing = self.tf_missing = self.duplicates = 0
        # Odom (reliable) whose cloud never arrived = a sweep the best-effort cloud topic dropped.
        self.dropped_clouds = self.orphan_tf = 0
        self.unknown_stamps = self.bad_frames = 0
        self.max_pos_err = self.max_rot_err = 0.0
        self.frame0_stamp = None
        self.wall_first = self.wall_last = None
        self.steps = 0
        self.stamp_span_ns = 0
        self.prev_cloud = None  # (pass, frame, stamp_ns)

        self.create_subscription(PointCloud2, "/sensor/lidar/raw_points", self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/sensor/ego/odom", self.on_odom,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(TFMessage, "/tf", self.on_tf,
                                 QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
        self.create_timer(0.5, self.judge_settled)
        self.get_logger().info(f"waiting for the first sweep ({self.drive.n_frames} frames known from {self.drive.root})")

    # ------------------------------------------------------------ callbacks

    def on_cloud(self, msg):
        ns = stamp_ns(msg.header.stamp)
        key = self.clouds.add(ns)
        now = time.monotonic()
        self.wall_first = self.wall_first or now
        self.wall_last = now
        self.n_clouds += 1
        if msg.header.frame_id != "velodyne":
            self.bad_frames += 1
        frame = self.drive.frame_of_stamp(ns)
        if frame is None:
            self.unknown_stamps += 1
        else:
            if frame == 0:
                self.frame0_stamp = ns
            if self.prev_cloud and self.prev_cloud[0] == key[0] and frame > self.prev_cloud[1]:
                self.steps += frame - self.prev_cloud[1]
                self.stamp_span_ns += ns - self.prev_cloud[2]
            self.prev_cloud = (key[0], frame, ns)
        if self.n_clouds == 1:
            self.get_logger().info(f"first sweep: frame {frame}, stamp {ns}, frame_id={msg.header.frame_id}")
        if self.n_clouds % SUMMARY_EVERY == 0:
            self.judge_settled()
            self.log_summary()

    def on_odom(self, msg):
        ns = stamp_ns(msg.header.stamp)
        self.odoms.add(ns)
        if msg.header.frame_id != "world" or msg.child_frame_id != "velodyne":
            self.bad_frames += 1
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.check_pose(ns, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    def on_tf(self, msg):
        for tf in msg.transforms:
            if tf.header.frame_id != "world" or tf.child_frame_id != "velodyne":
                continue
            ns = stamp_ns(tf.header.stamp)
            self.tfs.add(ns)
            t, q = tf.transform.translation, tf.transform.rotation
            self.check_pose(ns, (t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def check_pose(self, ns, position, quat):
        frame = self.drive.frame_of_stamp(ns)
        if frame is None:
            self.unknown_stamps += 1
            return
        err = float(np.linalg.norm(np.array(position) - self.T_wv[frame, :3, 3]))
        self.max_pos_err = max(self.max_pos_err, err)
        self.max_rot_err = max(self.max_rot_err, rotation_error_rad(quat, self.quat[frame]))

    # ------------------------------------------------------------ judging

    def judge_settled(self, everything=False):
        now = time.monotonic()
        for key, t0 in list(self.clouds.first_seen.items()):
            if not everything and now - t0 < SETTLE_S:
                continue
            n_odom, n_tf = self.odoms.counts.pop(key, 0), self.tfs.counts.pop(key, 0)
            self.odoms.first_seen.pop(key, None)
            self.tfs.first_seen.pop(key, None)
            self.clouds.first_seen.pop(key)
            self.clouds.counts.pop(key)
            self.judged += 1
            if n_odom == 1 and n_tf == 1:
                self.paired += 1
            self.odom_missing += n_odom == 0
            self.tf_missing += n_tf == 0
            self.duplicates += n_odom > 1 or n_tf > 1
        for stream, counter in ((self.odoms, "dropped_clouds"), (self.tfs, "orphan_tf")):
            for key, t0 in list(stream.first_seen.items()):  # stamps whose cloud never arrived
                if everything or now - t0 > 2 * SETTLE_S:
                    stream.first_seen.pop(key)
                    stream.counts.pop(key, None)
                    setattr(self, counter, getattr(self, counter) + 1)

    def log_summary(self, final=False):
        unmatched = self.judged - self.paired
        stamp_rate = self.steps / (self.stamp_span_ns / 1e9) if self.stamp_span_ns else float("nan")
        wall = (self.wall_last - self.wall_first) if self.wall_first and self.wall_last else 0.0
        wall_rate = (self.n_clouds - 1) / wall if wall > 0 else float("nan")
        ok = (unmatched == 0 and self.judged > 0 and self.unknown_stamps == 0 and self.bad_frames == 0
              and self.max_pos_err <= POSE_TOL_M and self.max_rot_err <= ROT_TOL_RAD)
        self.get_logger().info(
            f"{'FINAL ' if final else ''}summary: clouds={self.n_clouds} judged={self.judged} paired={self.paired} "
            f"unmatched={unmatched} (odom_missing={self.odom_missing} tf_missing={self.tf_missing} "
            f"duplicates={self.duplicates}) dropped_clouds={self.dropped_clouds} orphan_tf={self.orphan_tf} "
            f"unknown_stamps={self.unknown_stamps} "
            f"bad_frame_ids={self.bad_frames} pose_err_max={self.max_pos_err * 1e3:.4f} mm "
            f"rot_err_max={self.max_rot_err:.2e} rad stamp_rate={stamp_rate:.3f} Hz wall_rate={wall_rate:.3f} Hz "
            f"passes={self.clouds.pass_no + 1} frame0_stamp={self.frame0_stamp} "
            f"live_check={'OK' if ok else 'NOT OK'}")


def main():
    rclpy.init()
    try:
        node = CloudProbe()
    except kitti.DataError as exc:
        print(f"cloud_probe: {exc}", file=sys.stderr)
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):  # Ctrl-C or SIGTERM from docker compose down
        pass
    node.judge_settled(everything=True)
    node.log_summary(final=True)
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
