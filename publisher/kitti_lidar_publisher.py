#!/usr/bin/env python3
"""Replays a KITTI drive as a live LiDAR sensor.

Each tick publishes, with the sweep's dataset timestamp: the world -> velodyne TF,
the ego odometry (pose of the velodyne frame in world, twist in velodyne), then
the point cloud. The world frame is ENU with the first OXTS position as origin
(see core/poses.py).
"""
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from core import kitti, poses

FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]
WORLD, VELODYNE = "world", "velodyne"


class KittiLidarPublisher(Node):
    def __init__(self):
        super().__init__("kitti_lidar_publisher")
        self.declare_parameter("data_dir", "/data")
        self.declare_parameter("drive", "2011_09_26_drive_0005")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)

        # Validate the whole drive and precompute every pose before publishing anything.
        self.drive = kitti.load_drive(self.get_parameter("data_dir").value, self.get_parameter("drive").value)
        n = self.drive.n_frames
        self.T_wv = poses.world_velo_poses(self.drive.oxts, self.drive.T_velo_imu)
        self.quat = [poses.rot_to_quat_xyzw(T[:3, :3]) for T in self.T_wv]
        self.lin, self.ang = poses.ego_twist_velo(self.drive.oxts, self.T_wv, self.drive.T_velo_imu)

        self.start = self.get_parameter("start_frame").value
        end = self.get_parameter("end_frame").value
        self.end = n - 1 if end < 0 else min(end, n - 1)
        if not 0 <= self.start <= self.end:
            raise ValueError(f"start_frame {self.start} must be in 0..{self.end} (drive has {n} sweeps)")
        self.loop = self.get_parameter("loop").value
        self.idx = self.start

        self.tf = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(
            Odometry, "/sensor/ego/odom", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.cloud_pub = self.create_publisher(PointCloud2, "/sensor/lidar/raw_points", qos_profile_sensor_data)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info(
            f"{n} sweeps from {self.drive.root}, frames {self.start}..{self.end}, poses ready "
            f"(world = ENU, origin at OXTS frame 0)")

    def tick(self):
        if self.idx > self.end:
            if not self.loop:
                return
            self.idx = self.start
        i = self.idx
        stamp_ns = int(self.drive.stamps_ns[i])
        pts = kitti.load_sweep(self.drive.sweep_paths[i])

        self.tf.sendTransform(self._transform(i, stamp_ns))
        self.odom_pub.publish(self._odom(i, stamp_ns))
        self.cloud_pub.publish(point_cloud2.create_cloud(self._header(stamp_ns, VELODYNE), FIELDS, pts))
        self.idx += 1

    def _header(self, stamp_ns, frame_id):
        header = Header()
        header.frame_id = frame_id
        header.stamp.sec = stamp_ns // 1_000_000_000
        header.stamp.nanosec = stamp_ns % 1_000_000_000
        return header

    def _transform(self, i, stamp_ns):
        msg = TransformStamped()
        msg.header = self._header(stamp_ns, WORLD)
        msg.child_frame_id = VELODYNE
        t = msg.transform.translation
        t.x, t.y, t.z = (float(v) for v in self.T_wv[i, :3, 3])
        q = msg.transform.rotation
        q.x, q.y, q.z, q.w = (float(v) for v in self.quat[i])
        return msg

    def _odom(self, i, stamp_ns):
        msg = Odometry()
        msg.header = self._header(stamp_ns, WORLD)
        msg.child_frame_id = VELODYNE
        p = msg.pose.pose.position
        p.x, p.y, p.z = (float(v) for v in self.T_wv[i, :3, 3])
        q = msg.pose.pose.orientation
        q.x, q.y, q.z, q.w = (float(v) for v in self.quat[i])
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        lin.x, lin.y, lin.z = (float(v) for v in self.lin[i])
        ang.x, ang.y, ang.z = (float(v) for v in self.ang[i])
        return msg


def main():
    rclpy.init()
    try:
        node = KittiLidarPublisher()
    except (kitti.DataError, ValueError) as exc:
        print(f"kitti_lidar_publisher: {exc}", file=sys.stderr)
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):  # Ctrl-C or SIGTERM from docker compose down
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
