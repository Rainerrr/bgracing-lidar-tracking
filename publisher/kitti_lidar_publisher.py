#!/usr/bin/env python3
"""Replays a KITTI drive as a live LiDAR sensor (Phase 0: point clouds only)."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

import kitti_utils

FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


class KittiLidarPublisher(Node):
    def __init__(self):
        super().__init__("kitti_lidar_publisher")
        self.declare_parameter("data_dir", "/data")
        self.declare_parameter("drive", "2011_09_26_drive_0005")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)

        root = kitti_utils.drive_dir(self.get_parameter("data_dir").value, self.get_parameter("drive").value)
        velo = root / "velodyne_points"
        self.files = kitti_utils.list_frames(velo / "data")
        self.stamps_ns = kitti_utils.load_timestamps_ns(velo / "timestamps.txt")
        if len(self.stamps_ns) != len(self.files):
            raise RuntimeError(f"{len(self.files)} sweeps but {len(self.stamps_ns)} timestamps in {velo}")

        self.start = self.get_parameter("start_frame").value
        end = self.get_parameter("end_frame").value
        self.end = len(self.files) - 1 if end < 0 else min(end, len(self.files) - 1)
        self.loop = self.get_parameter("loop").value
        self.idx = self.start

        self.pub = self.create_publisher(PointCloud2, "/sensor/lidar/raw_points", qos_profile_sensor_data)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info(f"{len(self.files)} sweeps from {root}, frames {self.start}..{self.end}")

    def tick(self):
        if self.idx > self.end:
            if not self.loop:
                return
            self.idx = self.start
        pts = kitti_utils.load_bin(self.files[self.idx])
        stamp_ns = self.stamps_ns[self.idx]

        msg = point_cloud2.create_cloud(self._header(stamp_ns), FIELDS, pts)
        self.pub.publish(msg)
        self.idx += 1

    def _header(self, stamp_ns):
        header = Header()
        header.frame_id = "velodyne"
        header.stamp.sec = stamp_ns // 1_000_000_000
        header.stamp.nanosec = stamp_ns % 1_000_000_000
        return header


def main():
    rclpy.init()
    node = KittiLidarPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
