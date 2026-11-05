#!/usr/bin/env python3
"""Phase 0 probe: logs every received sweep to prove cross-container delivery."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class CloudProbe(Node):
    def __init__(self):
        super().__init__("cloud_probe")
        self.n = 0
        self.create_subscription(PointCloud2, "/sensor/lidar/raw_points", self.on_cloud, qos_profile_sensor_data)

    def on_cloud(self, msg):
        self.n += 1
        if self.n % 10 == 1:
            s = msg.header.stamp
            self.get_logger().info(f"sweep {self.n}: {msg.width} points, frame_id={msg.header.frame_id}, stamp={s.sec}.{s.nanosec:09d}")


def main():
    rclpy.init()
    node = CloudProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
