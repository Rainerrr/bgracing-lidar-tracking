import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def tracker_cmd(env):
    """The tracker node's command line. Parameters come from the same YAML the offline runner reads; KITTI_DRIVE,
    when set, replaces its drive (compose sets it for both containers). The name is validated by the node."""
    cmd = ["python3", "/app/processor/obstacle_tracker_node.py", "--ros-args", "--params-file", "/config/tracker.yaml"]
    drive = env.get("KITTI_DRIVE")
    if drive:
        cmd += ["-p", f"drive:={drive}"]
    return cmd


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            parameters=[{"port": 8765, "address": "0.0.0.0"}],
        ),
        ExecuteProcess(cmd=tracker_cmd(os.environ), output="screen"),
    ])
