from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            parameters=[{"port": 8765, "address": "0.0.0.0"}],
        ),
        # Phase 0: replaced by obstacle_tracker_node in Phase 1.
        ExecuteProcess(cmd=["python3", "/app/processor/cloud_probe.py"], output="screen"),
    ])
