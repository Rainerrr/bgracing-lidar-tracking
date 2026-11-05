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
        # Phase 2: detection. Parameters come from the same YAML the offline runner reads.
        ExecuteProcess(cmd=["python3", "/app/processor/obstacle_tracker_node.py",
                            "--ros-args", "--params-file", "/config/tracker.yaml"], output="screen"),
    ])
