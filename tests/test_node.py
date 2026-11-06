"""Live node bookkeeping without a publisher (real drive; ROS in-process): odom pairing and stamp handling.

The node's callbacks are called directly, so the no-odom, odom-after-cloud, duplicate, unknown-stamp and loop
paths are exercised deterministically (a live run rarely hits them: no_odom was 0 over 446 frames).
"""
import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from core import kitti, poses
from processor.obstacle_tracker_node import ObstacleTrackerNode

from conftest import DATA_DIR, TRACKER_CONFIG


@pytest.fixture
def node(drive, tmp_path):
    rclpy.init(args=["--ros-args", "--params-file", str(TRACKER_CONFIG),
                     "-p", f"output_dir:={tmp_path}", "-p", f"data_dir:={DATA_DIR}"])
    n = ObstacleTrackerNode()
    n.T = poses.world_velo_poses(drive.oxts, drive.T_velo_imu)
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def header(ns, frame_id):
    h = Header()
    h.frame_id = frame_id
    h.stamp.sec, h.stamp.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
    return h


def cloud(node, f, ns=None):
    ns = int(node.drive.stamps_ns[f]) if ns is None else ns
    pts = kitti.load_sweep(node.drive.sweep_paths[f])[:, :3]
    return point_cloud2.create_cloud_xyz32(header(ns, "velodyne"), pts)


def odom(node, f):
    ns = int(node.drive.stamps_ns[f])
    m = Odometry()
    m.header = header(ns, "world")
    T = node.T[f]
    q = poses.rot_to_quat_xyzw(T[:3, :3])
    p = m.pose.pose.position
    p.x, p.y, p.z = (float(v) for v in T[:3, 3])
    o = m.pose.pose.orientation
    o.x, o.y, o.z, o.w = (float(v) for v in q)
    return m


def test_cloud_without_odom_is_skipped_and_counted_then_odom_after_it_still_pairs_later(node):
    node.on_cloud(cloud(node, 0))  # odom has not arrived (or arrives after its cloud)
    assert node.no_odom == 1 and node.log == [] and node.last_stamp is None
    node.on_odom(odom(node, 0))
    node.on_odom(odom(node, 1))
    node.on_cloud(cloud(node, 1))
    assert [(e["pass"], e["frame"]) for e in node.log] == [(0, 1)]
    T = node.odom[int(node.drive.stamps_ns[1])]
    assert np.array_equal(T, poses.carried_pose(node.T[1]))  # the bit-identical pose path


def test_duplicate_and_unknown_stamps_are_skipped_and_counted(node):
    node.on_odom(odom(node, 0))
    node.on_cloud(cloud(node, 0))
    node.on_cloud(cloud(node, 0))
    node.on_cloud(cloud(node, 1, ns=int(node.drive.stamps_ns[1]) + 1))
    assert node.duplicate_stamps == 1 and node.unknown_stamps == 1 and len(node.log) == 1


def test_a_backward_stamp_starts_a_new_pass_and_resets_the_tracker(node):
    for f in (5, 6, 7, 8):
        node.on_odom(odom(node, f))
        node.on_cloud(cloud(node, f))
    assert node.log[-1]["track_rows"], "tracks confirmed after 4 frames"
    node.on_odom(odom(node, 0))
    node.on_cloud(cloud(node, 0))
    assert node.pass_no == 1 and node.log[-1]["pass"] == 1 and node.log[-1]["track_rows"] == []
    m = node.metrics()
    assert m["passes"] == 2 and m["processed_frames"][-1] == [1, 0] and m["track_rows"] == sum(
        len(e["track_rows"]) for e in node.log)
