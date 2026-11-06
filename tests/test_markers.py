"""E2.5 node-side marker construction: DELETEALL first, one box and one text label per detection."""
from types import SimpleNamespace as NS

from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from processor import markers


def header():
    h = Header()
    h.frame_id = "velodyne"
    h.stamp.sec, h.stamp.nanosec = 1317042272, 335337762
    return h


def det(x=10.0):
    return NS(x=x, y=2.0, z=-0.9, dx=4.5, dy=1.8, dz=1.4, n_points=120)


def test_empty_frame_is_deleteall_only():
    arr = markers.box_markers(header(), [])
    assert len(arr.markers) == 1 and arr.markers[0].action == Marker.DELETEALL


def test_one_box_and_one_label_per_detection():
    arr = markers.box_markers(header(), [det(10), det(20)])
    assert arr.markers[0].action == Marker.DELETEALL and len(arr.markers) == 5
    kinds = [m.type for m in arr.markers[1:]]
    assert kinds.count(Marker.LINE_LIST) == 2 and kinds.count(Marker.TEXT_VIEW_FACING) == 2
    box = next(m for m in arr.markers if m.type == Marker.LINE_LIST)
    assert len(box.points) == 24 and box.header.frame_id == "velodyne" and box.header.stamp.sec == 1317042272
    label = next(m for m in arr.markers if m.type == Marker.TEXT_VIEW_FACING)
    assert "4.5x1.8x1.4 m" in label.text and "120 pts" in label.text, "status is carried by text, not by color"
    assert len({(m.ns, m.id) for m in arr.markers[1:]}) == 4


# ---------------------------------------------------------------- Phase 3: track markers

import math  # noqa: E402

from processor.tracking import TrackState  # noqa: E402


def track(tid, state, x=10.0, vx=5.0):
    speed = abs(vx)
    return TrackState(tid, x, 2.0, vx, 0.0, speed, state, 4.0, 2.0, 1.5, -0.9, 3)


def world_header():
    h = header()
    h.frame_id = "world"
    return h


def test_track_markers_start_with_deleteall_and_carry_the_pass():
    for arr in markers.track_markers(world_header(), [], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], pass_no=3):
        assert len(arr.markers) == 1 and arr.markers[0].action == Marker.DELETEALL and arr.markers[0].id == 3


def test_one_box_and_label_per_track_arrows_only_for_moving():
    boxes, arrows, labels = markers.track_markers(
        world_header(), [track(1, "moving"), track(2, "static", x=20.0, vx=0.1)], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert [m.type for m in boxes.markers[1:]] == [Marker.LINE_LIST] * 2
    assert [m.type for m in arrows.markers[1:]] == [Marker.ARROW]
    assert [m.text for m in labels.markers[1:]] == ["#1 moving 18 km/h", "#2 static 0 km/h"]
    assert all(m.header.frame_id == "world" for arr in (boxes, arrows, labels) for m in arr.markers)
    a = arrows.markers[1]
    assert a.points[1].x - a.points[0].x == 2.5  # 5 m/s x 0.5 s


def test_box_is_rotated_into_the_tracking_frame():
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    boxes, _, _ = markers.track_markers(world_header(), [track(1, "static", vx=0.0)], [[c, -s, 0], [s, c, 0], [0, 0, 1]])
    xs = [p.x for p in boxes.markers[1].points]
    ys = [p.y for p in boxes.markers[1].points]
    # l = 4 along velodyne x becomes 4 along world y after a 90 degree yaw
    assert abs((max(xs) - min(xs)) - 2.0) < 1e-9 and abs((max(ys) - min(ys)) - 4.0) < 1e-9
