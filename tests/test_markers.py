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
