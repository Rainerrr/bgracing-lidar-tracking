"""Foxglove/RViz markers for detections. ROS.

Every array starts with a DELETEALL marker, so boxes of earlier frames never linger and an empty frame
clears the view. Each detection adds one LINE_LIST wireframe and one TEXT_VIEW_FACING label that carries
the box extents and point count as text. All boxes share one color, so color never encodes status.
"""
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def _marker(header, ns, mid, mtype):
    m = Marker()
    m.header = header
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.lifetime.sec = m.lifetime.nanosec = 0
    m.frame_locked = False
    m.pose.orientation.w = 1.0
    return m


def box_markers(header, detections):
    """MarkerArray for one frame: DELETEALL, then (LINE_LIST box + TEXT label) per detection."""
    arr = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    for i, d in enumerate(detections):
        box = _marker(header, "boxes", i, Marker.LINE_LIST)
        box.scale.x = 0.06
        box.color.r, box.color.g, box.color.b, box.color.a = 0.1, 0.85, 0.2, 1.0
        corners = [(d.x + sx * d.dx / 2, d.y + sy * d.dy / 2, d.z + sz * d.dz / 2)
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for a, b in _EDGES:
            for k in (a, b):
                box.points.append(Point(x=corners[k][0], y=corners[k][1], z=corners[k][2]))
        arr.markers.append(box)

        label = _marker(header, "labels", i, Marker.TEXT_VIEW_FACING)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = d.x, d.y, d.z + d.dz / 2 + 0.4
        label.scale.z = 0.45
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"{d.dx:.1f}x{d.dy:.1f}x{d.dz:.1f} m, {d.n_points} pts"
        arr.markers.append(label)
    return arr
