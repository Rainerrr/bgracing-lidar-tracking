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


# ---------------------------------------------------------------- tracks (world frame)

_STATE_RGB = {"moving": (1.0, 0.55, 0.0), "static": (0.55, 0.65, 0.8)}  # color is redundant: the label states it
ARROW_SECONDS = 0.5  # arrow length = speed x 0.5 s


def _cleared(header, pass_no):
    arr = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    clear.id = int(pass_no)  # carries the pass for the live checker; DELETEALL ignores ns and id
    arr.markers.append(clear)
    return arr


def track_markers(header, tracks, R_frame_velo, pass_no=0):
    """(boxes, arrows, labels) MarkerArrays for one frame's confirmed tracks; each starts with DELETEALL.

    header: the tracking frame (world, or velodyne with ego compensation off) and the cloud's stamp.
    A box is the matched detection's velodyne-axis box (l, w, h) at the track's filtered (x, y) and the
    detection's z, rotated into the tracking frame by R_frame_velo, so it is drawn as the sensor saw it.
    Arrows only for moving tracks. Labels: '#ID state N km/h'.
    """
    boxes, arrows, labels = _cleared(header, pass_no), _cleared(header, pass_no), _cleared(header, pass_no)
    R = [[float(v) for v in row] for row in R_frame_velo]
    for t in tracks:
        rgb = _STATE_RGB[t.state]
        box = _marker(header, "track_boxes", t.track_id, Marker.LINE_LIST)
        box.scale.x = 0.08
        box.color.r, box.color.g, box.color.b, box.color.a = (*rgb, 1.0)
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    o = (sx * t.l / 2, sy * t.w / 2, sz * t.h / 2)
                    corners.append((t.x + R[0][0] * o[0] + R[0][1] * o[1] + R[0][2] * o[2],
                                    t.y + R[1][0] * o[0] + R[1][1] * o[1] + R[1][2] * o[2],
                                    t.z + R[2][0] * o[0] + R[2][1] * o[1] + R[2][2] * o[2]))
        for a, b in _EDGES:
            for k in (a, b):
                box.points.append(Point(x=corners[k][0], y=corners[k][1], z=corners[k][2]))
        boxes.markers.append(box)

        if t.state == "moving":
            arrow = _marker(header, "track_arrows", t.track_id, Marker.ARROW)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.15, 0.35, 0.35
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (*rgb, 1.0)
            arrow.points.append(Point(x=t.x, y=t.y, z=t.z))
            arrow.points.append(Point(x=t.x + t.vx * ARROW_SECONDS, y=t.y + t.vy * ARROW_SECONDS, z=t.z))
            arrows.markers.append(arrow)

        label = _marker(header, "track_labels", t.track_id, Marker.TEXT_VIEW_FACING)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = t.x, t.y, t.z + t.h / 2 + 0.5
        label.scale.z = 0.5
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"#{t.track_id} {t.state} {t.speed * 3.6:.0f} km/h"
        labels.markers.append(label)
    return boxes, arrows, labels


# ---------------------------------------------------------------- ego car (velodyne frame)

# Approximate KITTI ego car (a VW Passat wagon) relative to the LiDAR. The sensor sits 1.73 m above the ground
# (the calibration-direction test pins this); the body outline is NOT measured: the cloud has no returns on
# the body, and the length, width and front/rear offset are rounded values for a car of this class.
# The marker says so in its label. It sits in the velodyne frame, so it follows the car by itself.
EGO_SIZE = (4.6, 1.8, 1.5)      # length, width, height (m)
EGO_CENTER = (-0.8, 0.0, -0.98)  # box center in velodyne axes: sensor height 1.73 m - half the body height - gap
EGO_RGB = (0.9, 0.9, 0.95)       # light grey; the text label says what it is, color carries no meaning


def ego_vehicle_markers(header):
    """MarkerArray with DELETEALL, a translucent body box, a sphere at the LiDAR origin, a heading arrow and a label."""
    arr = _cleared(header, 0)
    body = _marker(header, "ego", 0, Marker.CUBE)
    body.pose.position.x, body.pose.position.y, body.pose.position.z = EGO_CENTER
    body.scale.x, body.scale.y, body.scale.z = EGO_SIZE
    body.color.r, body.color.g, body.color.b, body.color.a = (*EGO_RGB, 0.35)
    arr.markers.append(body)

    sensor = _marker(header, "ego", 1, Marker.SPHERE)
    sensor.scale.x = sensor.scale.y = sensor.scale.z = 0.3
    sensor.color.r, sensor.color.g, sensor.color.b, sensor.color.a = 1.0, 0.2, 0.2, 1.0
    arr.markers.append(sensor)

    heading = _marker(header, "ego", 2, Marker.ARROW)
    heading.scale.x, heading.scale.y, heading.scale.z = 0.15, 0.35, 0.35
    heading.color.r, heading.color.g, heading.color.b, heading.color.a = 1.0, 0.2, 0.2, 1.0
    heading.points.append(Point(x=0.0, y=0.0, z=0.0))
    heading.points.append(Point(x=2.0, y=0.0, z=0.0))
    arr.markers.append(heading)

    label = _marker(header, "ego", 3, Marker.TEXT_VIEW_FACING)
    label.pose.position.x, label.pose.position.y = 0.0, 0.0
    label.pose.position.z = EGO_CENTER[2] + EGO_SIZE[2] / 2 + 0.6
    label.scale.z = 0.5
    label.color.r = label.color.g = label.color.b = label.color.a = 1.0
    label.text = "ego car (LiDAR = red dot, arrow = forward; body size approximate)"
    arr.markers.append(label)
    return arr
