"""Pose math: OXTS -> world poses, frame transforms and the ego twist.

World frame (frozen at the end of Phase 1): ENU axes (x east, y north, z up),
origin at the first OXTS position. This is pykitti's convention: Mercator
projection of lat/lon with the scale taken from the first latitude, altitude
for z, and absolute roll/pitch/yaw with R = Rz(yaw) . Ry(pitch) . Rx(roll).

Naming: T_a_b maps a point from frame b into frame a (p_a = T_a_b . p_b).
The calibration file gives T_velo_imu, so T_world_velo = T_world_imu . inv(T_velo_imu).
"""
import numpy as np

EARTH_RADIUS_M = 6378137.0

# OXTS columns (oxts/dataformat.txt)
LAT, LON, ALT, ROLL, PITCH, YAW = range(6)
VN, VE, VU = 6, 7, 10
WX, WY, WZ = 17, 18, 19


def rotation_rpy(roll, pitch, yaw):
    """R = Rz(yaw) . Ry(pitch) . Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def T_world_imu(oxts):
    """(N, 4, 4) IMU poses in the ENU world frame; the first OXTS position is the origin."""
    oxts = np.asarray(oxts, dtype=np.float64)
    scale = np.cos(np.radians(oxts[0, LAT]))
    east = scale * EARTH_RADIUS_M * np.radians(oxts[:, LON])
    north = scale * EARTH_RADIUS_M * np.log(np.tan(np.radians(90.0 + oxts[:, LAT]) / 2.0))
    t = np.stack([east, north, oxts[:, ALT]], axis=1)
    t -= t[0]
    poses = np.tile(np.eye(4), (len(oxts), 1, 1))
    for i, row in enumerate(oxts):
        poses[i, :3, :3] = rotation_rpy(row[ROLL], row[PITCH], row[YAW])
    poses[:, :3, 3] = t
    return poses


def T_world_velo(T_world_imu_, T_velo_imu):
    """Velodyne poses in the world frame: T_world_imu . inv(T_velo_imu). Works on (4, 4) or (N, 4, 4)."""
    return np.asarray(T_world_imu_) @ np.linalg.inv(T_velo_imu)


def world_velo_poses(oxts, T_velo_imu):
    """(N, 4, 4) T_world_velo for every frame of a drive."""
    return T_world_velo(T_world_imu(oxts), T_velo_imu)


def transform_points(T, points):
    """Apply a 4x4 transform to (N, 3) points (or one (3,) point)."""
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def rot_to_quat_xyzw(R):
    """Unit quaternion (x, y, z, w) with w >= 0 for a rotation matrix (ROS order)."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def ego_twist_velo(oxts, T_world_velo_, T_velo_imu):
    """Linear (N, 3) and angular (N, 3) velocity of the velodyne frame, expressed in the velodyne frame.

    This is nav_msgs/Odometry's twist convention (child frame). The linear part is
    the IMU's ENU velocity (ve, vn, vu) plus the lever-arm term omega x r, where r
    is the IMU -> velodyne offset; the angular part is OXTS (wx, wy, wz) rotated
    from the IMU frame into the velodyne frame.
    """
    oxts = np.asarray(oxts, dtype=np.float64)
    T_wv = np.asarray(T_world_velo_).reshape(-1, 4, 4)
    R_world_imu = (T_wv @ T_velo_imu)[:, :3, :3]
    R_world_velo = T_wv[:, :3, :3]
    lever_imu = np.linalg.inv(T_velo_imu)[:3, 3]  # velodyne origin in the IMU frame
    v_imu_world = oxts[:, [VE, VN, VU]]
    w_world = np.einsum("nij,nj->ni", R_world_imu, oxts[:, [WX, WY, WZ]])
    r_world = R_world_imu @ lever_imu
    v_velo_world = v_imu_world + np.cross(w_world, r_world)
    linear = np.einsum("nji,nj->ni", R_world_velo, v_velo_world)  # R_world_velo^T . v
    angular = np.einsum("nji,nj->ni", R_world_velo, w_world)
    return linear, angular
