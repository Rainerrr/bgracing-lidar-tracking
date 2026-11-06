"""E1.1 pose math (synthetic) and the calibration-direction known-answer test (real calibration)."""
import numpy as np
import pytest

from core import kitti, poses

from conftest import oxts_row

ER = poses.EARTH_RADIUS_M
SYNTH_LEVER = np.array([0.8, -0.3, 0.8])  # velodyne origin in the IMU frame for conftest.CALIB_TEXT


TILTED = poses.rotation_rpy(np.pi / 2, 0.2, np.pi / 6)  # a calibration rotation far from identity


def synth_calib(R_velo_imu=np.eye(3)):
    """T_velo_imu that puts the velodyne origin at SYNTH_LEVER in the IMU frame.

    With the default identity rotation it matches conftest.CALIB_TEXT (T = (-0.8, 0.3, -0.8)).
    """
    T = np.eye(4)
    T[:3, :3] = R_velo_imu
    T[:3, 3] = -R_velo_imu @ SYNTH_LEVER
    return T


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------- T_world_imu (ENU)

def test_first_pose_is_origin_with_absolute_rotation():
    roll, pitch, yaw = 0.02, -0.01, 1.2
    T = poses.T_world_imu([oxts_row(roll=roll, pitch=pitch, yaw=yaw), oxts_row(lat=49.0001)])
    assert np.allclose(T[0, :3, 3], 0.0)
    assert np.allclose(T[0, :3, :3], poses.rotation_rpy(roll, pitch, yaw))


def test_yaw_is_heading_from_east_counter_clockwise():
    T = poses.T_world_imu([oxts_row(yaw=np.pi / 2)])
    assert np.allclose(T[0, :3, :3] @ [1, 0, 0], [0, 1, 0])  # vehicle forward points north (+y)


def test_known_offsets_give_expected_enu_translation():
    rows = [oxts_row(lat=49.0, lon=8.4, alt=110.0), oxts_row(lat=49.0001, lon=8.4001, alt=112.0)]
    t = poses.T_world_imu(rows)[1, :3, 3]
    assert t[0] == pytest.approx(np.cos(np.radians(49.0)) * ER * np.radians(1e-4), rel=1e-6)  # east
    assert t[1] == pytest.approx(ER * np.radians(1e-4), rel=1e-3)  # north: one 1e-4 deg step of latitude
    assert t[2] == pytest.approx(2.0)  # up


@pytest.mark.parametrize("R_velo_imu", [np.eye(3), TILTED], ids=["identity", "tilted"])
def test_world_fixed_point_survives_velo_round_trip_on_turning_poses(R_velo_imu):
    """Observe a world-fixed point from turning poses by an independent route, then map it back."""
    rows = [oxts_row(lat=49.0 + 1e-5 * i, lon=8.4 + 2e-5 * i, roll=0.01 * i, pitch=-0.005 * i, yaw=0.3 * i)
            for i in range(6)]
    T_vi = synth_calib(R_velo_imu)
    T_wi = poses.T_world_imu(rows)
    T_wv = poses.T_world_velo(T_wi, T_vi)
    p_world = np.array([12.0, -3.0, 0.5])
    for i in range(len(rows)):
        p_imu = T_wi[i, :3, :3].T @ (p_world - T_wi[i, :3, 3])
        p_velo = poses.transform_points(T_vi, p_imu)
        assert np.allclose(poses.transform_points(T_wv[i], p_velo), p_world, atol=1e-6)


# ---------------------------------------------------------------- quaternion

def test_quaternion_xyzw_for_known_yaw():
    q = poses.rot_to_quat_xyzw(poses.rotation_rpy(0, 0, np.pi / 2))
    assert np.allclose(q, [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])


def test_quaternion_matches_matrix_for_many_rotations():
    rng = np.random.default_rng(1)
    for roll, pitch, yaw in rng.uniform(-np.pi, np.pi, size=(200, 3)) * [1, 0.49, 1]:
        R = poses.rotation_rpy(roll, pitch, yaw)
        q = poses.rot_to_quat_xyzw(R)
        assert q[3] >= 0 and np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(quat_to_rot(q), R, atol=1e-9)


# ---------------------------------------------------------------- ego twist

@pytest.mark.parametrize("R_velo_imu", [np.eye(3), TILTED], ids=["identity", "tilted"])
def test_twist_pure_rotation_is_lever_arm(R_velo_imu):
    """Standing still and turning: the velodyne moves only through omega x r, seen in its own axes."""
    w_imu = np.array([0.05, -0.02, 0.5])
    rows = [oxts_row(wx=w_imu[0], wy=w_imu[1], wz=w_imu[2])]
    T_vi = synth_calib(R_velo_imu)
    linear, angular = poses.ego_twist_velo(rows, poses.world_velo_poses(rows, T_vi), T_vi)
    assert np.allclose(angular[0], R_velo_imu @ w_imu)
    assert np.allclose(linear[0], R_velo_imu @ np.cross(w_imu, SYNTH_LEVER))


@pytest.mark.parametrize("R_velo_imu", [np.eye(3), TILTED], ids=["identity", "tilted"])
def test_twist_driving_north_is_forward_in_velodyne_frame(R_velo_imu):
    rows = [oxts_row(yaw=np.pi / 2, vn=3.0)]
    T_vi = synth_calib(R_velo_imu)
    linear, angular = poses.ego_twist_velo(rows, poses.world_velo_poses(rows, T_vi), T_vi)
    assert np.allclose(linear[0], R_velo_imu @ [3.0, 0, 0])  # IMU forward, in velodyne axes
    assert np.allclose(angular[0], 0)


# ---------------------------------------------------------------- calibration direction (real data)

KNOWN_LEVER_ARM = np.array([0.81, -0.31, 0.80])  # Velodyne 0.81 m ahead of and 0.80 m above the IMU


def velodyne_offset_in_imu(oxts, T_velo_imu, frame):
    """Where the pipeline puts the velodyne relative to the IMU, expressed in the IMU frame."""
    T_wi = poses.T_world_imu(oxts)[frame]
    T_wv = poses.world_velo_poses(oxts, T_velo_imu)[frame]
    return T_wi[:3, :3].T @ (T_wv[:3, 3] - T_wi[:3, 3])


def assert_known_lever_arm(oxts, T_velo_imu):
    for frame in (0, len(oxts) // 2, len(oxts) - 1):
        offset = velodyne_offset_in_imu(oxts, T_velo_imu, frame)
        assert np.allclose(offset, KNOWN_LEVER_ARM, atol=0.01), (
            f"frame {frame}: velodyne at {offset.round(3)} in the IMU frame, expected {KNOWN_LEVER_ARM} "
            "(KITTI mounts the Velodyne at 1.73 m and the IMU at 0.93 m)")


def test_calibration_direction_matches_kitti_mounting(drive):
    assert_known_lever_arm(drive.oxts, drive.T_velo_imu)


@pytest.mark.parametrize("mutation", ["inverted", "omitted"])
def test_calibration_direction_check_catches_mistakes(drive, mutation):
    """The parked-object check can't see these mistakes (1.34 m / 1.13 m); this test must."""
    wrong = np.linalg.inv(drive.T_velo_imu) if mutation == "inverted" else np.eye(4)
    with pytest.raises(AssertionError):
        assert_known_lever_arm(drive.oxts, wrong)


def test_real_ego_speed_matches_oxts(drive):
    """The twist magnitude stays within the lever-arm term of the OXTS ground speed."""
    T_wv = poses.world_velo_poses(drive.oxts, drive.T_velo_imu)
    linear, _ = poses.ego_twist_velo(drive.oxts, T_wv, drive.T_velo_imu)
    oxts_speed = np.hypot(drive.oxts[:, poses.VN], drive.oxts[:, poses.VE])
    assert np.all(np.abs(np.linalg.norm(linear[:, :2], axis=1) - oxts_speed) < 0.3)
    assert 3.0 < oxts_speed.min() and oxts_speed.max() < 6.6


# ---------------------------------------------------------------- Phase 3: carried pose path

def test_quat_to_rot_inverts_rot_to_quat_on_every_branch():
    for R in (np.eye(3), TILTED, poses.rotation_rpy(0.0, 0.0, np.pi), poses.rotation_rpy(np.pi, 0.1, 0.0),
              poses.rotation_rpy(0.0, np.pi, 0.3), poses.rotation_rpy(0.3, -0.2, 2.9)):
        assert np.allclose(poses.quat_to_rot(poses.rot_to_quat_xyzw(R)), R, atol=1e-12)


def test_quat_to_rot_rejects_a_zero_quaternion():
    with pytest.raises(ValueError):
        poses.quat_to_rot([0.0, 0.0, 0.0, 0.0])


def test_carried_pose_is_bit_identical_to_a_float64_message_round_trip(drive):
    """The offline path (carried_pose) and the live path (odom fields -> pose_matrix) give the same bits."""
    T_wv = poses.world_velo_poses(drive.oxts, drive.T_velo_imu)
    for T in T_wv:
        q = poses.rot_to_quat_xyzw(T[:3, :3])
        msg = {"p": [float(v) for v in T[:3, 3]], "q": [float(v) for v in q]}  # what Odometry carries (float64)
        live = poses.pose_matrix(msg["p"], msg["q"])
        assert np.array_equal(live, poses.carried_pose(T))
        # KITTI's calibration rotation is orthonormal only to ~5e-8, which a quaternion can't carry:
        # measured max difference 5e-8 (2 um at 40 m). Live and offline both use the carried pose.
        assert np.allclose(live, T, atol=1e-6)
