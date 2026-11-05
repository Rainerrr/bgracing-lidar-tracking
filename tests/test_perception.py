"""E2.1: perception core on synthetic clouds (no data, no ROS), each check with a negative control."""
import dataclasses

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from processor import perception as pc

from conftest import box_surface, plane_patch, wall

GROUND = plane_patch(5, 35, -15, 15)  # flat ground at z = -1.7
STAMP = 1317042272335337762


def scene(*objects, ground=GROUND):
    return np.vstack([ground, *objects]) if ground is not None else np.vstack(objects)


# ---------------------------------------------------------------- one car on flat ground

def test_flat_ground_and_one_car_gives_one_detection(params):
    car = box_surface(15.0, 3.0, -1.7, 4.5, 1.8, 1.5)
    res = pc.detect(scene(car), STAMP, params)
    assert res.ground_status == pc.GROUND_OK and len(res.detections) == 1
    d = res.detections[0]
    # ground removal takes the car's two lowest voxel layers (within 0.2 m of the plane): true center z = -0.95
    truth = np.array([15.0, 3.0, -0.95])
    got = np.array([d.x, d.y, d.z])
    assert np.all(np.abs(got - truth) <= 0.15), got
    assert abs(d.dx - 4.5) <= 0.3 and abs(d.dy - 1.8) <= 0.3 and abs(d.dz - 1.35) <= 0.3
    # negative control: the same assertion with a 0.5 m offset must fail
    assert not np.all(np.abs(got - (truth + (0.5, 0.0, 0.0))) <= 0.15)
    assert res.stages["raw"] >= res.stages["roi"] >= res.stages["voxel"] > res.stages["non_ground"] >= res.stages["clustered"] > 0


def test_stage_counts_are_consistent(params):
    res = pc.detect(scene(box_surface(15, 0, -1.7, 4.5, 1.8, 1.5)), STAMP, params)
    assert res.clusters_found >= res.clusters_kept == len(res.detections)
    assert res.stages["clustered"] == sum(d.n_points for d in res.detections)
    assert len(res.filtered_xyz) == res.stages["non_ground"]


# ---------------------------------------------------------------- ground tilt

def tilted(deg):
    return plane_patch(10, 20, -3, 3, z0=0.0, tilt_deg=deg)


@pytest.mark.parametrize("deg", [20, 24])
def test_tilt_within_limit_is_removed_as_ground(params, deg):
    res = pc.detect(tilted(deg), STAMP, params)
    assert res.ground_status == pc.GROUND_OK
    assert res.stages["non_ground"] <= 0.02 * res.stages["voxel"]


def test_tilt_beyond_limit_is_not_accepted(params, deg=30):
    res = pc.detect(tilted(deg), STAMP, params)
    assert res.ground_status == pc.NO_PLANE
    assert res.stages["non_ground"] == res.stages["voxel"]


def test_tilt_control_a_loose_accept_rule_fails_the_30_degree_case(params):
    loose = dataclasses.replace(params, ransac_min_normal_z=0.5)
    res = pc.detect(tilted(30), STAMP, loose)
    assert res.ground_status == pc.GROUND_OK and res.stages["non_ground"] < res.stages["voxel"]


@pytest.mark.parametrize("stamp", [STAMP + k * 103_000_000 for k in range(1, 21)])
def test_tilt_decision_does_not_depend_on_the_stamp(params, stamp):
    assert pc.detect(tilted(24), stamp, params).ground_status == pc.GROUND_OK
    assert pc.detect(tilted(30), stamp, params).ground_status == pc.NO_PLANE


# ---------------------------------------------------------------- degenerate input and failure paths

@pytest.mark.parametrize("pts", [np.empty((0, 4)), np.empty((0, 3)), np.array([[15.0, 0.0, 0.0]]),
                                 np.array([[15.0, 0.0, 0.0], [20.0, 1.0, 0.5]])])
def test_fewer_than_three_points_gives_no_detections(params, pts):
    res = pc.detect(pts, STAMP, params)
    assert res.detections == () and res.ground_status == pc.TOO_FEW and len(res.filtered_xyz) == 0


def test_cloud_entirely_outside_the_roi(params):
    res = pc.detect(plane_patch(60, 70, -5, 5), STAMP, params)
    assert res.stages["raw"] > 0 and res.stages["roi"] == 0
    assert res.detections == () and res.ground_status == pc.TOO_FEW


def test_ego_box_is_removed(params):
    res = pc.detect(box_surface(0.0, 0.0, -1.7, 4.0, 1.8, 1.5), STAMP, params)
    assert res.stages["roi"] < res.stages["raw"]
    assert res.detections == ()


def test_fewer_than_min_samples_points_gives_no_detections(params):
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(10, 30, 7), rng.uniform(-10, 10, 7), rng.uniform(-1, 1, 7)])
    res = pc.detect(pts, STAMP, params)
    assert res.detections == () and res.ground_status != pc.TOO_FEW


def test_no_plane_removes_nothing_and_still_clusters(params):
    face = wall(15, 0, 18, 0, height=2.5)  # vertical: no candidate plane has |n_z| > 0.9
    res = pc.detect(face, STAMP, params)
    assert res.ground_status == pc.NO_PLANE
    assert res.stages["non_ground"] == res.stages["voxel"]
    assert len(res.detections) == 1


def test_input_with_extra_columns_and_float32_is_accepted(params):
    car = box_surface(15, 3, -1.7, 4.5, 1.8, 1.5)
    xyzi = np.column_stack([scene(car), np.full(len(scene(car)), 0.5)]).astype(np.float32)
    assert len(pc.detect(xyzi, STAMP, params).detections) == 1


# ---------------------------------------------------------------- walls, cars and the oriented size filter

def test_long_wall_is_filtered_out(params):
    res = pc.detect(scene(wall(20, -10, 20, 10)), STAMP, params)
    assert res.clusters_found >= 1 and res.detections == ()


def test_diagonal_long_wall_is_filtered_out(params):
    res = pc.detect(scene(wall(10, -10, 24, 4)), STAMP, params)  # ~19.8 m at 45 degrees
    assert res.clusters_found >= 1 and res.detections == ()


def test_diagonal_car_survives_because_the_filter_is_oriented(params):
    car = box_surface(15, 3, -1.7, 4.5, 1.8, 1.5, yaw_deg=45)
    res = pc.detect(scene(car), STAMP, params)
    assert len(res.detections) == 1
    d = res.detections[0]
    assert d.dx > 3.0 and d.dy > 3.0, "axis-aligned extents exceed the 3 m width limit, the oriented ones do not"


def test_diagonal_van_sized_box_survives(params):
    van = box_surface(18, -2, -1.7, 5.5, 1.9, 2.1, yaw_deg=60)
    assert len(pc.detect(scene(van), STAMP, params).detections) == 1


def test_size_filter_boundaries(params):
    ok = lambda n, l, w, h: pc.passes_size_filter(n, l, w, h, params)
    assert ok(10, 6.99, 1.8, 1.5) and not ok(10, 7.01, 1.8, 1.5)
    assert ok(10, 4.5, 2.99, 1.5) and not ok(10, 4.5, 3.01, 1.5)
    assert ok(10, 4.5, 1.8, 0.31) and not ok(10, 4.5, 1.8, 0.29)
    assert ok(10, 4.5, 1.8, 2.99) and not ok(10, 4.5, 1.8, 3.01)
    assert ok(10, 4.5, 1.8, 1.5) and not ok(9, 4.5, 1.8, 1.5)


def test_low_clutter_is_filtered_by_height(params):
    curb = box_surface(15, 5, -1.7, 4.0, 1.0, 0.3)  # only the top layer survives ground removal
    assert pc.detect(scene(curb), STAMP, params).detections == ()


# ---------------------------------------------------------------- merge rule

def two_boxes(gap):
    a = box_surface(16.0, 0.0, -1.7, 2.0, 1.0, 1.2)  # x in [15, 17]
    b = box_surface(17.0 + gap + 1.0, 0.0, -1.7, 2.0, 1.0, 1.2)
    return scene(a, b)


def test_boxes_half_a_metre_apart_merge_into_one_detection(params):
    res = pc.detect(two_boxes(0.5), STAMP, params)
    assert len(res.detections) == 1  # eps 0.6 bridges a 0.5 m gap
    assert res.detections[0].dx > 4.0


def test_boxes_eight_tenths_apart_stay_two_detections(params):
    assert len(pc.detect(two_boxes(0.8), STAMP, params).detections) == 2


# ---------------------------------------------------------------- determinism and voxel order

def test_same_cloud_same_stamp_gives_identical_results(params):
    cloud = scene(box_surface(15, 3, -1.7, 4.5, 1.8, 1.5))
    a, b = pc.detect(cloud, STAMP, params), pc.detect(cloud, STAMP, params)
    assert a.detections == b.detections and a.stages == b.stages
    assert np.array_equal(a.filtered_xyz, b.filtered_xyz)


def test_point_order_does_not_matter(params):
    cloud = scene(box_surface(15, 3, -1.7, 4.5, 1.8, 1.5))
    shuffled = cloud[np.random.default_rng(3).permutation(len(cloud))]
    a, b = pc.detect(cloud, STAMP, params), pc.detect(shuffled, STAMP, params)
    assert a.detections == b.detections and np.array_equal(a.filtered_xyz, b.filtered_xyz)


def test_ransac_stream_depends_on_seed_and_stamp(params):
    draw = lambda seed, stamp: np.random.default_rng([seed, stamp]).integers(0, 1000, size=(params.ransac_iters, 3))
    assert np.array_equal(draw(42, STAMP), draw(42, STAMP))
    assert not np.array_equal(draw(42, STAMP), draw(43, STAMP))
    assert not np.array_equal(draw(42, STAMP), draw(42, STAMP + 1))


def test_ground_mask_is_identical_across_calls_and_changes_with_seed(params):
    pts = pc.voxel_downsample(scene(box_surface(15, 3, -1.7, 4.5, 1.8, 1.5)), 0.15)[0]
    m1, _ = pc.fit_ground(pts, STAMP, params)
    m2, _ = pc.fit_ground(pts, STAMP, params)
    assert np.array_equal(m1, m2)


# ---------------------------------------------------------------- voxel grid

def test_voxel_centers_are_grid_centers_including_negative_coordinates():
    pts = np.array([[0.01, 0.02, 0.03], [0.1, 0.1, 0.1], [-0.01, -0.02, -0.03], [1.0, -1.0, 0.5]])
    centers, ijk = pc.voxel_downsample(pts, 0.15)
    assert len(centers) == 3  # the first two share a voxel
    assert np.allclose(centers, (ijk + 0.5) * 0.15)
    assert ((-1, -1, -1) == ijk).all(axis=1).any() and ((0, 0, 0) == ijk).all(axis=1).any()


def test_voxel_downsample_empty():
    centers, ijk = pc.voxel_downsample(np.empty((0, 3)), 0.15)
    assert centers.shape == (0, 3) and ijk.shape == (0, 3)


# ---------------------------------------------------------------- clustering equivalence

def same_partition(a, b):
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len({x for x, _ in pairs}) == len({y for _, y in pairs}) and \
        all((x == -1) == (y == -1) for x, y in pairs)


def test_column_dbscan_equals_plain_dbscan_on_every_voxel(params):
    cloud = scene(box_surface(15, 3, -1.7, 4.5, 1.8, 1.5), box_surface(25, -6, -1.7, 4.0, 1.7, 1.4, yaw_deg=30),
                  wall(20, 8, 20, 12), ground=None)
    centers, ijk = pc.voxel_downsample(cloud, params.voxel_size)
    plain = DBSCAN(eps=params.dbscan_eps, min_samples=params.dbscan_min_samples).fit(centers[:, :2]).labels_
    assert same_partition(pc.cluster_xy(ijk, centers, params), plain)
    assert len(set(plain) - {-1}) >= 3


def test_cluster_xy_on_empty_input(params):
    assert len(pc.cluster_xy(np.empty((0, 3), dtype=np.int64), np.empty((0, 3)), params)) == 0


# ---------------------------------------------------------------- oriented extents

def rect(l, w, yaw_deg, step=0.05):
    u, v = np.meshgrid(np.arange(-l / 2, l / 2 + 1e-9, step), np.arange(-w / 2, w / 2 + 1e-9, step))
    t = np.radians(yaw_deg)
    xy = np.stack([u.ravel(), v.ravel()], axis=1) @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T
    return xy + (10.0, -3.0)


@pytest.mark.parametrize("yaw", [0, 20, 45, 77, 135])
def test_oriented_extents_recover_the_rectangle_at_any_yaw(yaw):
    length, width = pc.oriented_extents(rect(4.5, 1.8, yaw), 0.0)
    assert length == pytest.approx(4.5, abs=0.06) and width == pytest.approx(1.8, abs=0.06)


def test_oriented_extents_add_one_cell():
    length, width = pc.oriented_extents(rect(4.5, 1.8, 30), 0.15)
    assert length == pytest.approx(4.65, abs=0.06) and width == pytest.approx(1.95, abs=0.06)


def test_oriented_extents_degenerate_inputs():
    assert pc.oriented_extents(np.array([[1.0, 2.0]]), 0.15) == (0.15, 0.15)
    assert pc.oriented_extents(np.array([[0.0, 0.0], [3.0, 4.0]]), 0.0)[0] == pytest.approx(5.0)
    line = np.column_stack([np.linspace(0, 6, 40), np.linspace(0, 8, 40)])  # collinear: hull is degenerate
    length, width = pc.oriented_extents(line, 0.0)
    assert length == pytest.approx(10.0, abs=1e-6) and width == pytest.approx(0.0, abs=1e-6)
