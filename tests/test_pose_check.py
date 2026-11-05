"""E1.2 pose check: parked ground truth stays still in the world frame, and the check can fail.

All distances are 2D (x, y) box centers, matching the 2D tracker. The bar (1.2 m)
was set in the Phase 1 spec before this code existed, from the measured worst
case of 1.088 m. Never raise it to make this pass.
"""
from collections import defaultdict

import numpy as np
import pytest

from core import kitti, poses

PARKED = (3, 5, 6, 7, 8, 9, 10, 11, 13, 14)  # world net speed < 0.5 m/s (spec's static definition)
BAR_M = 1.2
NEGATIVE_MIN_M = 3 * BAR_M
MOVING = {0: 5.74, 1: 4.38}  # lead van, cyclist: world net speed (m/s)
MOVING_TOL = 0.3
STATIC_SPEED = 0.5


@pytest.fixture(scope="module")
def tracks(drive, tracklets):
    """Per tracklet: frames, xy centers in the world frame and in the velodyne frame."""
    T_wv = poses.world_velo_poses(drive.oxts, drive.T_velo_imu)
    by_id = defaultdict(list)
    for box in kitti.gt_boxes(tracklets):
        by_id[box.id].append(box)
    out = {}
    for tid, boxes in by_id.items():
        frames = np.array([b.frame for b in boxes])
        velo = np.array([[b.x, b.y, b.z] for b in boxes])
        world = np.array([poses.transform_points(T_wv[f], p) for f, p in zip(frames, velo)])
        out[tid] = {"cls": boxes[0].cls, "frames": frames, "velo": velo[:, :2], "world": world[:, :2]}
    return out


def max_deviation(xy):
    return float(np.linalg.norm(xy - xy.mean(axis=0), axis=1).max())


def net_speed(xy, frames, stamps_ns):
    dt = (stamps_ns[frames[-1]] - stamps_ns[frames[0]]) / 1e9
    return float(np.linalg.norm(xy[-1] - xy[0]) / dt)


def test_parked_tracklets_stay_still_in_world(tracks):
    """E1.2a: each parked tracklet stays within 1.2 m of its own mean position."""
    rows = [(tid, tracks[tid]["cls"], len(tracks[tid]["frames"]), max_deviation(tracks[tid]["world"]),
             max_deviation(tracks[tid]["velo"])) for tid in PARKED]
    table = "\n".join(["tracklet  class  frames  world_max_dev_m  velodyne_max_dev_m"] +
                      [f"#{t:<8} {c:<6} {n:>6}  {w:>15.3f}  {v:>18.3f}" for t, c, n, w, v in rows])
    worlds = [w for *_, w, _ in rows]
    print(f"\nE1.2 pose check (2D box centers), bar {BAR_M:.1f} m, negative control >= {NEGATIVE_MIN_M:.1f} m\n{table}\n"
          f"world: median {np.median(worlds):.3f} m, worst {max(worlds):.3f} m")
    failing = [(t, round(w, 3)) for t, _, _, w, _ in rows if w > BAR_M]
    assert not failing, f"parked tracklets drift more than {BAR_M} m in the world frame: {failing}\n{table}"


def test_negative_control_parked_tracklets_move_in_velodyne_frame(tracks):
    """E1.2b: without the ego pose the same objects sweep past at ego speed, so the check can fail."""
    too_still = [(t, round(max_deviation(tracks[t]["velo"]), 3)) for t in PARKED
                 if max_deviation(tracks[t]["velo"]) < NEGATIVE_MIN_M]
    assert not too_still, f"velodyne-frame deviation below {NEGATIVE_MIN_M} m: {too_still}"


@pytest.mark.parametrize("tid, expected", sorted(MOVING.items()))
def test_moving_control_keeps_real_motion(tracks, drive, tid, expected):
    """E1.2c: a transform that froze everything would also freeze the lead van and the cyclist."""
    t = tracks[tid]
    assert net_speed(t["world"], t["frames"], drive.stamps_ns) == pytest.approx(expected, abs=MOVING_TOL)


def test_parked_list_follows_its_rule(tracks, drive):
    """The fixture list is exactly the tracklets slower than 0.5 m/s in the world frame."""
    static = sorted(t for t, v in tracks.items()
                    if len(v["frames"]) > 1 and net_speed(v["world"], v["frames"], drive.stamps_ns) < STATIC_SPEED)
    assert tuple(static) == PARKED
