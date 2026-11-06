"""E3.1 tracker unit tests on synthetic targets (no data, no ROS). Bars from phase-3 FEATURE_SPEC D9."""
import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from processor import config, records, tracking
from processor.tracking import Tracker, WorldDetection

from conftest import TRACKER_CONFIG

P = config.load_params(TRACKER_CONFIG)
DT_NS = 100_000_000  # 0.1 s between frames
T0 = 1_317_042_272_335_337_762


def det(x, y):
    return WorldDetection(float(x), float(y), 0.5, 4.5, 1.8, 1.5)


def stamp(frame, dt_ns=DT_NS):
    return T0 + frame * dt_ns


def run(positions_by_frame, params=P, dt_ns=DT_NS):
    """positions_by_frame: list of lists of (x, y). Returns [tracks per frame]."""
    tr = Tracker(params)
    return [tr.step(f, stamp(f, dt_ns), [det(*p) for p in ps]) for f, ps in enumerate(positions_by_frame)]


def target(n, v=5.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = []
    for f in range(n):
        t = f * DT_NS * 1e-9
        e = rng.normal(0.0, noise, 2) if noise else (0.0, 0.0)
        pos.append([(v * t + e[0], 2.0 + e[1])])
    return pos


# ---------------------------------------------------------------- convergence (D9)

def test_noise_free_target_reads_within_0_05_from_frame_10():
    out = run(target(51))
    errors = [abs(out[f][0].speed - 5.0) for f in range(10, 51)]
    assert all(len(out[f]) == 1 for f in range(10, 51))
    assert max(errors) <= 0.05, f"max error {max(errors):.4f} m/s from frame 10"


def test_noisy_target_median_error_within_0_2_over_frames_10_to_50():
    out = run(target(51, noise=0.1, seed=0))
    errors = [abs(out[f][0].speed - 5.0) for f in range(10, 51)]
    assert all(len(out[f]) == 1 and out[f][0].track_id == 1 for f in range(10, 51))
    assert np.median(errors) <= 0.2, f"median error {np.median(errors):.4f} m/s"


def test_control_a_wrong_dt_breaks_the_convergence_bound():
    """Positions move at 0.1 s spacing but the stamps say 0.05 s: the filter reads about 10 m/s."""
    out = run(target(51), dt_ns=DT_NS // 2)
    assert abs(out[30][0].speed - 5.0) > 1.0


# ---------------------------------------------------------------- association

def test_crossing_targets_keep_their_ids():
    """A moves +y, B moves -y, 1.0 m apart sideways; they pass each other around frame 20."""
    frames = [[(0.0, -10.0 + 0.5 * f), (1.0, 10.0 - 0.5 * f)] for f in range(41)]
    out = run(frames)
    ids_a = {next(t.track_id for t in out[f] if t.x < 0.5) for f in range(5, 41)}
    ids_b = {next(t.track_id for t in out[f] if t.x >= 0.5) for f in range(5, 41)}
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def _tracker_with(positions, params):
    tr = Tracker(params)
    for x, y in positions:
        tr.tracks.append(tracking._Track(det(x, y), params))
    return tr


def test_greedy_differs_from_hungarian_on_a_documented_case():
    """Tracks at x=0 and x=1; detections at 0.9 and 2.5. Greedy takes the closest pair (track 1, 0.9) first and
    leaves track 0 with nothing in its gate; Hungarian matches both (0 -> 0.9, 1 -> 2.5)."""
    dets = [det(0.9, 0.0), det(2.5, 0.0)]
    hung = _tracker_with([(0.0, 0.0), (1.0, 0.0)], P)._associate(dets)
    greedy = _tracker_with([(0.0, 0.0), (1.0, 0.0)], dataclasses.replace(P, association="greedy"))._associate(dets)
    assert hung == [(0, 0), (1, 1)]
    assert greedy == [(1, 0)]


@pytest.mark.parametrize("offset,same", [(1.99, True), (2.01, False)])
def test_gate_edge(offset, same):
    frames = [[(0.0, 0.0)]] * 4 + [[(offset, 0.0)]]
    out = run(frames)
    assert out[3][0].track_id == 1
    if same:
        assert [t.track_id for t in out[4]] == [1] and out[4][0].x > 0.0
    else:
        assert [t.track_id for t in out[4]] == [1] and out[4][0].x == 0.0  # coasts; the far detection is tentative


# ---------------------------------------------------------------- lifecycle

def test_confirms_on_the_third_hit_and_is_deleted_on_the_fifth_miss():
    frames = [[(0.0, 0.0)]] * 3 + [[]] * 5
    out = run(frames)
    assert out[0] == [] and out[1] == []
    assert [t.track_id for t in out[2]] == [1]
    assert all([t.track_id for t in out[f]] == [1] for f in range(3, 7))  # misses 1..4: still there
    assert out[7] == []  # 5th miss


def test_a_tentative_track_that_misses_is_deleted_and_ids_stay_consecutive():
    frames = [[(0.0, 0.0)]] * 2 + [[]] + [[(0.0, 0.0)]] * 3
    out = run(frames)
    assert out[2] == [] and out[3] == [] and out[4] == []
    assert [t.track_id for t in out[5]] == [1]  # a fresh track; the dropped tentative used no ID


# ---------------------------------------------------------------- time: reset, gaps, duplicates

def test_backward_stamp_needs_a_reset_and_reset_restarts_ids():
    tr = Tracker(P)
    for f in range(4):
        tr.step(f, stamp(f), [det(0, 0)])
    with pytest.raises(ValueError, match="backward"):  # control: a runner that forgets to reset fails loudly
        tr.step(0, stamp(0), [det(5, 5)])
    tr.reset()
    assert tr.tracks == []
    out = [tr.step(f, stamp(f), [det(5, 5)]) for f in range(3)]
    assert [t.track_id for t in out[2]] == [1]


@pytest.mark.parametrize("gap,survives", [(3, True), (4, True), (5, False)])
def test_forward_gap_ages_tracks_by_missing_frames(gap, survives):
    tr = Tracker(P)
    for f in range(10):
        tr.step(f, stamp(f), [det(5.0 * f * 0.1, 0.0)])
    f = 10 + gap  # frames 10 .. 10+gap-1 are missing
    out = tr.step(f, stamp(f), [det(5.0 * f * 0.1, 0.0)])
    if survives:
        assert [t.track_id for t in out] == [1]
        assert abs(out[0].speed - 5.0) < 0.5
    else:
        assert out == [] and tr.deleted_total == 1


def test_tracker_never_reads_wall_time():
    """A processing stall changes wall time only; the tracker sees stamps and frame indices."""
    src = open(tracking.__file__).read()
    assert "import time" not in src and "datetime" not in src


def test_duplicate_stamp_is_skipped_and_counted():
    tr = Tracker(P)
    for f in range(3):
        tr.step(f, stamp(f), [det(0, 0)])
    hits = tr.tracks[0].hits
    assert tr.step(3, stamp(2), [det(0, 0)]) is None
    assert tr.duplicates == 1 and tr.tracks[0].hits == hits and len(tr.tracks) == 1


def test_empty_detection_sets_age_tracks_without_error():
    out = run([[]] * 3 + [[(0.0, 0.0)]] * 3 + [[]] * 2)
    assert out[0] == [] and [t.track_id for t in out[7]] == [1]


# ---------------------------------------------------------------- rows and state

def test_rows_are_sorted_formatted_and_carry_the_state():
    out = run([[(0.0, 0.0), (10.0, 0.0)]] * 3)
    rows = records.track_rows(0, 2, stamp(2), list(reversed(out[2])))
    assert [r.split(",")[3] for r in rows] == ["1", "2"]
    assert rows[0].count(",") == records.TRACK_HEADER.count(",")
    assert rows[0].split(",")[9] == "static" and rows[0].endswith(",4.5000,1.8000,1.5000")


def states(v, noise=0.0, n=60):
    return {f: o[0].state for f, o in enumerate(run(target(n, v=v, noise=noise))) if o}


def test_confident_motion_state_fast_target_is_moving_almost_at_once():
    s = states(5.0, noise=0.1)
    assert all(s[f] == "moving" for f in range(5, 60))


def test_confident_motion_state_stationary_noisy_target_is_never_moving():
    """D10: a 0.3 m-noise stationary target can read above 0.7 m/s, but never confidently."""
    assert set(states(0.0, noise=0.3).values()) == {"static"}


def test_confident_motion_state_young_tracks_wait_for_certainty():
    """1.5 m/s: static while the velocity is uncertain, moving once sigma has shrunk (measured first frame 13)."""
    s = states(1.5)
    assert s[5] == "static" and all(s[f] == "moving" for f in range(20, 60))


def test_confident_motion_state_documented_cost_a_1_m_s_walker_stays_static():
    """D10 tradeoff: sigma settles near 0.31 m/s, so the steady threshold is about 0.7 + 2 x 0.31 = 1.32 m/s."""
    tr = Tracker(P)
    for f, ps in enumerate(target(60, v=1.0)):
        out = tr.step(f, stamp(f), [det(*q) for q in ps])
    assert out[0].state == "static" and abs(out[0].speed - 1.0) < 0.05
    sigma = tr.speed_sigma(tr.tracks[0], out[0].speed)
    assert 0.25 < sigma < 0.35, sigma


def test_motion_state_k_zero_is_the_plain_speed_rule():
    p0 = dataclasses.replace(P, moving_sigma=0.0)
    out = run(target(60, v=1.0), params=p0)
    assert out[59][0].state == "moving"  # control: without the sigma term 1.0 m/s > 0.7 is moving


def test_tracker_imports_no_ros():
    code = "import sys, processor.tracking; print(','.join(m for m in sys.modules if m.split('.')[0] == 'rclpy'))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == ""
