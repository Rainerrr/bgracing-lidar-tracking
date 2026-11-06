"""Multi-object tracker (docs/spec.md steps 8-11): constant-velocity Kalman filter per track, gated
Hungarian (or greedy) association, confirm/delete lifecycle and motion state. ROS-free.

Both runners call the same Tracker.step() with the same world-frame detections, so live and offline give
the same tracks for the same processed frames (E3.4).

Time and gaps (phase-3 FEATURE_SPEC D4):
- Delta t comes from the dataset stamps, never from wall time.
- A forward gap of k missing frames counts as k misses for every track before the frame is processed;
  a forward gap never resets anything.
- A duplicate stamp is skipped (step returns None) and counted.
- A backward stamp is the dataset loop: the caller increments its pass and calls reset() first; step()
  refuses a backward stamp so a missed reset can't go unnoticed.

Motion state (phase-3 FEATURE_SPEC D10): a confirmed track is "moving" only if it is confidently faster than
moving_speed: speed - moving_sigma * sigma_speed > moving_speed, where sigma_speed is the filter's velocity
standard deviation along the direction of travel. Young or drifting tracks have a large sigma and stay static.

Track IDs are given at confirmation, starting at 1 after every reset, so (pass, track_id) is the key and
confirmed IDs are consecutive.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from core import poses

_BIG = 1e6  # finite cost for gated pairs: linear_sum_assignment raises on a row of inf
_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_I4 = np.eye(4)


@dataclass(frozen=True)
class WorldDetection:
    """A detection with its center in the tracking frame. dx, dy, dz stay the velodyne-axis extents."""
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float


@dataclass(frozen=True)
class TrackState:
    """One confirmed track at one frame (one tracks.csv row, plus z for markers)."""
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    state: str  # "moving" or "static" (confident rule, see the module docstring)
    l: float
    w: float
    h: float
    z: float
    frames_since_confirm: int  # frame index minus the frame the track was confirmed in


def to_world(detections, T):
    """Detection centers moved by the 4x4 T (identity when ego compensation is off)."""
    if not detections:
        return []
    c = poses.transform_points(T, [(d.x, d.y, d.z) for d in detections])
    return [WorldDetection(float(c[i, 0]), float(c[i, 1]), float(c[i, 2]), d.dx, d.dy, d.dz)
            for i, d in enumerate(detections)]


class _Track:
    __slots__ = ("x", "P", "hits", "misses", "track_id", "confirm_frame", "l", "w", "h", "z")

    def __init__(self, det, p):
        self.x = np.array([det.x, det.y, 0.0, 0.0])
        self.P = np.diag([p.r_pos ** 2, p.r_pos ** 2, p.v0_sigma ** 2, p.v0_sigma ** 2])
        self.hits, self.misses = 1, 0
        self.track_id = None  # given at confirmation
        self.confirm_frame = None
        self._shape(det)

    def _shape(self, det):
        self.l, self.w, self.h, self.z = det.dx, det.dy, det.dz, det.z


class Tracker:
    def __init__(self, params):
        self.p = params
        self.r2 = params.r_pos ** 2
        self.duplicates = 0
        self.confirmed_total = 0
        self.deleted_total = 0
        self.reset()

    def reset(self):
        """Forget every track; the next confirmed track gets ID 1. Counters are kept."""
        self.tracks = []
        self.next_id = 1
        self.last_stamp = None
        self.last_frame = None

    # ------------------------------------------------------------ filter

    def _predict(self, t, dt):
        F = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        q = self.p.q_accel
        a, b, c = q * dt ** 4 / 4.0, q * dt ** 3 / 2.0, q * dt ** 2
        Q = np.array([[a, 0.0, b, 0.0], [0.0, a, 0.0, b], [b, 0.0, c, 0.0], [0.0, b, 0.0, c]])
        t.x = F @ t.x
        t.P = F @ t.P @ F.T + Q

    def _update(self, t, det):
        z = np.array([det.x, det.y])
        S = _H @ t.P @ _H.T + self.r2 * np.eye(2)
        K = t.P @ _H.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - _H @ t.x)
        A = _I4 - K @ _H
        t.P = A @ t.P @ A.T + self.r2 * (K @ K.T)  # Joseph form: stays symmetric and positive
        t._shape(det)

    # ------------------------------------------------------------ association

    def _associate(self, dets):
        """[(track index, detection index)] within the gate."""
        if not self.tracks or not dets:
            return []
        pred = np.array([t.x[:2] for t in self.tracks])
        meas = np.array([(d.x, d.y) for d in dets])
        dist = np.hypot(pred[:, None, 0] - meas[None, :, 0], pred[:, None, 1] - meas[None, :, 1])
        gate = self.p.gate_m
        if self.p.association == "greedy":
            pairs, used_t, used_d = [], set(), set()
            ti, di = np.nonzero(dist <= gate)
            for c, i, j in sorted(zip(dist[ti, di].tolist(), ti.tolist(), di.tolist())):
                if i not in used_t and j not in used_d:
                    pairs.append((i, j))
                    used_t.add(i)
                    used_d.add(j)
            return sorted(pairs)
        rows, cols = linear_sum_assignment(np.where(dist <= gate, dist, _BIG))
        return [(int(r), int(c)) for r, c in zip(rows, cols) if dist[r, c] <= gate]

    # ------------------------------------------------------------ one frame

    def _miss(self, t, n):
        """Add n misses. Returns True if the track survives."""
        t.misses += n
        return t.track_id is not None and t.misses < self.p.max_misses

    def step(self, frame, stamp_ns, dets):
        """Process one frame. Returns the confirmed tracks (sorted by ID), or None for a duplicate stamp."""
        if self.last_stamp is not None:
            if stamp_ns == self.last_stamp:
                self.duplicates += 1
                return None
            if stamp_ns < self.last_stamp:
                raise ValueError(f"backward stamp {stamp_ns} < {self.last_stamp}: call reset() first")
            gap = max(frame - self.last_frame - 1, 0)
            if gap:
                self._retire([t for t in self.tracks if not self._miss(t, gap)])
            dt = (stamp_ns - self.last_stamp) * 1e-9
            for t in self.tracks:
                self._predict(t, dt)
        self.last_stamp, self.last_frame = stamp_ns, frame

        pairs = self._associate(dets)
        matched_t = {i for i, _ in pairs}
        matched_d = {j for _, j in pairs}
        for i, j in pairs:
            t = self.tracks[i]
            self._update(t, dets[j])
            t.hits += 1
            t.misses = 0
            if t.track_id is None and t.hits >= self.p.confirm_hits:
                t.track_id, t.confirm_frame = self.next_id, frame
                self.next_id += 1
                self.confirmed_total += 1
        self._retire([t for i, t in enumerate(self.tracks) if i not in matched_t and not self._miss(t, 1)])
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(_Track(d, self.p))

        out = []
        for t in self.tracks:
            if t.track_id is None:
                continue
            vx, vy = float(t.x[2]), float(t.x[3])
            speed = float(np.hypot(vx, vy))
            out.append(TrackState(t.track_id, float(t.x[0]), float(t.x[1]), vx, vy, speed, self._state(t, speed),
                                  t.l, t.w, t.h, t.z, frame - t.confirm_frame))
        return sorted(out, key=lambda s: s.track_id)

    def speed_sigma(self, t, speed):
        """Velocity standard deviation along the direction of travel (m/s)."""
        if speed <= 0.0:
            return float(np.sqrt(max(t.P[2, 2], t.P[3, 3])))
        u = t.x[2:] / speed
        return float(np.sqrt(max(u @ t.P[2:, 2:] @ u, 0.0)))

    def _state(self, t, speed):
        confident = speed - self.p.moving_sigma * self.speed_sigma(t, speed)
        return "moving" if confident > self.p.moving_speed else "static"

    def _retire(self, dead):
        if not dead:
            return
        ids = {id(t) for t in dead}
        self.deleted_total += sum(t.track_id is not None for t in dead)
        self.tracks = [t for t in self.tracks if id(t) not in ids]
