"""Data layer for a KITTI raw "synced + rectified" drive.

Layout under ``data_dir`` (what ``scripts/download_kitti.sh`` produces)::

    <date>/calib_imu_to_velo.txt
    <date>/<drive>_sync/velodyne_points/{timestamps.txt, data/*.bin}
    <date>/<drive>_sync/oxts/data/*.txt
    <date>/<drive>_sync/tracklet_labels.xml

Time key: the velodyne ``timestamps.txt`` strings, read as UTC and turned into
integer nanoseconds. Nothing else is ever used as a time key.
"""
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

DOWNLOAD_HINT = "run scripts/download_kitti.sh (see the README)"
POINT_BYTES = 16  # float32 x, y, z, reflectance
OXTS_FIELDS = 30


class DataError(RuntimeError):
    """A drive file is inconsistent. The message names the file."""


class MissingDataError(DataError, FileNotFoundError):
    """A drive file or folder is missing. The message names the path and the download script."""


def _missing(path, what):
    return MissingDataError(f"{what} not found: {path}; {DOWNLOAD_HINT}")


# ---------------------------------------------------------------- timestamps

def parse_timestamp_ns(line):
    """'2011-09-26 13:04:32.335337762' -> integer ns since the Unix epoch, read as UTC."""
    day, _, frac = line.strip().partition(".")
    dt = datetime.strptime(day, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1_000_000_000 + int(frac.ljust(9, "0")[:9])


def load_timestamps_ns(path):
    path = Path(path)
    if not path.is_file():
        raise _missing(path, "timestamps file")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    try:
        return np.array([parse_timestamp_ns(line) for line in lines], dtype=np.int64)
    except ValueError as exc:
        raise DataError(f"bad timestamp in {path}: {exc}") from exc


# ---------------------------------------------------------------- sweeps

def list_sweeps(velo_data_dir):
    """Sorted .bin paths of a velodyne_points/data folder."""
    velo_data_dir = Path(velo_data_dir)
    sweeps = sorted(velo_data_dir.glob("*.bin"))
    if not sweeps:
        raise _missing(velo_data_dir, "velodyne sweeps (*.bin)")
    return sweeps


def check_sweep_file(path):
    """Reject empty or truncated sweeps (size not a whole number of points)."""
    size = Path(path).stat().st_size
    if size == 0 or size % POINT_BYTES:
        raise DataError(f"truncated sweep {path}: {size} bytes is not a positive multiple of {POINT_BYTES}")


def load_sweep(path):
    """float32 array of shape (N, 4): x, y, z, reflectance, in the velodyne frame."""
    check_sweep_file(path)
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


# ---------------------------------------------------------------- OXTS and calibration

def load_oxts(oxts_data_dir):
    """float64 array (N, 30), one row per OXTS file, columns as in oxts/dataformat.txt."""
    oxts_data_dir = Path(oxts_data_dir)
    files = sorted(oxts_data_dir.glob("*.txt"))
    if not files:
        raise _missing(oxts_data_dir, "OXTS rows (*.txt)")
    rows = []
    for f in files:
        fields = f.read_text().split()
        if len(fields) != OXTS_FIELDS:
            raise DataError(f"bad OXTS row {f}: {len(fields)} fields, expected {OXTS_FIELDS}")
        try:
            rows.append([float(x) for x in fields])
        except ValueError as exc:
            raise DataError(f"bad OXTS row {f}: {exc}") from exc
    return np.array(rows, dtype=np.float64)


def load_calib_imu_to_velo(path):
    """4x4 T_velo_imu: maps a point from the IMU frame into the velodyne frame."""
    path = Path(path)
    if not path.is_file():
        raise _missing(path, "IMU-to-velodyne calibration")
    values = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("R", "T"):
            try:
                values[key] = np.array(rest.split(), dtype=np.float64)
            except ValueError as exc:
                raise DataError(f"bad calibration {path}: {exc}") from exc
    if values.get("R", np.empty(0)).size != 9 or values.get("T", np.empty(0)).size != 3:
        raise DataError(f"bad calibration {path}: needs 'R:' with 9 values and 'T:' with 3")
    T = np.eye(4)
    T[:3, :3] = values["R"].reshape(3, 3)
    T[:3, 3] = values["T"]
    return T


# ---------------------------------------------------------------- the drive

def drive_dir(data_dir, drive):
    """data_dir/<date>/<drive>_sync, e.g. 2011_09_26/2011_09_26_drive_0005_sync."""
    return Path(data_dir) / drive[:10] / f"{drive}_sync"


@dataclass(frozen=True)
class Drive:
    root: Path
    sweep_paths: tuple
    stamps_ns: np.ndarray  # int64 (N,)
    oxts: np.ndarray  # float64 (N, 30)
    T_velo_imu: np.ndarray  # 4x4

    @property
    def n_frames(self):
        return len(self.sweep_paths)

    @property
    def tracklet_path(self):
        return self.root / "tracklet_labels.xml"

    def frame_of_stamp(self, stamp_ns):
        """Frame index whose stamp equals stamp_ns exactly, or None."""
        i = int(np.searchsorted(self.stamps_ns, stamp_ns))
        if i < len(self.stamps_ns) and self.stamps_ns[i] == stamp_ns:
            return i
        return None


def load_drive(data_dir, drive):
    """Load and validate everything except the point data and the tracklets.

    Every sweep file's size is checked, so a truncated download fails here,
    before anything is published or computed.
    """
    root = drive_dir(data_dir, drive)
    if not root.is_dir():
        raise _missing(root, "drive folder")
    velo = root / "velodyne_points"
    sweeps = list_sweeps(velo / "data")
    for path in sweeps:
        check_sweep_file(path)
    stamps = load_timestamps_ns(velo / "timestamps.txt")
    if len(stamps) != len(sweeps):
        raise DataError(f"{len(sweeps)} sweeps but {len(stamps)} timestamps in {velo}")
    if np.any(np.diff(stamps) <= 0):
        raise DataError(f"timestamps in {velo / 'timestamps.txt'} are not strictly increasing")
    oxts = load_oxts(root / "oxts" / "data")
    if len(oxts) != len(sweeps):
        raise DataError(f"{len(sweeps)} sweeps but {len(oxts)} OXTS rows in {root / 'oxts' / 'data'}")
    T_velo_imu = load_calib_imu_to_velo(Path(data_dir) / drive[:10] / "calib_imu_to_velo.txt")
    return Drive(root=root, sweep_paths=tuple(sweeps), stamps_ns=stamps, oxts=oxts, T_velo_imu=T_velo_imu)


# ---------------------------------------------------------------- tracklets (ground truth)

@dataclass(frozen=True)
class Tracklet:
    id: int  # position in tracklet_labels.xml, 0-based
    cls: str  # Car, Van, Pedestrian, Cyclist, ...
    l: float
    w: float
    h: float
    first_frame: int
    poses: np.ndarray  # (K, 6): tx, ty, tz, rx, ry, rz in the velodyne frame; tz is the box bottom

    @property
    def frames(self):
        return np.arange(self.first_frame, self.first_frame + len(self.poses))


GtBox = namedtuple("GtBox", "frame id cls x y z l w h yaw")
GtBox.__doc__ = "One ground-truth box in the velodyne frame; z is the box center."


def _text(elem, tag, path):
    child = elem.find(tag)
    if child is None or child.text is None:
        raise DataError(f"bad tracklet file {path}: missing <{tag}>")
    return child.text


def load_tracklets(path):
    """Parse tracklet_labels.xml. Raises MissingDataError if the file is absent."""
    path = Path(path)
    if not path.is_file():
        raise _missing(path, "tracklet file")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise DataError(f"cannot parse tracklet file {path}: {exc}") from exc
    container = root.find("tracklets")
    if container is None:
        raise DataError(f"bad tracklet file {path}: no <tracklets> element")
    tracklets = []
    for i, item in enumerate(container.findall("item")):
        pose_items = item.find("poses").findall("item") if item.find("poses") is not None else []
        poses = np.array(
            [[float(_text(p, k, path)) for k in ("tx", "ty", "tz", "rx", "ry", "rz")] for p in pose_items],
            dtype=np.float64,
        ).reshape(-1, 6)
        declared = int(_text(item.find("poses"), "count", path)) if item.find("poses") is not None else 0
        if declared != len(poses):
            raise DataError(f"bad tracklet file {path}: tracklet {i} declares {declared} poses, has {len(poses)}")
        tracklets.append(Tracklet(
            id=i,
            cls=_text(item, "objectType", path),
            l=float(_text(item, "l", path)),
            w=float(_text(item, "w", path)),
            h=float(_text(item, "h", path)),
            first_frame=int(_text(item, "first_frame", path)),
            poses=poses,
        ))
    declared = int(_text(container, "count", path))
    if declared != len(tracklets):
        raise DataError(f"bad tracklet file {path}: declares {declared} tracklets, has {len(tracklets)}")
    return tracklets


def gt_boxes(tracklets):
    """Flatten tracklets into per-frame boxes. The only place tz (box bottom) becomes the center."""
    boxes = []
    for t in tracklets:
        for frame, (tx, ty, tz, _rx, _ry, rz) in zip(t.frames, t.poses):
            boxes.append(GtBox(int(frame), t.id, t.cls, tx, ty, tz + t.h / 2, t.l, t.w, t.h, rz))
    return boxes
