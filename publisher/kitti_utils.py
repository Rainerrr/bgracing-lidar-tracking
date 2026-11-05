"""Loading helpers for a KITTI raw "synced + rectified" drive."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def drive_dir(data_dir, drive):
    """data_dir/<date>/<drive>_sync, e.g. 2011_09_26/2011_09_26_drive_0005_sync."""
    date = drive[:10]
    return Path(data_dir) / date / f"{drive}_sync"


def list_frames(velo_dir):
    """Sorted .bin paths of a drive's velodyne_points/data folder."""
    frames = sorted(Path(velo_dir).glob("*.bin"))
    if not frames:
        raise FileNotFoundError(f"no .bin files in {velo_dir}; run scripts/download_kitti.sh")
    return frames


def parse_timestamp_ns(line):
    """'2011-09-26 13:02:25.964389376' -> integer nanoseconds since the Unix epoch."""
    day, _, frac = line.strip().partition(".")
    dt = datetime.strptime(day, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1_000_000_000 + int(frac.ljust(9, "0")[:9])


def load_timestamps_ns(path):
    return [parse_timestamp_ns(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_bin(path):
    """float32 array of shape (N, 4): x, y, z, reflectance."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
