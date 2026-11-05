"""Shared fixtures. Tests run inside the processor image: no ROS, no publisher.

Real-data tests use the drive mounted at KITTI_DATA_DIR (default /data). If it
is missing they FAIL with the download hint instead of skipping, so a green
verify always means the data was checked. Use ``-m "not data"`` only for quick
local iteration; it is not evidence.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from core import kitti

DATA_DIR = Path(os.environ.get("KITTI_DATA_DIR", "/data"))
DRIVE = "2011_09_26_drive_0005"


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the real KITTI drive under KITTI_DATA_DIR")


def pytest_collection_modifyitems(items):
    for item in items:
        if "drive" in getattr(item, "fixturenames", ()) or "tracklets" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.data)


@pytest.fixture(scope="session")
def drive():
    try:
        return kitti.load_drive(DATA_DIR, DRIVE)
    except kitti.MissingDataError as exc:
        pytest.fail(f"KITTI drive not available ({exc}). Real-data tests never skip.", pytrace=False)


@pytest.fixture(scope="session")
def tracklets(drive):
    try:
        return kitti.load_tracklets(drive.tracklet_path)
    except kitti.MissingDataError as exc:
        pytest.fail(f"{exc}. Real-data tests never skip.", pytrace=False)


# ---------------------------------------------------------------- synthetic drives

# One Car tracklet over frames 1-2, in the boost_serialization layout of tracklet_labels.xml.
TRACKLET_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<!DOCTYPE boost_serialization>
<boost_serialization signature="serialization::archive" version="9">
<tracklets class_id="0" tracking_level="0" version="0">
  <count>1</count><item_version>1</item_version>
  <item class_id="1" tracking_level="0" version="1">
    <objectType>Car</objectType><h>1.5</h><w>1.8</w><l>4.0</l><first_frame>1</first_frame>
    <poses class_id="2" tracking_level="0" version="0"><count>2</count><item_version>2</item_version>
      <item><tx>10</tx><ty>2</ty><tz>-1.7</tz><rx>0</rx><ry>0</ry><rz>0.5</rz><state>1</state></item>
      <item><tx>11</tx><ty>2</ty><tz>-1.7</tz><rx>0</rx><ry>0</ry><rz>0.5</rz><state>1</state></item>
    </poses><finished>1</finished>
  </item>
</tracklets>
</boost_serialization>
"""


CALIB_TEXT = (
    "calib_time: 25-May-2012 16:47:16\n"
    "R: 1 0 0 0 1 0 0 0 1\n"
    "T: -0.8 0.3 -0.8\n"
)


def format_stamp(ns):
    """Integer ns since the epoch -> a KITTI timestamps.txt line (UTC)."""
    sec, frac = divmod(ns, 1_000_000_000)
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + f".{frac:09d}"


def oxts_row(lat=49.0, lon=8.4, alt=110.0, roll=0.0, pitch=0.0, yaw=0.0, **extra):
    """A 30-field OXTS row; keyword names follow oxts/dataformat.txt (vn, ve, vu, wx, wy, wz)."""
    row = np.zeros(kitti.OXTS_FIELDS)
    row[:6] = lat, lon, alt, roll, pitch, yaw
    columns = {"vn": 6, "ve": 7, "vf": 8, "vl": 9, "vu": 10, "wx": 17, "wy": 18, "wz": 19}
    for name, value in extra.items():
        row[columns[name]] = value
    return row


@pytest.fixture
def make_drive(tmp_path):
    """Build a small synthetic drive and return (data_dir, drive). Arguments break it on purpose."""

    def build(n=3, points=5, stamps=None, oxts_rows=None, tracklets_xml=None, calib=CALIB_TEXT):
        root = kitti.drive_dir(tmp_path, DRIVE)
        (root / "velodyne_points" / "data").mkdir(parents=True)
        (root / "oxts" / "data").mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in range(n):
            rng.random((points, 4), dtype=np.float32).tofile(root / "velodyne_points" / "data" / f"{i:010d}.bin")
        if stamps is None:
            stamps = [format_stamp(1317042272335337762 + i * 103_000_000) for i in range(n)]
        (root / "velodyne_points" / "timestamps.txt").write_text("\n".join(stamps) + "\n")
        rows = oxts_rows if oxts_rows is not None else [oxts_row() for _ in range(n)]
        for i, row in enumerate(rows):
            text = row if isinstance(row, str) else " ".join(repr(float(v)) for v in row)
            (root / "oxts" / "data" / f"{i:010d}.txt").write_text(text + "\n")
        if calib is not None:
            (tmp_path / DRIVE[:10] / "calib_imu_to_velo.txt").write_text(calib)
        if tracklets_xml is not None:
            (root / "tracklet_labels.xml").write_text(tracklets_xml)
        return tmp_path, DRIVE

    return build
