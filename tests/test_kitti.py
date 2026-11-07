"""Data layer: timestamps, sweeps, OXTS, calibration, tracklets, failure paths, pinned inputs."""
import hashlib
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pytest

from core import kitti

from conftest import TRACKLET_XML, oxts_row

FRAME0_NS = 1317042272335337762  # timestamps.txt line 1: 2011-09-26 13:04:32.335337762, read as UTC

# SHA-256 of the extracted inputs (measured Nov 5, 2026). A reviewer with these files has the same inputs.
PINNED_SHA256 = {
    "2011_09_26_drive_0005_sync/velodyne_points/timestamps.txt": "33529e08de8e7604d64d4f00398b501497568312dbf9398b96655688c8bd8387",
    "calib_imu_to_velo.txt": "07ca9ed33a2d968fc6358617d2c70e048b6f8e0f6ee5b73ae963afc1b19456ce",
    "2011_09_26_drive_0005_sync/tracklet_labels.xml": "37b2fb315b0ed2cb8c1231fc2a685379ba3bccae25195ce22c68164bfc77072b",
}


# ---------------------------------------------------------------- timestamps (synthetic)

@pytest.mark.parametrize("line, expected", [
    ("2011-09-26 13:04:32.335337762", FRAME0_NS),
    ("2011-09-26 13:04:32.5", 1317042272_500000000),
    ("2011-09-26 13:04:32", 1317042272_000000000),
    ("2011-09-26 13:04:32.0000000011", 1317042272_000000001),  # extra digits are cut, not rounded
])
def test_parse_timestamp_exact_ns(line, expected):
    assert kitti.parse_timestamp_ns(line) == expected


def test_parse_timestamp_ignores_local_timezone():
    code = "from core import kitti; print(kitti.parse_timestamp_ns('2011-09-26 13:04:32.335337762'))"
    for tz in ("UTC", "Europe/Berlin", "America/Los_Angeles"):
        out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TZ": tz},
                             capture_output=True, text=True, check=True).stdout
        assert int(out) == FRAME0_NS, tz


# ---------------------------------------------------------------- synthetic drive: success and failure paths

def test_synthetic_drive_loads(make_drive):
    data_dir, drive = make_drive(n=3, points=5)
    d = kitti.load_drive(data_dir, drive)
    assert d.n_frames == 3 and d.oxts.shape == (3, 30) and d.T_velo_imu.shape == (4, 4)
    assert d.stamps_ns.dtype == np.int64 and d.stamps_ns[0] == FRAME0_NS
    assert kitti.load_sweep(d.sweep_paths[0]).shape == (5, 4)
    assert d.frame_of_stamp(int(d.stamps_ns[2])) == 2
    assert d.frame_of_stamp(int(d.stamps_ns[2]) + 1) is None


def test_missing_data_folder_names_download_script(tmp_path):
    with pytest.raises(kitti.MissingDataError, match="download_kitti.sh") as exc:
        kitti.load_drive(tmp_path / "nothing", "2011_09_26_drive_0005")
    assert "2011_09_26_drive_0005_sync" in str(exc.value)


def test_missing_timestamps_names_download_script(make_drive):
    data_dir, drive = make_drive()
    (kitti.drive_dir(data_dir, drive) / "velodyne_points" / "timestamps.txt").unlink()
    with pytest.raises(kitti.MissingDataError, match=r"timestamps\.txt.*download_kitti\.sh"):
        kitti.load_drive(data_dir, drive)


def test_sweep_timestamp_count_mismatch(make_drive):
    data_dir, drive = make_drive(n=3, stamps=["2011-09-26 13:04:32.1", "2011-09-26 13:04:32.2"])
    with pytest.raises(kitti.DataError, match="3 sweeps but 2 timestamps"):
        kitti.load_drive(data_dir, drive)


def test_stamps_must_increase(make_drive):
    data_dir, drive = make_drive(n=2, stamps=["2011-09-26 13:04:32.2", "2011-09-26 13:04:32.1"])
    with pytest.raises(kitti.DataError, match="strictly increasing"):
        kitti.load_drive(data_dir, drive)


@pytest.mark.parametrize("size", [0, 16 * 5 + 7])
def test_truncated_sweep_names_the_file(make_drive, size):
    data_dir, drive = make_drive(n=3)
    bad = kitti.drive_dir(data_dir, drive) / "velodyne_points" / "data" / "0000000001.bin"
    bad.write_bytes(b"\0" * size)
    with pytest.raises(kitti.DataError, match=r"truncated sweep .*0000000001\.bin"):
        kitti.load_drive(data_dir, drive)
    with pytest.raises(kitti.DataError, match="truncated sweep"):
        kitti.load_sweep(bad)


def test_missing_oxts_row(make_drive):
    data_dir, drive = make_drive(n=3, oxts_rows=[oxts_row(), oxts_row()])
    with pytest.raises(kitti.DataError, match="3 sweeps but 2 OXTS rows"):
        kitti.load_drive(data_dir, drive)


def test_malformed_oxts_row_names_the_file(make_drive):
    short = " ".join(["0"] * 29)
    data_dir, drive = make_drive(n=3, oxts_rows=[oxts_row(), short, oxts_row()])
    with pytest.raises(kitti.DataError, match=r"0000000001\.txt: 29 fields, expected 30"):
        kitti.load_drive(data_dir, drive)


def test_non_numeric_oxts_field_names_the_file(make_drive):
    garbled = " ".join(["0"] * 29 + ["nan?"])
    data_dir, drive = make_drive(n=3, oxts_rows=[oxts_row(), oxts_row(), garbled])
    with pytest.raises(kitti.DataError, match=r"bad OXTS row .*0000000002\.txt"):
        kitti.load_drive(data_dir, drive)


def test_non_numeric_calibration_names_the_file(make_drive):
    data_dir, drive = make_drive(calib="R: 1 0 0 0 1 0 0 0 x\nT: 0 0 0\n")
    with pytest.raises(kitti.DataError, match=r"bad calibration .*calib_imu_to_velo\.txt"):
        kitti.load_drive(data_dir, drive)


def test_missing_calibration_names_download_script(make_drive):
    data_dir, drive = make_drive(calib=None)
    with pytest.raises(kitti.MissingDataError, match=r"calib_imu_to_velo\.txt.*download_kitti\.sh"):
        kitti.load_drive(data_dir, drive)


def test_missing_tracklets_raises(make_drive):
    data_dir, drive = make_drive()
    d = kitti.load_drive(data_dir, drive)
    with pytest.raises(kitti.MissingDataError, match="tracklet_labels.xml"):
        kitti.load_tracklets(d.tracklet_path)


def test_synthetic_tracklet_and_box_center(make_drive):
    data_dir, drive = make_drive(tracklets_xml=TRACKLET_XML)
    (t,) = kitti.load_tracklets(kitti.drive_dir(data_dir, drive) / "tracklet_labels.xml")
    assert (t.id, t.cls, t.first_frame, len(t.poses)) == (0, "Car", 1, 2)
    assert list(t.frames) == [1, 2]
    boxes = kitti.gt_boxes([t])
    assert [b.frame for b in boxes] == [1, 2]
    assert boxes[0].z == pytest.approx(-1.7 + 1.5 / 2)  # tz is the box bottom
    assert (boxes[0].x, boxes[0].y, boxes[0].yaw) == (10.0, 2.0, 0.5)


def test_tracklet_pose_count_mismatch(make_drive):
    data_dir, drive = make_drive(tracklets_xml=TRACKLET_XML.replace("<count>2</count>", "<count>3</count>"))
    with pytest.raises(kitti.DataError, match="declares 3 poses, has 2"):
        kitti.load_tracklets(kitti.drive_dir(data_dir, drive) / "tracklet_labels.xml")


def test_tracklet_entity_bomb_fails_fast(make_drive):
    """Security: a billion-laughs file must give a clean error, not hang or eat memory."""
    entities = '<!ENTITY a "aaaaaaaaaa">' + "".join(
        f'<!ENTITY {chr(98 + i)} "{("&" + chr(97 + i) + ";") * 10}">' for i in range(9))
    bomb = f'<?xml version="1.0"?><!DOCTYPE l [{entities}]><boost_serialization><tracklets>&j;</tracklets></boost_serialization>'
    data_dir, drive = make_drive(tracklets_xml=bomb)
    start = time.monotonic()
    with pytest.raises(kitti.DataError, match="cannot parse tracklet file"):
        kitti.load_tracklets(kitti.drive_dir(data_dir, drive) / "tracklet_labels.xml")
    assert time.monotonic() - start < 5


def test_tracklet_external_entity_is_not_resolved(make_drive, tmp_path):
    """Security: an external entity must not pull a local file into the parsed data."""
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET")
    xxe = (f'<?xml version="1.0"?><!DOCTYPE b [<!ENTITY x SYSTEM "file://{secret}">]>'
           "<boost_serialization><tracklets><count>0</count><note>&x;</note></tracklets></boost_serialization>")
    data_dir, drive = make_drive(tracklets_xml=xxe)
    try:
        result = kitti.load_tracklets(kitti.drive_dir(data_dir, drive) / "tracklet_labels.xml")
    except kitti.DataError:
        return
    assert "SECRET" not in repr(result)


# ---------------------------------------------------------------- the real drive

def test_real_drive_counts(drive):
    assert drive.n_frames == 154
    assert len(drive.stamps_ns) == 154
    assert drive.oxts.shape == (154, 30)


def test_real_first_stamp_is_line_one(drive):
    assert int(drive.stamps_ns[0]) == FRAME0_NS


def test_real_frame0_sweep(drive):
    pts = kitti.load_sweep(drive.sweep_paths[0])
    assert pts.shape == (123397, 4)
    assert pts.dtype == np.float32


def test_real_tracklets(tracklets):
    assert len(tracklets) == 15
    assert sum(len(t.poses) for t in tracklets) == 717
    assert Counter(t.cls for t in tracklets) == {"Car": 9, "Van": 3, "Pedestrian": 2, "Cyclist": 1}


def test_real_boxes_use_center_height(tracklets):
    t = tracklets[0]
    box = kitti.gt_boxes([t])[0]
    assert box.z == pytest.approx(t.poses[0, 2] + t.h / 2)
    assert len(kitti.gt_boxes(tracklets)) == 717


@pytest.mark.parametrize("relpath", sorted(PINNED_SHA256))
def test_real_inputs_match_pinned_sha256(drive, relpath):
    path = drive.root.parent / relpath
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_SHA256[relpath], (
        f"{path} differs from the input these results were measured on; re-run scripts/download_kitti.sh")
