"""Offline runner CLI: failure paths and the files it writes (synthetic drive, no real data needed)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import TRACKER_CONFIG, TRACKLET_XML

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_offline.py"


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_missing_data_exits_1_with_the_download_hint(tmp_path):
    r = cli("--config", str(TRACKER_CONFIG), "--data-dir", str(tmp_path / "nothing"), "--out-dir", str(tmp_path / "o"))
    assert r.returncode == 1 and "download_kitti.sh" in r.stderr


def test_live_output_folder_is_refused_before_writing(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text(Path(TRACKER_CONFIG).read_text().replace("output_dir: /output", f"output_dir: {tmp_path / 'live'}"))
    r = cli("--config", str(cfg), "--out-dir", str(tmp_path / "live"))
    assert r.returncode == 1 and "live output folder" in r.stderr and not (tmp_path / "live").exists()


def test_bad_config_names_the_key(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text(Path(TRACKER_CONFIG).read_text().replace("seed: 42", "sed: 42"))
    r = cli("--config", str(cfg), "--out-dir", str(tmp_path / "o"))
    assert r.returncode == 1 and ("unknown key 'sed'" in r.stderr or "missing key 'seed'" in r.stderr)


def test_synthetic_drive_writes_all_four_files(make_drive, tmp_path):
    data_dir, _ = make_drive(n=3, points=40, tracklets_xml=TRACKLET_XML)
    out = tmp_path / "o"
    r = cli("--config", str(TRACKER_CONFIG), "--data-dir", str(data_dir), "--out-dir", str(out))
    assert r.returncode == 0, r.stderr
    assert (out / "detections.csv").read_text().splitlines()[0] == "frame,stamp_ns,x,y,z,dx,dy,dz,n_points"
    frames = (out / "frames.csv").read_text().splitlines()
    assert len(frames) == 4 and frames[1].split(",")[0] == "0"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["frames_processed"] == 3 and metrics["mode"] == "offline" and metrics["seed"] == 42
    assert {"machine", "numpy", "sklearn"} <= metrics["platform"].keys() and "scope" in metrics["time_ms"]
    assert json.loads((out / "detection_recall.json").read_text())["ground_truth"] in ("present", "missing")


def test_missing_tracklets_still_writes_detections(make_drive, tmp_path):
    data_dir, _ = make_drive(n=3, points=40)
    out = tmp_path / "o"
    r = cli("--config", str(TRACKER_CONFIG), "--data-dir", str(data_dir), "--out-dir", str(out))
    assert r.returncode == 0 and "ground truth missing" in r.stderr
    assert json.loads((out / "detection_recall.json").read_text())["ground_truth"] == "missing"
    assert (out / "detections.csv").is_file() and (out / "metrics.json").is_file()
