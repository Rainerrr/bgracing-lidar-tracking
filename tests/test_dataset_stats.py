"""Dataset statistics: runs without the publisher, deterministic, complete, degrades without ground truth."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import DATA_DIR, DRIVE, TRACKLET_XML

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dataset_stats.py"
REQUIRED_KEYS = {
    "frames", "duration_s", "mean_rate_hz", "points_per_frame", "range_hist", "reflectance_hist",
    "ego_speed_mps", "gt_objects_per_frame", "ground_truth",
}


def load_script():
    spec = importlib.util.spec_from_file_location("dataset_stats", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(data_dir, out_dir):
    return subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(data_dir), "--drive", DRIVE,
                           "--out-dir", str(out_dir)], capture_output=True, text=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def two_real_runs(tmp_path_factory, drive):
    outs = []
    for name in ("run1", "run2"):
        out = tmp_path_factory.mktemp(name)
        result = run(DATA_DIR, out)
        assert result.returncode == 0, result.stderr
        outs.append(out)
    return outs


def test_two_runs_are_byte_identical(two_real_runs):
    a, b = two_real_runs
    for name in ("dataset_stats.json", "dataset_stats.png"):
        assert digest(a / name) == digest(b / name), name
    print(f"\njson sha256 {digest(a / 'dataset_stats.json')}\npng  sha256 {digest(a / 'dataset_stats.png')}")


def test_json_has_every_listed_quantity(two_real_runs):
    stats = json.loads((two_real_runs[0] / "dataset_stats.json").read_text())
    assert REQUIRED_KEYS <= stats.keys()
    stages = stats["stage_points"]
    order = ["raw", "roi", "voxel", "non_ground", "clustered"]  # pipeline order; the JSON itself is key-sorted
    assert set(stages) == set(order)
    means = [stages[k]["mean"] for k in order]
    assert means == sorted(means, reverse=True), "each stage keeps no more points than the one before"
    assert stages["raw"]["share_of_raw"] == 1.0 and all(0.0 < v["share_of_raw"] <= 1.0 for v in stages.values())
    assert stages["raw"]["series"] == stats["points_per_frame"]["series"]
    assert all(len(v["series"]) == 154 for v in stages.values()) and sum(stats["ground_status_counts"].values()) == 154
    assert stats["frames"] == 154 and stats["ground_truth"] == "present"
    assert stats["duration_s"] == pytest.approx(15.80, abs=0.01)
    assert stats["mean_rate_hz"] == pytest.approx(9.68, abs=0.05)
    assert stats["points_per_frame"]["series"][0] == 123397
    assert sum(stats["range_hist"]["counts"]) + stats["range_hist"]["beyond_last_edge"] == sum(
        stats["points_per_frame"]["series"])
    assert sum(stats["reflectance_hist"]["counts"]) == sum(stats["points_per_frame"]["series"])
    assert len(stats["ego_speed_mps"]["series"]) == 154
    assert 3.0 < stats["ego_speed_mps"]["min"] < stats["ego_speed_mps"]["max"] < 6.6
    assert stats["gt_tracklets_by_class"] == {"Car": 9, "Van": 3, "Pedestrian": 2, "Cyclist": 1}
    assert sum(sum(v) for v in stats["gt_objects_per_frame"].values()) == 717


@pytest.mark.parametrize("with_gt", [True, False])
def test_figure_panels_are_labeled(make_drive, params, with_gt):
    module = load_script()
    data_dir, drive = make_drive(n=3, points=8, tracklets_xml=TRACKLET_XML if with_gt else None)
    fig = module.build_figure(module.compute_stats(data_dir, drive, params))
    axes = fig.get_axes()
    assert len(axes) == 6
    for ax in axes:
        assert ax.get_title(), "every panel needs a title"
        if ax.axison:
            assert "(" in ax.get_xlabel() and "(" in ax.get_ylabel(), f"{ax.get_title()}: axis labels need units"
    texts = " ".join(t.get_text() for ax in axes for t in ax.texts)
    assert ("MISSING" in texts) != with_gt, "missing ground truth must be stated in text"


def test_missing_ground_truth_still_writes_stats(make_drive, tmp_path):
    data_dir, _ = make_drive(n=3, points=8)
    result = run(data_dir, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    assert "ground truth missing" in result.stderr
    stats = json.loads((tmp_path / "out" / "dataset_stats.json").read_text())
    assert stats["ground_truth"] == "missing" and stats["gt_objects_per_frame"] is None
    assert (tmp_path / "out" / "dataset_stats.png").stat().st_size > 0


def test_missing_data_fails_with_download_hint(tmp_path):
    result = run(tmp_path / "nothing", tmp_path / "out")
    assert result.returncode == 1
    assert "download_kitti.sh" in result.stderr
