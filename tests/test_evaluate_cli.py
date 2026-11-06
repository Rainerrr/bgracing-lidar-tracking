"""E4.7: scripts/evaluate.py failure paths and its output (synthetic drive, no real data needed)."""
import subprocess
import sys
from pathlib import Path

from conftest import TRACKER_CONFIG, TRACKLET_XML

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


def cli(tmp_path, *args, out=None, data_dir=None):
    argv = [sys.executable, str(SCRIPT), "--config", str(TRACKER_CONFIG), "--offline-dir", str(tmp_path / "offline"),
            "--out", str(out or tmp_path / "evaluation.md")]
    if data_dir is not None:
        argv += ["--data-dir", str(data_dir)]
    return subprocess.run(argv + list(args), capture_output=True, text=True)


def test_missing_tracklets_exit_1_naming_the_download_script_and_write_nothing(make_drive, tmp_path):
    data_dir, _ = make_drive(n=3, points=40)  # no tracklet_labels.xml
    r = cli(tmp_path, data_dir=data_dir)
    assert r.returncode == 1 and "download_kitti.sh" in r.stderr and "tracklet" in r.stderr
    assert not (tmp_path / "evaluation.md").exists() and not (tmp_path / "offline").exists()  # not even a partial run


def test_missing_drive_exits_1_with_the_download_hint(tmp_path):
    r = cli(tmp_path, data_dir=tmp_path / "nothing")
    assert r.returncode == 1 and "download_kitti.sh" in r.stderr and not (tmp_path / "evaluation.md").exists()


def test_unknown_set_key_exits_1_naming_it(make_drive, tmp_path):
    data_dir, _ = make_drive(n=3, points=40, tracklets_xml=TRACKLET_XML)
    r = cli(tmp_path, "--set", "nonsense=1", data_dir=data_dir)
    assert r.returncode == 1 and "nonsense" in r.stderr and not (tmp_path / "evaluation.md").exists()


def test_an_unwritable_output_path_exits_1_naming_it_before_running(make_drive, tmp_path):
    data_dir, _ = make_drive(n=3, points=40, tracklets_xml=TRACKLET_XML)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a folder should be")
    target = blocker / "evaluation.md"
    r = cli(tmp_path, out=target, data_dir=data_dir)
    assert r.returncode == 1 and str(target) in r.stderr
    assert not (tmp_path / "offline").exists()  # failed fast, before the two offline runs


def test_a_synthetic_drive_writes_a_complete_file_and_exits_0_even_though_the_gate_fails(make_drive, tmp_path):
    """Three frames, random points: nothing is confirmed, so the ablation cannot pass. The file is still complete
    (zero recall, n/a elsewhere), the exit code is 0 and the console says FAIL: verify, not the CLI, enforces the gate."""
    data_dir, _ = make_drive(n=3, points=40, tracklets_xml=TRACKLET_XML)
    r = cli(tmp_path, data_dir=data_dir)
    assert r.returncode == 0, r.stderr
    assert "run 1/2: ego compensation on" in r.stdout and "run 2/2: ego compensation off" in r.stdout
    assert "-> FAIL" in r.stdout
    text = (tmp_path / "evaluation.md").read_text()
    assert [ln for ln in text.splitlines() if ln.startswith("## ")][0] == "## 0. Summary" and "## 5. Method and reproducibility" in text
    assert "**Gate (D1): FAIL.**" in text and "n/a (" in text
    assert (tmp_path / "offline" / "tracks.csv").is_file() and (tmp_path / "offline" / "no_ego" / "tracks.csv").is_file()
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


def test_two_runs_write_the_same_bytes(make_drive, tmp_path):
    data_dir, _ = make_drive(n=3, points=40, tracklets_xml=TRACKLET_XML)
    assert cli(tmp_path, data_dir=data_dir, out=tmp_path / "a.md").returncode == 0
    assert cli(tmp_path, data_dir=data_dir, out=tmp_path / "b.md").returncode == 0
    assert (tmp_path / "a.md").read_bytes() == (tmp_path / "b.md").read_bytes()
