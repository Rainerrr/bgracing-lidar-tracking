"""The processor launch file passes KITTI_DRIVE to the tracker node, so compose can pick the drive for both
containers. No ROS graph is started."""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("REPO_DIR", "/repo"))  # the repo, mounted read-only by verify.sh (the image has no compose file)


def launch_module():
    spec = importlib.util.spec_from_file_location("processor_launch", ROOT / "processor" / "processor.launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_without_kitti_drive_the_yaml_drive_is_used():
    cmd = launch_module().tracker_cmd({})
    assert cmd[-2:] == ["--params-file", "/config/tracker.yaml"] and not any(a.startswith("drive:=") for a in cmd)


def test_kitti_drive_overrides_the_yaml_drive_after_the_params_file():
    cmd = launch_module().tracker_cmd({"KITTI_DRIVE": "2011_09_26_drive_0011"})
    assert cmd.index("drive:=2011_09_26_drive_0011") > cmd.index("--params-file")  # a later -p wins over the file
    assert cmd[-2:] == ["-p", "drive:=2011_09_26_drive_0011"]


def test_compose_gives_both_services_the_same_drive_variable():
    text = (REPO / "docker-compose.yml").read_text()
    assert text.count("KITTI_DRIVE=${KITTI_DRIVE:-2011_09_26_drive_0005}") == 2
    assert '"drive:=${KITTI_DRIVE:-2011_09_26_drive_0005}"' in text
