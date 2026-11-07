"""Isolation: the shared core stays ROS-free and numpy-only, so both images can import it."""
import subprocess
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[1] / "core"


def test_core_imports_no_ros_scipy_or_matplotlib():
    code = (
        "import sys, importlib, pathlib, core\n"
        "[importlib.import_module('core.' + f.stem) for f in pathlib.Path(core.__file__).parent.glob('*.py') if f.stem != '__init__']\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('rclpy', 'rosidl_runtime_py', 'scipy', 'sklearn', 'matplotlib'))\n"
        "print(core.__file__)\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.splitlines()
    assert Path(out[0]).resolve().parent == CORE_DIR, f"core imported from {out[0]}, expected {CORE_DIR}"
    assert out[1] == "", f"core pulled in {out[1]}; the publisher image has no scipy/matplotlib and core must stay ROS-free"



def test_processor_core_modules_import_no_rclpy():
    code = (
        "import sys, processor.config, processor.perception, processor.matching, processor.records, processor.tracking, processor.track_scoring\n"
        "print(','.join(sorted(m for m in sys.modules if m.split('.')[0] in ('rclpy', 'rosidl_runtime_py'))))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"ROS-free processor modules pulled in {out}"
