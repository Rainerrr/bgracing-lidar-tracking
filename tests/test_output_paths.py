"""Default output folders: the config file's drive keeps /output, any other drive gets /output/<drive>/, so running a
second drive never overwrites the first one's files. No data needed."""
import dataclasses

import pytest

from core import kitti
from processor import config
from scripts import evaluate, run_offline

DRIVE_A, DRIVE_B = "2011_09_26_drive_0005", "2011_09_26_drive_0011"


def test_the_config_drive_keeps_the_output_folder_and_another_drive_gets_its_own():
    assert str(config.output_root(DRIVE_A, DRIVE_A, "/output")) == "/output"
    assert str(config.output_root(DRIVE_B, DRIVE_A, "/output")) == "/output/2011_09_26_drive_0011"


@pytest.mark.parametrize("drive_name", ["../x", "2011_09_26_drive_0011/../..", "", None, "/abs"])
def test_a_malformed_drive_name_never_becomes_an_output_path(drive_name):
    with pytest.raises(kitti.DataError, match="YYYY_MM_DD_drive_NNNN"):
        config.output_root(drive_name, DRIVE_A, "/output")


@pytest.mark.parametrize("drive_name, ego, expected", [
    (DRIVE_A, True, "/output/offline"), (DRIVE_A, False, "/output/offline/no_ego"),
    (DRIVE_B, True, f"/output/{DRIVE_B}/offline"), (DRIVE_B, False, f"/output/{DRIVE_B}/offline/no_ego")])
def test_the_offline_runner_default_folder(params, drive_name, ego, expected):
    p = dataclasses.replace(params, drive=drive_name, ego_compensation=ego)
    assert run_offline.default_out_dir(p, DRIVE_A) == expected


def test_evaluate_on_another_drive_defaults_to_that_drives_folder(tmp_path, monkeypatch, capsys):
    """The data folder is empty, so the run stops at the missing drive, after it has made the output folder."""
    monkeypatch.setattr(config, "OUTPUT_ROOT", str(tmp_path / "output"))
    assert evaluate.main(["--set", f"drive={DRIVE_B}", "--data-dir", str(tmp_path / "data")]) == 1
    assert f"run scripts/download_kitti.sh {DRIVE_B}" in capsys.readouterr().err  # the exact command for this drive
    assert [p.name for p in (tmp_path / "output").iterdir()] == [DRIVE_B]


def test_evaluate_refuses_a_malformed_drive_before_making_any_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", str(tmp_path / "output"))
    assert evaluate.main(["--set", "drive=../x", "--data-dir", str(tmp_path / "data")]) == 1
    assert not (tmp_path / "output").exists()
