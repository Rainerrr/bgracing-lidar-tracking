"""Default output folders: the config file's drive keeps /output, any other drive gets /output/<drive>/, so running a
second drive never overwrites the first one's files. No data needed."""
import dataclasses

import pytest

from core import kitti
from processor import config
from scripts import evaluate, run_offline

D5, D11 = "2011_09_26_drive_0005", "2011_09_26_drive_0011"


def test_the_config_drive_keeps_the_output_folder_and_another_drive_gets_its_own():
    assert str(config.output_root(D5, D5, "/output")) == "/output"
    assert str(config.output_root(D11, D5, "/output")) == "/output/2011_09_26_drive_0011"


@pytest.mark.parametrize("drive_name", ["../x", "2011_09_26_drive_0011/../..", "", None, "/abs"])
def test_a_malformed_drive_name_never_becomes_an_output_path(drive_name):
    with pytest.raises(kitti.DataError, match="YYYY_MM_DD_drive_NNNN"):
        config.output_root(drive_name, D5, "/output")


@pytest.mark.parametrize("drive_name, ego, expected", [
    (D5, True, "/output/offline"), (D5, False, "/output/offline/no_ego"),
    (D11, True, f"/output/{D11}/offline"), (D11, False, f"/output/{D11}/offline/no_ego")])
def test_the_offline_runner_default_folder(params, drive_name, ego, expected):
    p = dataclasses.replace(params, drive=drive_name, ego_compensation=ego)
    assert run_offline.default_out_dir(p, D5) == expected


def test_evaluate_on_another_drive_defaults_to_that_drives_folder(tmp_path, monkeypatch, capsys):
    """The data folder is empty, so the run stops at the missing drive, after it has made the output folder."""
    monkeypatch.setattr(config, "OUTPUT_ROOT", str(tmp_path / "output"))
    assert evaluate.main(["--set", f"drive={D11}", "--data-dir", str(tmp_path / "data")]) == 1
    assert f"run scripts/download_kitti.sh {D11}" in capsys.readouterr().err  # the exact command for this drive
    assert [p.name for p in (tmp_path / "output").iterdir()] == [D11]


def test_evaluate_refuses_a_malformed_drive_before_making_any_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", str(tmp_path / "output"))
    assert evaluate.main(["--set", "drive=../x", "--data-dir", str(tmp_path / "data")]) == 1
    assert not (tmp_path / "output").exists()
