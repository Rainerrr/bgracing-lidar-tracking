"""E5.7: the README keeps the spec's outline, its links and images resolve, its commands are the real ones, and its
headline numbers are the ones in the committed samples. Needs the repo mounted read-only at REPO_DIR (verify.sh
does this); without it the tests fail, they never skip."""
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(os.environ.get("REPO_DIR", "/repo"))
OUTLINE = ["Architecture", "Dataset", "Pipeline", "Results", "How to run", "Limitations and next steps", "Credits"]


@pytest.fixture(scope="module")
def readme():
    path = REPO / "README.md"
    assert path.is_file(), f"{path} not found: run through scripts/verify.sh, which mounts the repo at /repo"
    return path.read_text(encoding="utf-8")


def test_outline_sections_in_order_and_troubleshooting(readme):
    headings = re.findall(r"^## (.+)$", readme, flags=re.M)
    assert headings == OUTLINE
    run = readme.split("## How to run", 1)[1].split("\n## ", 1)[0]
    assert "### Troubleshooting" in run and "### Expected outputs" in run


def test_no_work_in_progress_banner(readme):
    assert "work in progress" not in readme.lower()


def local_links(text):
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if not re.match(r"^(https?:|mailto:|#)", target):
            yield target.split("#", 1)[0]


@pytest.mark.parametrize("doc", ["README.md", "docs/spec.md"])
def test_links_and_images_resolve_and_avoid_local_only_folders(doc):
    path = REPO / doc
    assert path.is_file(), f"{path} not found"
    base = path.parent
    for target in local_links(path.read_text(encoding="utf-8")):
        assert (base / target).exists(), f"{doc}: broken link {target}"
        assert "docs/specs" not in target, f"{doc}: links into the git-ignored docs/specs: {target}"


def test_documented_commands_are_the_real_ones(readme):
    blocks = "\n".join(re.findall(r"```bash\n(.*?)```", readme, flags=re.S))
    for cmd in ("./scripts/download_kitti.sh", "docker compose up --build", "/app/scripts/dataset_stats.py",
                "/app/scripts/run_offline.py", "/app/scripts/evaluate.py", "./scripts/verify.sh"):
        assert cmd in blocks, f"README has no command block with {cmd}"
    for script in re.findall(r"/app/(scripts/\w+\.py)", blocks):
        assert (REPO / script).is_file(), f"README runs {script}, which does not exist"
    assert "ws://localhost:8765" in readme and (REPO / "viz" / "foxglove_layout.json").is_file()


def test_headline_numbers_match_the_samples(readme):
    samples = REPO / "docs" / "samples"
    ev = (samples / "evaluation.md").read_text(encoding="utf-8")
    gate = re.search(r"is ([\d.]+) m/s with ego compensation and ([\d.]+) m/s without: a ratio of ([\d.]+)", ev)
    assert gate, "gate sentence not found in docs/samples/evaluation.md"
    on, off, ratio = gate.groups()
    # the exact sentence, so a stale number elsewhere in the README cannot satisfy the check
    assert f"is **{on} m/s** with ego compensation and **{off} m/s** without it: a ratio of **{ratio}**" in readme
    car_van = json.loads((samples / "detection_recall.json").read_text())["car_van"]
    assert f"{car_van['matched']} / {car_van['total']}" in readme
    assert f"{car_van['recall']:.3f}" in readme
    cv = re.search(r"Car\+Van recall \(box-frames\) \| ([\d.]+) \((\d+) / (\d+)\)", ev)
    assert cv and f"{cv.group(2)} of {cv.group(3)}" in readme and cv.group(1) in readme
