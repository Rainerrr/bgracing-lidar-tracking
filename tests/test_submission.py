"""Submission boundary: the repository holds only the submission, and no file cites internal working notes.

scripts/verify.sh writes the list of tracked files (git ls-files, or the file tree when the project was downloaded
without git) and passes its path in TRACKED_LIST, with the repo mounted read-only at REPO_DIR. Without them the
tests fail, they never skip."""
import os
import re
from pathlib import Path

import pytest

REPO = Path(os.environ.get("REPO_DIR", "/repo"))

# Everything a reviewer needs: the code, the two images, the README and its evidence, CI, the Foxglove layout.
ALLOWED_FILES = {".github/workflows/verify.yml", ".gitignore", ".dockerignore", "LICENSE", "README.md",
                 "docker-compose.yml", "docs/design.md", "viz/foxglove_layout.json", "data/.gitkeep", "output/.gitkeep"}
ALLOWED_DIRS = ("config/", "core/", "processor/", "publisher/", "scripts/", "tests/", "docs/samples/")
REQUIRED = ("README.md", "LICENSE", "docker-compose.yml", "docs/design.md",
            "publisher/Dockerfile.publisher", "processor/Dockerfile.processor")

# References to working notes that are not part of the submission: criterion and decision numbers, phase numbers,
# and the notes' own names. Joined at runtime so this file does not match itself.
_INTERNAL = [
    r"(?<![\w.])[Ee][1-5]\.[0-9]+",          # criterion numbers: a letter E, then 3.2
    r"(?<![A-Za-z0-9])[Ee][1-5]_[0-9](?![0-9])",  # the same inside a name: a letter E, then 1_2
    r"\b" + "D" + r"[0-9]{1,2}\b",             # decision numbers: a letter D, then 10
    r"(?<![A-Za-z])[Pp]" + r"hase[ _-][0-9]",   # phase numbers, also in test names and file names
    "FEATURE" + "_SPEC",
    "docs/" + "specs",
    "docs/" + r"spec\.md",
]
INTERNAL = re.compile("|".join(f"(?:{p})" for p in _INTERNAL))
BINARY = (".png", ".gif")


@pytest.fixture(scope="module")
def tracked():
    listing = os.environ.get("TRACKED_LIST")
    assert listing and Path(listing).is_file(), "no file list: run through scripts/verify.sh, which writes it"
    paths = [ln for ln in Path(listing).read_text(encoding="utf-8").splitlines() if ln]
    assert paths, "the file list is empty"
    return paths


def outside_submission(paths):
    return [p for p in paths if p not in ALLOWED_FILES and not p.startswith(ALLOWED_DIRS)]


def internal_references(path, text):
    return [f"{path}:{i}: {m.group(0)}" for i, line in enumerate(text.splitlines(), 1)
            for m in INTERNAL.finditer(line)]


def test_every_file_belongs_to_the_submission(tracked):
    extra = outside_submission(tracked)
    assert not extra, "files outside the submission: " + ", ".join(extra)


def test_the_required_files_are_present(tracked):
    missing = [p for p in REQUIRED if p not in tracked or not (REPO / p).is_file()]
    assert not missing, "missing: " + ", ".join(missing)


def test_no_file_cites_internal_notes(tracked):
    hits = []
    for p in tracked:
        if p.endswith(BINARY):
            continue
        hits += internal_references(p, (REPO / p).read_text(encoding="utf-8"))
    assert not hits, f"{len(hits)} internal references:\n" + "\n".join(hits[:60])


# ---------------------------------------------------------------- the checks can fail

def test_control_paths_outside_the_submission_are_caught():
    assert outside_submission(["README.md", "core/kitti.py", "notes/plan.md", ".claude/agents/x.md", "docs/other.md"]) \
        == ["notes/plan.md", ".claude/agents/x.md", "docs/other.md"]


@pytest.mark.parametrize("line", ["bar from E" + "3.2", "(" + "D" + "10)", "the P" + "hase 3 run", "test_the_p" + "hase_3_x", "test_x_e" + "1_2_fixture",
                                  "see " + "FEATURE" + "_SPEC", "docs/" + "specs/x", "docs/" + "spec.md"],
                         ids=["criterion", "decision", "phase", "phase-in-a-name", "criterion-in-a-name", "spec-name", "notes-folder", "spec-file"])
def test_control_each_internal_reference_is_caught(line):
    assert internal_references("f.py", line), line


@pytest.mark.parametrize("line", ["2D DBSCAN", "1e3.5", "3.2 m", "v1.2.3", "ID 10", "dx/dy/dz", "phases of the drive", "1e3_5", "size_e1_23"])
def test_control_ordinary_text_is_not_flagged(line):
    assert not internal_references("f.py", line), line
