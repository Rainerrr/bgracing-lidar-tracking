"""The README keeps its outline, its links and images resolve, its commands are the real ones, and its
headline numbers are the ones in the committed samples. Needs the repo mounted read-only at REPO_DIR (verify.sh
does this); without it the tests fail, they never skip."""
import dataclasses
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(os.environ.get("REPO_DIR", "/repo"))
OUTLINE = ["Architecture", "Dataset", "Pipeline", "Results", "How to run", "Known gaps", "Credits"]


@pytest.fixture(scope="module")
def readme():
    path = REPO / "README.md"
    assert path.is_file(), f"{path} not found: run through scripts/verify.sh, which mounts the repo at /repo"
    return path.read_text(encoding="utf-8")


def test_outline_sections_in_order_and_troubleshooting(readme):
    headings = re.findall(r"^## (.+)$", re.sub(r"```.*?```", "", readme, flags=re.S), flags=re.M)
    assert headings == OUTLINE
    run = readme.split("## How to run", 1)[1].split("\n## ", 1)[0]
    assert "### Troubleshooting" in run and "### Expected outputs" in run


def test_no_work_in_progress_banner(readme):
    assert "work in progress" not in readme.lower()


def local_links(text):
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if not re.match(r"^(https?:|mailto:|#)", target):
            yield target.split("#", 1)[0]


@pytest.fixture(scope="module")
def tracked():
    listing = os.environ.get("TRACKED_LIST")
    assert listing and Path(listing).is_file(), "no file list: run through scripts/verify.sh, which writes it"
    return {ln for ln in Path(listing).read_text(encoding="utf-8").splitlines() if ln}


@pytest.mark.parametrize("doc", ["README.md", "docs/design.md"])
def test_links_and_images_resolve_and_avoid_local_only_folders(doc, tracked):
    path = REPO / doc
    assert path.is_file(), f"{path} not found"
    base = path.parent
    for target in local_links(path.read_text(encoding="utf-8")):
        assert (base / target).exists(), f"{doc}: broken link {target}"
        rel = os.path.relpath(os.path.normpath(base / target), REPO)
        assert rel in tracked or any(t.startswith(rel.rstrip("/") + "/") for t in tracked), \
            f"{doc}: links to {target}, which is not part of the repository"


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


# ---------------------------------------------------------------- the README quotes the samples

def _samples():
    d = REPO / "docs" / "samples"
    return {n: json.loads((d / n).read_text()) for n in
            ("tracking_speeds.json", "tracking_speeds_no_ego.json", "dataset_stats.json", "detection_recall.json")}


def test_intro_bullets_quote_the_samples(readme):
    s, ev = _samples(), (REPO / "docs" / "samples" / "evaluation.md").read_text(encoding="utf-8")
    top = readme.split("## Architecture", 1)[0]
    ratio = re.search(r"a ratio of ([\d.]+)", ev).group(1)
    objs = s["tracking_speeds.json"]["moving"]["objects"]
    assert f"Car+Van recall {s['detection_recall.json']['car_van']['recall']:.3f}." in top
    assert f"read a median {s['tracking_speeds.json']['parked']['median_of_medians']:.2f} m/s" in top
    assert f"within {objs['0']['median_abs_error']:.2f} and {objs['1']['median_abs_error']:.2f} m/s" in top
    assert f"{ratio} times larger" in top


def test_tracking_tables_are_rows_of_the_samples(readme):
    s = _samples()
    on, off = s["tracking_speeds.json"], s["tracking_speeds_no_ego.json"]
    pct = lambda v: f"{round(v * 100)} %"
    for tid, t in on["parked"]["tracklets"].items():
        row = (f"| #{tid} | {t['cls']} | {t['matched_frames']} | {t['median_speed']:.2f} | {pct(t['share_moving'])} | "
               f"{off['parked']['tracklets'][tid]['median_speed']:.2f} |")
        assert row in readme, f"parked row missing or stale in the README: {row}"
    assert f"**{on['parked']['median_of_medians']:.2f}** (bar {on['parked']['bar_median_of_medians']})" in readme
    for oid, o in on["moving"]["objects"].items():
        cls = o["cls"] + (" (small sample)" if o["small_sample"] else "")
        row = (f"| #{oid} | {cls} | {o['matched_frames']} | {o['median_gt_speed']:.2f} m/s | "
               f"{o['median_abs_error']:.2f}, {o['p90_abs_error']:.2f} m/s | {pct(o['share_moving'])} | "
               f"{off['moving']['objects'][oid]['median_abs_error']:.2f} m/s |")
        assert row in readme, f"moving row missing or stale in the README: {row}"


def test_dataset_table_matches_dataset_stats(readme):
    d = _samples()["dataset_stats.json"]
    pts, speed = d["points_per_frame"], d["ego_speed_mps"]
    period = d["sensor_period_s"]
    assert f"| Sweeps | {d['frames']} over {d['duration_s']:.2f} s |" in readme
    assert f"{d['mean_rate_hz']:.2f} Hz mean (period {period['min'] * 1000:.1f}–{period['max'] * 1000:.1f} ms)" in readme
    assert f"mean {pts['mean']:,.0f}; min {pts['min']:,}; max {pts['max']:,}" in readme
    assert f"{speed['min']:.1f}–{speed['max']:.1f} m/s, mean {speed['mean']:.1f} m/s" in readme
    by = d["gt_tracklets_by_class"]
    assert f"{sum(by.values())} tracklets ({by['Car']} Car, {by['Van']} Van, {by['Pedestrian']} Pedestrian, {by['Cyclist']} Cyclist), {d['gt_poses_total']} boxes" in readme


def test_layout_parses_and_anchors_resolve(readme):
    layout = json.loads((REPO / "viz" / "foxglove_layout.json").read_text())
    assert "configById" in layout and "layout" in layout
    text = re.sub(r"```.*?```", "", readme, flags=re.S)  # headings inside code fences are not headings
    slugs = {re.sub(r"[^a-z0-9 -]", "", h.lower()).strip().replace(" ", "-") for h in re.findall(r"^#{1,6} (.+)$", text, flags=re.M)}
    for anchor in re.findall(r"\]\(#([^)\s]+)\)", text):
        assert anchor in slugs, f"README anchor #{anchor} has no heading"


# ---------------------------------------------------------------- the second drive section

SECOND_DRIVE = "2011_09_26_drive_0011"


def _d11():
    d = REPO / "docs" / "samples" / SECOND_DRIVE
    return {n: json.loads((d / n).read_text()) for n in
            ("tracking_speeds.json", "tracking_speeds_no_ego.json", "dataset_stats.json", "detection_recall.json")}, \
        (d / "evaluation.md").read_text(encoding="utf-8")


def _drive_section(readme):
    return readme.split("### Second drive (KITTI drive 0011)", 1)[1].split("\n## ", 1)[0]


def test_second_drive_statistics_match_its_dataset_stats(readme):
    d = _d11()[0]["dataset_stats.json"]
    pts, speed, period, by = d["points_per_frame"], d["ego_speed_mps"], d["sensor_period_s"], d["gt_tracklets_by_class"]
    assert f"| Sweeps | {d['frames']} over {d['duration_s']:.2f} s |" in readme
    assert f"{d['mean_rate_hz']:.2f} Hz mean (period {period['min'] * 1000:.1f}–{period['max'] * 1000:.1f} ms)" in readme
    assert f"mean {pts['mean']:,.0f}; min {pts['min']:,}; max {pts['max']:,}" in readme
    assert f"{speed['min']:.1f}–{speed['max']:.1f} m/s, mean {speed['mean']:.1f} m/s" in readme
    assert f"{sum(by.values())} tracklets" in readme and f"{d['gt_poses_total']:,} boxes" in readme
    still = sum(v < 0.1 for v in speed["series"])
    assert f"below 0.1 m/s in the last {still} sweeps" in readme
    assert all(v < 0.1 for v in speed["series"][-still:])  # the stop is one stretch at the end


def test_second_drive_paragraph_quotes_the_samples_and_matches_their_verdicts(readme):
    s, ev = _d11()
    section = _drive_section(readme)
    cv = s["detection_recall.json"]["car_van"]
    assert f"{cv['matched'] / cv['total']:.3f} ({cv['matched']} / {cv['total']} box-frames)" in section
    ratio = re.search(r"a ratio of ([\d.]+), against a bar", ev).group(1)
    assert f"{ratio} times larger" in section
    assert "| " not in section  # a paragraph, not a table
    # the text says the two speed tests missed their bars: it must stay true of the committed sample
    assert s["tracking_speeds.json"]["parked"]["pass"] is False and s["tracking_speeds.json"]["moving"]["pass"] is False
    assert "did not meet their bars" in section and "further tuning" in section


def test_second_drive_samples_use_the_first_drives_parameters():
    from processor import config
    base = config.load_params(os.environ.get("TRACKER_CONFIG", "/config/tracker.yaml"))
    for name in ("tracking_speeds.json", "tracking_speeds_no_ego.json"):
        got = _d11()[0][name]["params"]
        want = dataclasses.asdict(base)
        want["drive"] = SECOND_DRIVE
        if name.endswith("no_ego.json"):
            want["ego_compensation"] = False
        assert got == json.loads(json.dumps(want)), f"{name}: parameters differ from config/tracker.yaml"


def test_the_parameter_check_catches_a_changed_value():
    from processor import config
    base = config.load_params(os.environ.get("TRACKER_CONFIG", "/config/tracker.yaml"))
    got = _d11()[0]["tracking_speeds.json"]["params"]
    changed = json.loads(json.dumps(dataclasses.asdict(dataclasses.replace(base, dbscan_eps=base.dbscan_eps + 0.1))))
    changed["drive"] = SECOND_DRIVE
    assert got != changed


def test_second_drive_says_it_is_not_an_untouched_test_set(readme):
    flat = " ".join(_drive_section(readme).split())
    assert "no further tuning" in flat and "not an untouched test set" in flat
    for word in ("held-out", "held out", "dev drive", "phase"):
        assert word not in readme.lower(), word


def test_the_readme_names_its_author_at_the_top(readme):
    assert readme.split("\n", 4)[2] == "By Rene Bitton"


def test_known_gaps_lists_five_numbered_items_at_the_bottom(readme):
    gaps = readme.split("## Known gaps", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r"^(\d)\. \*\*", gaps, flags=re.M) == ["1", "2", "3", "4", "5"]
    for topic in ("`moving`", "Buses and long trucks", "second drive", "classes", "Thin evidence"):
        assert topic in gaps
