"""Report PNG (no data): renders empty and normal input, byte-stable, static vs moving by the tracker's state."""
import hashlib

from processor import report

ROWS = [f"0,{f},{f},1,{f * 0.5:.4f},0.0000,5.0000,0.0000,5.0000,moving,4.5,1.8,1.5" for f in range(10)] + \
       [f"0,{f},{f},2,10.0000,5.0000,0.9000,0.0000,0.9000,{'moving' if f == 0 else 'static'},4.5,1.8,1.5" for f in range(10)]


def test_report_is_byte_stable_and_counts_by_state(tmp_path, monkeypatch):
    seen = {}
    real = report.plt.subplots

    def spy(*a, **k):
        fig, ax = real(*a, **k)
        seen["ax"] = ax
        return fig, ax
    monkeypatch.setattr(report.plt, "subplots", spy)
    report.render_report(tmp_path / "a.png", ROWS, [(0, 0), (1, 1)])
    report.render_report(tmp_path / "b.png", ROWS, [(0, 0), (1, 1)])
    assert hashlib.sha256((tmp_path / "a.png").read_bytes()).digest() == hashlib.sha256((tmp_path / "b.png").read_bytes()).digest()
    # track 2 is 'moving' in 1 of 10 rows: drawn static
    assert "1 moving, 1 static" in seen["ax"].get_title()
    labels = [t.get_text() for t in seen["ax"].get_legend().get_texts()]
    assert any(l.startswith("static tracks (n = 1)") for l in labels) and any("moving track #1" in l for l in labels)


def test_report_with_no_tracks_says_so(tmp_path):
    report.render_report(tmp_path / "e.png", [], [(0, 0), (1, 1)])
    assert (tmp_path / "e.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_ground_truth_tags_and_the_footnote_appear_only_when_given(tmp_path, monkeypatch):
    seen = {}
    real = report.plt.subplots

    def spy(*a, **k):
        fig, ax = real(*a, **k)
        seen["fig"], seen["ax"] = fig, ax
        return fig, ax
    monkeypatch.setattr(report.plt, "subplots", spy)
    report.render_report(tmp_path / "t.png", ROWS, [(0, 0)], tags={(0, 1): "GT #0 Van", (0, 2): "GT #2 Pedestrian"})
    labels = [t.get_text() for t in seen["ax"].get_legend().get_texts()]
    assert labels[labels.index("ego path") + 1].startswith("moving track #1 = GT #0 Van")  # tagged moving first
    assert any(l.startswith("track #2 = GT #2 Pedestrian, drawn static") for l in labels)
    assert any("processor does not classify" in t.get_text() for t in seen["fig"].texts)
    report.render_report(tmp_path / "u.png", ROWS, [(0, 0)])
    assert not seen["fig"].texts and not any("GT #" in t.get_text() for t in seen["ax"].get_legend().get_texts())
