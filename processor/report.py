"""obstacle_tracking_report.png: top-down map of the tracks of one pass. ROS-free (matplotlib, Agg).

Static vs moving is carried by line style and legend text, never by color alone:
- static tracks: thin, dashed, grey, under ONE legend entry with their count (clutter makes dozens of them);
- moving tracks: solid, named by ID in the legend, labelled at their last position, with a velocity arrow.
A track is drawn as moving if the tracker labelled it "moving" (the confident motion rule) in at
least half of its rows, so a parked car labelled moving in a few frames is still drawn as static.
Ground-truth tags (offline only): tracks matched to a KITTI tracklet that really moves are tagged "GT #n Class"
on the map and in the legend, listed first, and a footnote says the labels come from the ground truth: the
processor classifies nothing. A tagged track the tracker labels static (a walker, by the confident rule) is drawn black and dash-dotted.
Byte-stable for the same input on one architecture: no PNG metadata, fixed size and DPI.
"""
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from processor.track_scoring import parse_track_rows  # noqa: E402

ARROW_SECONDS = 0.5  # arrow length = speed x 0.5 s, as the live arrows


FOOTNOTE = ("GT tags: each track is tagged with the KITTI ground-truth tracklet it matches in most of its matched "
            "sweeps (Hungarian, 2.0 m gate, front half), shown only for tracklets that really move (net speed >= 0.5 m/s). "
            "The class names come from the ground truth; the processor does not classify objects.")


def render_report(path, track_rows, ego_xy, tracking_frame="world", title_extra="", tags=None):
    """Write the PNG. track_rows: tracks.csv data lines; the last pass present is drawn.
    tags: {(pass, track_id): "GT #n Class"} from track_scoring.gt_tags (offline), or None (live: no tags)."""
    rows = parse_track_rows(track_rows)
    last_pass = max((r["pass"] for r in rows), default=0)
    tracks = defaultdict(list)
    for r in rows:
        if r["pass"] == last_pass:
            tracks[r["track_id"]].append(r)
    tag = {tid: t for (p, tid), t in (tags or {}).items() if p == last_pass and tid in tracks}
    drawn_moving = {tid for tid, rs in tracks.items() if 2 * sum(r["state"] == "moving" for r in rs) >= len(rs)}
    moving = sorted(drawn_moving, key=lambda tid: (tid not in tag, tid))  # tagged first in the legend
    tagged_static = sorted(tid for tid in tag if tid not in drawn_moving)
    static = sorted(set(tracks) - drawn_moving - set(tagged_static))

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    for i, tid in enumerate(static):
        rs = tracks[tid]
        ax.plot([r["x"] for r in rs], [r["y"] for r in rs], color="0.6", lw=0.8, ls="--",
                label=f"static tracks (n = {len(static)}), includes unlabelled objects such as poles and bushes" if i == 0 else None)
    ego = np.asarray(ego_xy, dtype=float).reshape(-1, 2)
    if len(ego):
        ax.plot(ego[:, 0], ego[:, 1], color="black", lw=2.5, label="ego path")
        ax.annotate("ego start", ego[0], textcoords="offset points", xytext=(6, -12), fontsize=8)
    cmap = plt.get_cmap("tab10")
    for i, tid in enumerate(moving):
        rs = tracks[tid]
        color = cmap(i % 10)
        med = float(np.median([r["speed"] for r in rs]))
        name = f"moving track #{tid} = {tag[tid]}" if tid in tag else f"moving track #{tid}"
        ax.plot([r["x"] for r in rs], [r["y"] for r in rs], color=color, lw=3.0 if tid in tag else 2.0,
                label=f"{name}: median {med:.1f} m/s ({med * 3.6:.0f} km/h)")
        last = rs[-1]
        ax.annotate("", xy=(last["x"] + last["vx"] * ARROW_SECONDS, last["y"] + last["vy"] * ARROW_SECONDS),
                    xytext=(last["x"], last["y"]), arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0})
        _label(ax, tid, last, tag.get(tid))
    for tid in tagged_static:
        rs = tracks[tid]
        med = float(np.median([r["speed"] for r in rs]))
        share = sum(r["state"] == "moving" for r in rs) / len(rs)
        ax.plot([r["x"] for r in rs], [r["y"] for r in rs], color="black", lw=1.5, ls="-.",
                label=f"track #{tid} = {tag[tid]}, drawn static (moving in {share:.0%} of sweeps): median {med:.1f} m/s")
        _label(ax, tid, rs[-1], tag[tid])
    if not tracks:
        ax.text(0.5, 0.5, "no confirmed tracks", transform=ax.transAxes, ha="center", va="center")
    xl, yl = ("east (m)", "north (m)") if tracking_frame == "world" else ("x velodyne (m)", "y velodyne (m)")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.3)
    ax.set_title(f"Confirmed tracks, {tracking_frame} frame, pass {last_pass}: {len(moving)} moving, "
                 f"{len(static)} static{title_extra}", fontsize=10)
    n_entries = len(moving) + len(tagged_static) + (1 if static else 0) + (1 if len(ego) else 0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5, framealpha=0.9,
              ncol=1 if n_entries <= 30 else 2, borderaxespad=0.0)
    if tags is not None:
        fig.text(0.01, 0.01, FOOTNOTE, fontsize=8, ha="left", va="bottom", wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=100, metadata={"Software": None})
    plt.close(fig)


def _label(ax, tid, row, tag):
    if tag:
        ax.annotate(f"#{tid} = {tag}", (row["x"], row["y"]), textcoords="offset points", xytext=(6, 6), fontsize=8,
                    fontweight="bold", color="black", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "lw": 0.6})
    else:
        ax.annotate(f"#{tid}", (row["x"], row["y"]), textcoords="offset points", xytext=(5, 4), fontsize=7, color="black")
