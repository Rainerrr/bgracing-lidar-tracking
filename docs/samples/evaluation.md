# Evaluation: 2011_09_26_drive_0005

Offline run over all 154 frames, once with ego compensation and once without, scored against the KITTI tracklets. Regenerate with `docker compose run --rm --no-deps processor python3 /app/scripts/evaluate.py`. Seed 42, association gate 2.0 m, warm-up 5 frames after a track's confirmation.

## 0. Summary

*How to read this:* The headline numbers, each pointing to its part. One result is gated: removing ego compensation must at least triple the velocity error of parked cars. Everything else is reported, with no pass/fail bar by design, because bars are not set after the results are known.

**Gate: PASS.** Static-object velocity RMSE is 0.653 m/s with ego compensation and 5.348 m/s without: a ratio of 8.19, against a bar of at least 3.0 (part 4).

| Result | Value | Judged by | Part |
|---|---|---|---|
| Static-object velocity RMSE, ego compensation off vs on | 8.19x | gated, bar 3.0x | 4 |
| Parked tracklets: median of per-tracklet median speed (m/s) | 0.379 | gated by the parked-speed test, bar 0.5; worst tracklet 0.660, bar 0.7 | 2 |
| Moving tracklets #0 Van, #1 Cyclist: median speed error (m/s) | 0.217 / 0.236 | gated by the moving-speed test, bar 0.5 | 2 |
| Car+Van recall (box-frames) | 0.910 (473 / 520) | no bar by design | 2 |
| Label coverage: share of in-region track-frames explained by a label | 0.112 (629 / 5624) | no bar by design; a lower bound | 2 |
| Static accuracy / parked flicker | 0.984 / 0.016 | no bar by design | 2 |
| Moving-label recall | 0.981 (253 / 258) | no bar by design | 2 |
| ID switches | 6 | no bar by design | 2 |
| Unlabelled tracks: median speed (m/s) / position spread (m) | 0.358 / 1.006 | no bar by design; parked-car reference 0.413 / 0.599 | 3 |
| False-moving rate | 0.086 (338 / 3925) | no bar by design | 3 |

## 1. What the pipeline produced

*How to read this:* What the tracker output over the drive, before anything judges it: detections and tracks per frame, how long tracks live, how many have a KITTI label and how many read as moving. Read this first to know the scale of everything below; none of it is a quality claim.

| Quantity | Value |
|---|---|
| Frames | 154 |
| Detections per frame, mean / max | 67.7 / 95 |
| Confirmed tracks over the drive | 417 |
| Confirmed tracks per frame, mean / max | 74.2 / 103 |
| Tracks with a frame in the evaluation region | 238 |
| ... of which matched to a KITTI tracklet in some frame (labelled) | 18 |
| ... of which never matched (unlabelled) | 220 |
| Track-frames labelled `moving` | 0.114 (1301 / 11433) |
| Tracks labelled `moving` in at least half of their frames | 55 |
| Track lifetime in frames, median / p90, all tracks | 17.0 / 66.4 |
| Track lifetime, labelled tracks | 37.0 / 128.1 |
| Track lifetime, unlabelled tracks | 20.5 / 82.0 |

## 2. Accuracy against KITTI labels

*How to read this:* KITTI labels only cars, vans, pedestrians and cyclists, and only in front of the car. These numbers judge the tracker on those objects alone, so they say nothing about poles, signs or walls. The share of tracks explained by a label is therefore a lower bound on how many tracks are real objects, not an error rate.

**By class** (GT box-frames in the evaluation region; recall, center error and velocity RMSE: no bar by design)

| Class | Tracklets | GT box-frames | Matched | Missed | Recall | Center error (m) | Velocity RMSE (m/s) | Note |
|---|---|---|---|---|---|---|---|---|
| Car | 9 | 248 | 241 | 7 | 0.972 | 0.493 (n = 241) | 0.749 (n = 209) |  |
| Cyclist | 1 | 154 | 152 | 2 | 0.987 | 0.365 (n = 152) | 0.912 (n = 147) | small sample |
| Pedestrian | 2 | 22 | 4 | 18 | 0.182 | 0.076 (n = 4) | n/a (scored frames = 0) | small sample |
| Van | 3 | 272 | 232 | 40 | 0.853 | 0.787 (n = 232) | 0.708 (n = 213) |  |
| **Overall** | 15 | 696 | 629 | 67 | 0.904 | 0.568 (n = 629) | static / moving below |  |

Car+Van recall: 0.910 (473 / 520) (the detection-only recall in detection_recall.json uses the same 520 box-frames).

| Metric | Value | Judged by |
|---|---|---|
| Label coverage (precision against labels): matched / in-region track-frames | 0.112 (629 / 5624) | no bar by design; a lower bound, 4995 track-frames have no label |
| Velocity RMSE, static GT (m/s) | 0.653 (n = 311) | no bar by design; the ablation (part 4) gates it |
| Velocity RMSE, moving GT (m/s) | 0.910 (n = 258) | no bar by design |
| Static accuracy: static GT frames labelled `static` | 0.984 (306 / 311) | no bar by design |
| Parked flicker: static GT frames labelled `moving` (= 1 - static accuracy) | 0.016 (5 / 311) | no bar by design |
| Moving-label recall: moving GT frames labelled `moving` | 0.981 (253 / 258) | no bar by design; walkers read `static` by design (confident motion rule) |
| ID switches, all tracklets | 6 | no bar by design |

**By tracklet** (frames after the warm-up: speed in m/s, |speed error| against the +-5-frame GT velocity, share labelled `moving`)

| Tracklet | Class | GT is | GT box-frames | Matched | Recall | ID switches | Scored frames | Median speed | Median abs. error | Share moving |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Van | moving | 137 | 108 | 0.788 | 0 | 103 | 4.656 | 0.217 | 1.000 |
| 1 | Cyclist | moving | 154 | 152 | 0.987 | 0 | 147 | 4.547 | 0.236 | 1.000 |
| 2 | Pedestrian | moving | 6 | 4 | 0.667 | 0 | 0 | n/a (scored frames = 0) | n/a (scored frames = 0) | n/a (scored frames = 0) |
| 3 | Van | static | 107 | 97 | 0.907 | 1 | 88 | 0.255 | 0.203 | 0.000 |
| 4 | Car | moving | 11 | 11 | 1.000 | 0 | 8 | 1.533 | 1.019 | 0.375 |
| 5 | Car | static | 36 | 36 | 1.000 | 0 | 34 | 0.503 | 0.287 | 0.088 |
| 6 | Car | static | 36 | 36 | 1.000 | 2 | 28 | 0.088 | 0.145 | 0.000 |
| 7 | Car | static | 33 | 33 | 1.000 | 0 | 33 | 0.224 | 0.139 | 0.000 |
| 8 | Van | static | 28 | 27 | 0.964 | 0 | 22 | 0.569 | 0.324 | 0.000 |
| 9 | Car | static | 34 | 34 | 1.000 | 0 | 34 | 0.113 | 0.105 | 0.000 |
| 10 | Car | static | 30 | 30 | 1.000 | 1 | 28 | 0.572 | 0.460 | 0.071 |
| 11 | Car | static | 24 | 21 | 0.875 | 0 | 16 | 0.536 | 0.325 | 0.000 |
| 12 | Pedestrian | moving | 16 | 0 | 0.000 | 0 | 0 | n/a (scored frames = 0) | n/a (scored frames = 0) | n/a (scored frames = 0) |
| 13 | Car | static | 16 | 16 | 1.000 | 2 | 9 | 0.660 | 0.593 | 0.000 |
| 14 | Car | static | 28 | 24 | 0.857 | 0 | 19 | 0.163 | 0.150 | 0.000 |

## 3. Physical consistency

*How to read this:* Most tracks have no KITTI label, which annotates only cars, vans, pedestrians and cyclists; many are probably poles, signs, trunks, bushes or walls, but nothing here classifies them. Without labels they can still be judged by physics: once ego motion is removed, a fixed object must stay still, like a labelled parked car. This shows consistency, not correctness: a still track could equally be a wall segment or leftover ground.

| Tracks | Count | Scored | Excluded (< 10 frames) | Median speed (m/s) | Median position spread (m) |
|---|---|---|---|---|---|
| Unlabelled (never matched) | 220 | 127 | 93 | 0.358 | 1.006 |
| Reference: matched to a parked tracklet | 14 | 12 | 2 | 0.413 | 0.599 |

Only tracks with at least 10 frames after the warm-up are scored; shorter ones are counted as excluded, not hidden. Speed is the per-track median; spread is the largest distance from the track's mean world position. The box center of a large object drifts more than that of a small one, so the two rows are not directly comparable. No bar by design.

| Metric | Value | Judged by |
|---|---|---|
| False-moving rate: unmatched in-region track-frames after the warm-up labelled `moving` | 0.086 (338 / 3925) | no bar by design; the baseline for future fixes |

This counts every in-region track-frame that matches no tracklet, including the unmatched frames of tracks that do match a tracklet in other frames (for example when a moving vehicle is briefly missed). It does not assume that unmatched objects are fixed.

## 4. Ablation: ego compensation off

*How to read this:* The same drive tracked in the sensor frame, without the ego pose. Every fixed object should then appear to move at about the ego speed (3.1 to 6.5 m/s), whether KITTI labels it or not. This is the project's main result: it shows that ego compensation, not luck, makes parked objects read still.

| Metric | Ego compensation on (world) | Ego compensation off (velodyne) | Judged by |
|---|---|---|---|
| Static-object velocity RMSE (m/s) | 0.653 (n = 311) | 5.348 (n = 277) | **gated** |
| Ratio off / on | 8.19x |  | **gated: bar 3.0x, PASS** |
| Moving-object velocity RMSE (m/s) | 0.910 (n = 258) | 4.832 (n = 258) | no bar by design |
| Parked flicker | 0.016 (5 / 311) | 1.000 (277 / 277) | no bar by design |
| Moving-label recall | 0.981 (253 / 258) | 0.612 (158 / 258) | no bar by design |
| False-moving rate | 0.086 (338 / 3925) | 0.978 (3818 / 3905) | no bar by design |
| Car+Van recall | 0.910 (473 / 520) | 0.921 (479 / 520) | no bar by design |
| ID switches | 6 | 15 | no bar by design |
| Unlabelled tracks: median speed (m/s) | 0.358 | 5.464 | no bar by design |
| Unlabelled tracks: median position spread (m) | 1.006 | 7.689 | no bar by design |
| Parked-car reference: median speed (m/s) | 0.413 | 5.487 | no bar by design |

The off run tracks in the velodyne frame, so its positions move with the car and its velocities are measured in the sensor axes; the ground-truth world velocity is rotated into those axes (rotation only, no ego-velocity subtraction). The two runs match different sets of frames (the counts n differ), so the ratio compares RMSE over different frames.

## 5. Method and reproducibility

*How to read this:* How every number was computed, and the one command that regenerates this file byte for byte on the same CPU architecture.

- **Matching:** per frame, one-to-one Hungarian assignment on 2D center distance, gate 2.0 m; ground truth of every class in the evaluation region against all confirmed tracks (the same matching as the tracking speed scores).
- **Evaluation region:** the processing ROI minus the ego box, front half only (x >= 0 in the velodyne frame). Ground truth is tested on its velodyne center; a track is mapped back to velodyne and tested on x and y (tracks.csv has no z).
- **Unit:** one ground-truth box-frame. Recall is matched / GT box-frames in the region; matched + missed equals that count.
- **Static:** a tracklet whose world net speed is below 0.5 m/s (the 10 parked tracklets); every other tracklet is moving.
- **Warm-up:** the first 5 frames after a track's confirmation are excluded from velocity RMSE, static accuracy, moving-label recall, the false-moving rate and world stillness. They are not excluded from recall, label coverage, center error, ID switches or lifetimes.
- **Velocity:** the track's filtered velocity against the world-frame GT velocity, a centered difference over +-5 frames; RMSE is the root of the mean squared vector difference.
- **ID switch:** per tracklet, a matched frame whose track ID differs from the previous matched track ID; frames without a match do not reset it.
- **Unlabelled track:** a track with at least one frame in the region that is never matched to a tracklet. World stillness scores tracks with at least 10 post-warm-up frames.
- **Zero denominators** print `n/a` with the name of the empty denominator, never 0.
- **Gate:** only the ablation ratio (at least 3.0) is a pass/fail bar, fixed before the first evaluation run.
- **Reproduce:** `docker compose run --rm --no-deps processor python3 /app/scripts/evaluate.py`. Two runs give byte-identical files on the same CPU architecture (verified on arm64 and on x86_64, the latter on a GitHub Actions runner).
