# Evaluation: 2011_09_26_drive_0011

Offline run over all 233 frames, once with ego compensation and once without, scored against the KITTI tracklets. Regenerate with `docker compose run --rm --no-deps processor python3 /app/scripts/evaluate.py --set drive=2011_09_26_drive_0011`. Seed 42, association gate 2.0 m, warm-up 5 frames after a track's confirmation.

## 0. Summary

*How to read this:* The headline numbers, each pointing to its part. One result is gated: removing ego compensation must at least triple the velocity error of parked cars. Everything else is reported, with no pass/fail bar by design, because bars are not set after the results are known.

**Gate: PASS.** Static-object velocity RMSE is 0.336 m/s with ego compensation and 5.214 m/s without: a ratio of 15.52, against a bar of at least 3.0 (part 4).

| Result | Value | Judged by | Part |
|---|---|---|---|
| Static-object velocity RMSE, ego compensation off vs on | 15.52x | gated, bar 3.0x | 4 |
| Parked tracklets: median of per-tracklet median speed (m/s) | n/a (2 of 7 parked tracklets not scored) | gated by the parked-speed test, bar 0.5; worst tracklet n/a (2 of 7 parked tracklets not scored), bar 0.7 | 2 |
| Moving tracklets #2 Car, #5 Car, #6 Car, #7 Van, #8 Truck, #9 Car, #15 Car: median speed error (m/s) | 1.601 / 1.128 / 0.958 / 1.050 / 2.077 / 0.185 / 1.456 | gated by the moving-speed test, bar 0.5 | 2 |
| Car+Van recall (box-frames) | 0.852 (306 / 359) | no bar by design | 2 |
| Label coverage: share of in-region track-frames explained by a label | 0.109 (588 / 5378) | no bar by design; a lower bound | 2 |
| Static accuracy / parked flicker | 1.000 / 0.000 | no bar by design | 2 |
| Moving-label recall | 0.863 (183 / 212) | no bar by design | 2 |
| ID switches | 13 | no bar by design | 2 |
| Unlabelled tracks: median speed (m/s) / position spread (m) | 0.219 / 0.540 | no bar by design; parked-car reference 0.247 / 1.110 | 3 |
| False-moving rate | 0.081 (320 / 3959) | no bar by design | 3 |

## 1. What the pipeline produced

*How to read this:* What the tracker output over the drive, before anything judges it: detections and tracks per frame, how long tracks live, how many have a KITTI label and how many read as moving. Read this first to know the scale of everything below; none of it is a quality claim.

| Quantity | Value |
|---|---|
| Frames | 233 |
| Detections per frame, mean / max | 45.0 / 60 |
| Confirmed tracks over the drive | 368 |
| Confirmed tracks per frame, mean / max | 48.0 / 60 |
| Tracks with a frame in the evaluation region | 190 |
| ... of which matched to a KITTI tracklet in some frame (labelled) | 28 |
| ... of which never matched (unlabelled) | 162 |
| Track-frames labelled `moving` | 0.134 (1501 / 11179) |
| Tracks labelled `moving` in at least half of their frames | 74 |
| Track lifetime in frames, median / p90, all tracks | 15.5 / 76.9 |
| Track lifetime, labelled tracks | 41.5 / 138.6 |
| Track lifetime, unlabelled tracks | 23.0 / 83.7 |

## 2. Accuracy against KITTI labels

*How to read this:* KITTI labels only cars, vans, pedestrians and cyclists, and only in front of the car. These numbers judge the tracker on those objects alone, so they say nothing about poles, signs or walls. The share of tracks explained by a label is therefore a lower bound on how many tracks are real objects, not an error rate.

**By class** (GT box-frames in the evaluation region; recall, center error and velocity RMSE: no bar by design)

| Class | Tracklets | GT box-frames | Matched | Missed | Recall | Center error (m) | Velocity RMSE (m/s) | Note |
|---|---|---|---|---|---|---|---|---|
| Car | 11 | 305 | 253 | 52 | 0.830 | 0.699 (n = 253) | 1.694 (n = 191) |  |
| Cyclist | 1 | 75 | 75 | 0 | 1.000 | 0.183 (n = 75) | 0.209 (n = 71) | small sample |
| Misc | 1 | 16 | 14 | 2 | 0.875 | 0.581 (n = 14) | 1.327 (n = 4) | small sample |
| Pedestrian | 1 | 156 | 154 | 2 | 0.987 | 0.322 (n = 154) | 0.157 (n = 149) | small sample |
| Truck | 1 | 72 | 39 | 33 | 0.542 | 1.068 (n = 39) | 2.775 (n = 30) | small sample |
| Van | 1 | 54 | 53 | 1 | 0.981 | 0.717 (n = 53) | 2.712 (n = 46) | small sample |
| **Overall** | 16 | 678 | 588 | 90 | 0.867 | 0.558 (n = 588) | static / moving below |  |

Car+Van recall: 0.852 (306 / 359) (the detection-only recall in detection_recall.json uses the same 359 box-frames).

| Metric | Value | Judged by |
|---|---|---|
| Label coverage (precision against labels): matched / in-region track-frames | 0.109 (588 / 5378) | no bar by design; a lower bound, 4790 track-frames have no label |
| Velocity RMSE, static GT (m/s) | 0.336 (n = 279) | no bar by design; the ablation (part 4) gates it |
| Velocity RMSE, moving GT (m/s) | 2.277 (n = 212) | no bar by design |
| Static accuracy: static GT frames labelled `static` | 1.000 (279 / 279) | no bar by design |
| Parked flicker: static GT frames labelled `moving` (= 1 - static accuracy) | 0.000 (0 / 279) | no bar by design |
| Moving-label recall: moving GT frames labelled `moving` | 0.863 (183 / 212) | no bar by design; walkers read `static` by design (confident motion rule) |
| ID switches, all tracklets | 13 | no bar by design |

**By tracklet** (frames after the warm-up: speed in m/s, |speed error| against the +-5-frame GT velocity, share labelled `moving`)

| Tracklet | Class | GT is | GT box-frames | Matched | Recall | ID switches | Scored frames | Median speed | Median abs. error | Share moving |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Car | moving | 17 | 15 | 0.882 | 0 | 10 | 7.084 | 1.041 | 1.000 |
| 1 | Car | static | 57 | 57 | 1.000 | 0 | 52 | 0.302 | 0.217 | 0.000 |
| 2 | Car | moving | 26 | 23 | 0.885 | 1 | 13 | 7.901 | 1.601 | 1.000 |
| 3 | Pedestrian | static | 156 | 154 | 0.987 | 0 | 149 | 0.058 | 0.029 | 0.000 |
| 4 | Cyclist | static | 75 | 75 | 1.000 | 0 | 71 | 0.101 | 0.050 | 0.000 |
| 5 | Car | moving | 38 | 37 | 0.974 | 0 | 32 | 7.162 | 1.128 | 1.000 |
| 6 | Car | moving | 57 | 57 | 1.000 | 3 | 41 | 6.395 | 0.958 | 0.976 |
| 7 | Van | moving | 54 | 53 | 0.981 | 1 | 46 | 6.605 | 1.050 | 1.000 |
| 8 | Truck | moving | 72 | 39 | 0.542 | 4 | 30 | 3.934 | 2.077 | 1.000 |
| 9 | Car | moving | 26 | 26 | 1.000 | 0 | 26 | 0.352 | 0.185 | 0.000 |
| 10 | Car | static | 3 | 1 | 0.333 | 0 | 0 | n/a (scored frames = 0) | n/a (scored frames = 0) | n/a (scored frames = 0) |
| 11 | Car | static | 8 | 5 | 0.625 | 1 | 0 | n/a (scored frames = 0) | n/a (scored frames = 0) | n/a (scored frames = 0) |
| 12 | Car | static | 11 | 8 | 0.727 | 0 | 3 | 1.019 | 0.762 | 0.000 |
| 13 | Misc | static | 16 | 14 | 0.875 | 1 | 4 | 1.286 | 1.074 | 0.000 |
| 14 | Car | moving | 15 | 0 | 0.000 | 0 | 0 | n/a (scored frames = 0) | n/a (scored frames = 0) | n/a (scored frames = 0) |
| 15 | Car | moving | 47 | 24 | 0.511 | 2 | 14 | 2.837 | 1.456 | 0.857 |

## 3. Physical consistency

*How to read this:* Most tracks have no KITTI label, which annotates only cars, vans, pedestrians and cyclists; many are probably poles, signs, trunks, bushes or walls, but nothing here classifies them. Without labels they can still be judged by physics: once ego motion is removed, a fixed object must stay still, like a labelled parked car. This shows consistency, not correctness: a still track could equally be a wall segment or leftover ground.

| Tracks | Count | Scored | Excluded (< 10 frames) | Median speed (m/s) | Median position spread (m) |
|---|---|---|---|---|---|
| Unlabelled (never matched) | 162 | 106 | 56 | 0.219 | 0.540 |
| Reference: matched to a parked tracklet | 9 | 7 | 2 | 0.247 | 1.110 |

Only tracks with at least 10 frames after the warm-up are scored; shorter ones are counted as excluded, not hidden. Speed is the per-track median; spread is the largest distance from the track's mean world position. The box center of a large object drifts more than that of a small one, so the two rows are not directly comparable. No bar by design.

| Metric | Value | Judged by |
|---|---|---|
| False-moving rate: unmatched in-region track-frames after the warm-up labelled `moving` | 0.081 (320 / 3959) | no bar by design; the baseline for future fixes |

This counts every in-region track-frame that matches no tracklet, including the unmatched frames of tracks that do match a tracklet in other frames (for example when a moving vehicle is briefly missed). It does not assume that unmatched objects are fixed.

## 4. Ablation: ego compensation off

*How to read this:* The same drive tracked in the sensor frame, without the ego pose. Every fixed object should then appear to move at about the ego speed (0.0 to 11.9 m/s), whether KITTI labels it or not. This is the project's main result: it shows that ego compensation, not luck, makes parked objects read still.

| Metric | Ego compensation on (world) | Ego compensation off (velodyne) | Judged by |
|---|---|---|---|
| Static-object velocity RMSE (m/s) | 0.336 (n = 279) | 5.214 (n = 282) | **gated** |
| Ratio off / on | 15.52x |  | **gated: bar 3.0x, PASS** |
| Moving-object velocity RMSE (m/s) | 2.277 (n = 212) | 2.943 (n = 216) | no bar by design |
| Parked flicker | 0.000 (0 / 279) | 0.617 (174 / 282) | no bar by design |
| Moving-label recall | 0.863 (183 / 212) | 0.866 (187 / 216) | no bar by design |
| False-moving rate | 0.081 (320 / 3959) | 0.528 (2081 / 3942) | no bar by design |
| Car+Van recall | 0.852 (306 / 359) | 0.830 (298 / 359) | no bar by design |
| ID switches | 13 | 11 | no bar by design |
| Unlabelled tracks: median speed (m/s) | 0.219 | 7.819 | no bar by design |
| Unlabelled tracks: median position spread (m) | 0.540 | 10.825 | no bar by design |
| Parked-car reference: median speed (m/s) | 0.247 | 7.971 | no bar by design |

The off run tracks in the velodyne frame, so its positions move with the car and its velocities are measured in the sensor axes; the ground-truth world velocity is rotated into those axes (rotation only, no ego-velocity subtraction). The two runs match different sets of frames (the counts n differ), so the ratio compares RMSE over different frames.

## 5. Method and reproducibility

*How to read this:* How every number was computed, and the one command that regenerates this file byte for byte on the same CPU architecture.

- **Matching:** per frame, one-to-one Hungarian assignment on 2D center distance, gate 2.0 m; ground truth of every class in the evaluation region against all confirmed tracks (the same matching as the tracking speed scores).
- **Evaluation region:** the processing ROI minus the ego box, front half only (x >= 0 in the velodyne frame). Ground truth is tested on its velodyne center; a track is mapped back to velodyne and tested on x and y (tracks.csv has no z).
- **Unit:** one ground-truth box-frame. Recall is matched / GT box-frames in the region; matched + missed equals that count.
- **Static:** a tracklet whose world net speed is below 0.5 m/s (the 7 parked tracklets); every other tracklet is moving.
- **Warm-up:** the first 5 frames after a track's confirmation are excluded from velocity RMSE, static accuracy, moving-label recall, the false-moving rate and world stillness. They are not excluded from recall, label coverage, center error, ID switches or lifetimes.
- **Velocity:** the track's filtered velocity against the world-frame GT velocity, a centered difference over +-5 frames; RMSE is the root of the mean squared vector difference.
- **ID switch:** per tracklet, a matched frame whose track ID differs from the previous matched track ID; frames without a match do not reset it.
- **Unlabelled track:** a track with at least one frame in the region that is never matched to a tracklet. World stillness scores tracks with at least 10 post-warm-up frames.
- **Zero denominators** print `n/a` with the name of the empty denominator, never 0.
- **Gate:** only the ablation ratio (at least 3.0) is a pass/fail bar, fixed before the first evaluation run.
- **Reproduce:** `docker compose run --rm --no-deps processor python3 /app/scripts/evaluate.py --set drive=2011_09_26_drive_0011`. Two runs give byte-identical files on the same CPU architecture (verified on arm64; not checked on x86_64).
