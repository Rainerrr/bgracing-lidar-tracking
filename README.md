# LiDAR Obstacle Tracking

BGRacing Autonomous home assignment: a two-container ROS 2 Jazzy pipeline that replays a KITTI raw LiDAR drive, detects obstacles, and tracks them in a world frame. See [docs/spec.md](docs/spec.md) for the full spec.

> Work in progress (Phase 3: tracking). The publisher streams the point cloud, the ego odometry and the `world → velodyne` transform; the processor detects obstacles, tracks them in the world frame with ego-motion-compensated velocities, and scores the speeds against the ground truth. The full evaluation tables come in Phase 4.

## Run

```bash
./scripts/download_kitti.sh                     # fetch the drive into ./data
docker compose up --build                       # run the pipeline
```

Then open the Foxglove web app at <https://app.foxglove.dev> in Chrome, Edge or Firefox (a free account is required), choose Open connection → Foxglove WebSocket → `ws://localhost:8765`, and set the 3D panel's display frame to `world`. The ego car then drives through a still scene. The processor publishes the filtered cloud (`/perception/filtered_points`) and, in `world`, one wireframe box per confirmed track (`/perception/bounding_boxes`), a velocity arrow per moving track (`/perception/obstacle_vectors`) and a label such as `#12 static 0 km/h` or `#3 moving 17 km/h` (`/perception/track_labels`). Hide the transform axes and links in the 3D panel's frame settings: Foxglove draws them at its own clock time, not at the dataset's, so they don't follow the car. Safari, and Brave with Shields on, can block the local connection.

The bridge listens on `127.0.0.1` only, because `foxglove_bridge` has no authentication. Open Foxglove on the machine that runs Docker.

## Dataset

KITTI raw drive `2011_09_26_drive_0005`, "synced + rectified": a city drive with moving and parked cars. The pipeline uses the Velodyne HDL-64 sweeps, the OXTS GPS/IMU poses, the IMU → Velodyne calibration and the tracklet ground truth.

![Dataset statistics](docs/samples/dataset_stats.png)

| Property | Value |
|---|---|
| Sweeps | 154 over 15.80 s |
| Sensor rate | 9.68 Hz mean (period 102.7–103.7 ms), not 10 Hz |
| Points per sweep | mean 121,732; min 98,532; max 124,122 (float32 x, y, z, reflectance) |
| Range | 83 % of points within 20 m of the sensor, 96 % within 40 m, none beyond 82 m |
| Ego speed (OXTS) | 3.1–6.5 m/s, mean 4.5 m/s |
| Points kept by each stage | 92.0 % after the ROI, 26.2 % after voxel downsampling, 12.5 % after ground removal, 3.8 % in the kept clusters (means 112,002 / 31,938 / 15,232 / 4,565 points per sweep) |
| Ground truth | 15 tracklets (9 Car, 3 Van, 2 Pedestrian, 1 Cyclist), 717 boxes, all in front of the car |

The numbers come from [`docs/samples/dataset_stats.json`](docs/samples/dataset_stats.json). Regenerate them (no publisher needed) into `output/`:

```bash
docker compose run --rm --no-deps processor python3 /app/scripts/dataset_stats.py
```

Two runs produce byte-identical files, and on the same CPU architecture they match `docs/samples/`.

### Frames and time

- `velodyne`: the sensor frame; clouds are published in it.
- `world`: ENU axes (x east, y north, z up) with the first GPS/IMU position as origin. Poses come from OXTS via a Mercator projection, the same convention as pykitti, and `T_world_velo = T_world_imu · inverse(T_velo_imu)`.
- Time: every message carries the sweep's stamp from `velodyne_points/timestamps.txt`, read as UTC. Nothing uses wall-clock time.

## Detection

Each sweep goes through the same ROS-free code in the live node and in the offline runner (`processor/perception.py`), in the `velodyne` frame:

1. **Region of interest:** x ∈ [−40, 40] m, y ∈ [−20, 20] m, z ∈ [−3, 2] m, minus the box around the ego car (|x| < 2.5 m, |y| < 1.5 m).
2. **Voxel downsampling:** one point per 0.15 m voxel, placed at the voxel's grid center, so the result does not depend on point order.
3. **Ground removal:** RANSAC plane fit (100 iterations, 0.2 m inlier distance), seeded per frame from a fixed seed and the sweep's stamp. The best plane is removed only if it is near-horizontal (|normal z| > 0.9, about 26° of tilt or less); otherwise the frame is marked `no_plane` and nothing is removed.
   Then the ground is refined per 10 × 10 m tile: a plane is fitted to each tile's voxels within 0.6 m of the global plane (trimmed twice to 0.15 m) and voxels within 0.2 m of it are removed too. One plane can't follow 80 m of road, and the band of ground it leaves 0.2–0.4 m above itself used to join cars to walls and bushes in the 2D clustering. A tile with fewer than 30 candidates, or a plane steeper than the rule above, keeps the global result.
4. **Clustering:** DBSCAN on (x, y), eps 0.6 m, at least 8 points.
5. **Size filter:** a cluster stays if it has at least 10 voxels, an oriented length ≤ 7 m and width ≤ 3 m (the minimum-area rectangle of its footprint), and a height between 0.3 and 3 m. The orientation matters: the car turns about 75° during the drive, so an axis-aligned width limit would reject 37–46 % of the vans in the ground truth even with perfect segmentation.
6. **Detection:** the axis-aligned box of each kept cluster (center, extents dx/dy/dz, voxel count). Boxes are not rotated.

![Foxglove: filtered cloud and one labeled box per detection, live](docs/samples/detection_foxglove.png)

The live view in Foxglove (fixed and display frame `velodyne`): the filtered cloud and one wireframe box per detection, each labeled with its size and point count.

All parameters are in [`config/tracker.yaml`](config/tracker.yaml), read by the live node and the offline runner alike.

Run it offline (no publisher; results go to `output/offline/`, never over the live files):

```bash
docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py
```

### Results

Scored against the KITTI tracklets in the front half of the region of interest (where ground truth exists), one-to-one Hungarian matching on 2D box centers with a 2.0 m gate, over all 154 sweeps. Source: [`docs/samples/detection_recall.json`](docs/samples/detection_recall.json); two runs give byte-identical files.

| Class | Matched / ground-truth boxes | Recall | Tracklets |
|---|---|---|---|
| **Car + Van (the gated number)** | **472 / 520** | **0.908** | 12 |
| Car | 234 / 248 | 0.944 | 9 |
| Van | 238 / 272 | 0.875 | 3 |
| Pedestrian (small sample) | 6 / 22 | 0.273 | 2 |
| Cyclist (small sample) | 153 / 154 | 0.994 | 1 |

- The mean distance between matched centers is 0.54 m (90th percentile 1.39 m). Boxes are axis-aligned and built from the points the sensor sees, so a car seen from one side can have its box center offset from the true center (up to 1.9 m even for a perfect segmentation of the ground-truth boxes).
- The weakest tracklet is parked van #3, matched in 76 of 107 frames (0.710). The local ground tiles helped it only a little; its misses are still clusters merged into something larger than a van, and I have not diagnosed what it merges with. Pedestrians are mostly missed (6 of 22 boxes, two tracklets); the cause has not been diagnosed. No parameter was tuned against these numbers.
- On average 32.0 detections per sweep in that region match no ground-truth box (maximum 54); the ground tiles did not change this. This is not a false-positive rate: the tracklets label only a few object types in front of the car, so unlabeled objects count as unmatched. Precision is evaluated in Phase 4.
- Time per sweep for the detection stages (the sweep's file read excluded), on an Apple M5 Pro in Docker (linux/arm64): 95th percentile 49.7–50.4 ms over three runs (mean about 45 ms), against a 75 ms bar and the 100 ms live period. Pinned to a single core the 95th percentile was 64 ms. Live, with the publisher and bridge sharing the CPU: about 48 ms (the whole callback about 53 ms), and no sweep was skipped in the recorded run (678 of 678). Timing varies by machine and is not part of the committed samples.
- The recall gate in `./scripts/verify.sh` is 0.8 and the measured value is 0.908 (56 box-frames of margin). Results are byte-identical per CPU architecture; on a different architecture (for example amd64) a float-level difference could in principle move a few frames, and this has not been tested there.

## Tracking

Detections become tracks in the `world` frame (east, north, up; origin at the drive's first GPS/IMU position). Each sweep is paired with the ego pose that carries the same dataset timestamp, and the same ROS-free code runs live and offline (`processor/tracking.py`):

1. **World frame:** each detection center is moved into `world` with the paired pose, so parked objects keep the same position while the car drives.
2. **Prediction:** a constant-velocity Kalman filter per track, state (x, y, vx, vy). Δt comes from the dataset timestamps, never from wall time. Process noise 1.0 (m/s²)², measurement noise 0.5 m. Both were fixed before any tracking code ran and have not been tuned.
3. **Association:** Hungarian assignment on the distance between predicted and detected centers, gated at 2.0 m.
4. **Lifecycle:** a track is confirmed after 3 hits and deleted after 5 consecutive misses. A missing sweep counts as a miss, so a processing stall never resets anything. Only the dataset loop (timestamps jump back) resets the tracks, and IDs restart at 1 for each pass.
5. **Motion state:** a track is `moving` only if it is *confidently* faster than 0.7 m/s: speed − 2σ > 0.7 m/s, where σ is the filter's own speed uncertainty. With the plain rule (speed > 0.7 m/s), 197 of 417 tracks read as moving: mostly short-lived poles, bushes and wall pieces whose box center slides as the car drives past. The cost: σ settles at about 0.31 m/s, so after a few frames the effective threshold is about 1.3 m/s and a walking pedestrian is labelled `static` (its label still shows its speed).

Velocities come from the filter state, not from differences of box centers.

![Top-down map of the tracks in the world frame](docs/samples/obstacle_tracking_report.png)

The report (`obstacle_tracking_report.png`): the ego path in black, moving tracks as solid lines with an ID and a velocity arrow (0.5 s ahead), static tracks thin, dashed and grey under one legend entry. A track is drawn as moving if the tracker labelled it `moving` in at least half of its sweeps. Some short arrows are still clutter the filter is confident about; see the limitations below.

To make the real moving objects easy to find, the offline report tags every track matched to a ground-truth tracklet that moves (net speed ≥ 0.5 m/s): `#30 = GT #0 Van`, `#25 = GT #1 Cyclist`, `#27 = GT #2 Pedestrian` and `#295 = GT #4 Car`. The last two walk or crawl below the motion-state threshold, so they are drawn static (black, dash-dotted). Pedestrian #12 is never detected, so it has no track. **These class names come from the ground truth, as the footnote on the image says: the processor does not classify objects.** The live report has no tags.

Run it offline (results in `output/offline/`; with ego compensation off in `output/offline/no_ego/`):

```bash
docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py
docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py --set ego_compensation=false
```

### Results

Tracks are matched to the KITTI tracklets per sweep in the front half of the region of interest (Hungarian, 2D centers, 2.0 m gate) and scored per tracklet. A track's first 5 frames after confirmation are left out. Ground-truth speed is the difference of the tracklet's world positions over ±5 sweeps (1 s); over ±1 sweep, label jitter alone makes parked cars read 0.2–0.45 m/s. Source: [`docs/samples/tracking_speeds.json`](docs/samples/tracking_speeds.json) and, without ego compensation, [`docs/samples/tracking_speeds_no_ego.json`](docs/samples/tracking_speeds_no_ego.json); two runs give byte-identical files.

**Parked objects** (the 10 tracklets that move less than 0.5 m/s in the ground truth), median filtered speed in m/s:

| Tracklet | Class | Frames scored | With ego compensation | Labelled moving | Without (ablation) |
|---|---|---|---|---|---|
| #3 | Van | 88 | 0.25 | 0 % | 4.68 |
| #5 | Car | 34 | 0.50 | 9 % | 5.41 |
| #6 | Car | 28 | 0.09 | 0 % | 5.70 |
| #7 | Car | 33 | 0.22 | 0 % | 6.50 |
| #8 | Van | 22 | 0.57 | 0 % | 4.98 |
| #9 | Car | 34 | 0.11 | 0 % | 6.41 |
| #10 | Car | 28 | 0.57 | 7 % | 5.81 |
| #11 | Car | 16 | 0.54 | 0 % | 5.16 |
| #13 | Car | 9 | 0.66 | 0 % | 5.41 |
| #14 | Car | 19 | 0.16 | 0 % | 6.26 |
| **Median of the medians** | | | **0.38** (bar 0.5) | | |

Every parked tracklet reads under 0.7 m/s (the bar; the worst is #13 at 0.66, scored on only 9 frames). Without ego compensation the same cars read 4.7–6.5 m/s: the car's own motion.

**Moving objects** (median |filtered speed − ground-truth speed| per sweep; both accelerate during the drive):

| Tracklet | Class | Frames scored | Ground-truth speed (median) | Error (median, 90th pct.) | Labelled moving | Without ego compensation |
|---|---|---|---|---|---|---|
| #0 | Van | 103 | 4.77 m/s | 0.22, 1.00 m/s | 100 % | 1.87 m/s |
| #1 | Cyclist (small sample) | 147 | 4.62 m/s | 0.24, 0.47 m/s | 100 % | 3.72 m/s |

The bar is 0.5 m/s per object. `./scripts/verify.sh` gates both tables and checks that they fail without ego compensation.

**Live.** The live node gives byte-for-byte the same tracks as the offline tracker fed the same sweeps: checked over 3 passes (33,315 track rows), including a pass with a dropped sweep. On the dev machine the whole live callback, file writes included, took 59 ms at the 95th percentile, against the 100 ms sweep period; the tracking step itself takes under 2 ms.

**Limitations.**
- About 74 confirmed tracks per sweep, most of them unlabelled static objects (poles, bushes, wall pieces). They are honest tracks, not errors, but some of them drift enough to be labelled moving; precision is evaluated in Phase 4.
- Walking pedestrians are labelled `static` (see the motion state). Pedestrians are also mostly missed by detection (recall 0.27).
- Box sizes in `tracks.csv` (`l, w, h`) are the detection's extents along the sensor's axes, so they change as the car turns.
- The van's error reaches 1.0 m/s at the 90th percentile, at 30–44 m where it is seen from behind and partly.

## Verify

```bash
./scripts/verify.sh
```

This builds both images, checks the publisher image's imports, and runs the unit tests inside the processor image, including the Car+Van recall gate (≥ 0.8), the tracking speed gates (parked and moving tables above, each with a control that must fail without ego compensation) and the two-run determinism checks. It needs the drive in `./data`: the real-data tests fail (they never skip) without it.

## Credits

Dataset: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/) (Geiger, Lenz, Stiller, Urtasun), licensed CC BY-NC-SA 3.0.
