# LiDAR Obstacle Tracking

BGRacing Autonomous home assignment: a two-container ROS 2 Jazzy pipeline that replays a KITTI raw LiDAR drive, detects obstacles, and tracks them in a world frame. See [docs/spec.md](docs/spec.md) for the full spec.

> Work in progress (Phase 2: detection). The publisher streams the point cloud, the ego odometry and the `world → velodyne` transform; the processor detects obstacles and draws a box around each one. Tracking, velocities and the evaluation tables come in Phases 3–4.

## Run

```bash
./scripts/download_kitti.sh                     # fetch the drive into ./data
docker compose up --build                       # run the pipeline
```

Then open the Foxglove web app at <https://app.foxglove.dev> in Chrome, Edge or Firefox (a free account is required), choose Open connection → Foxglove WebSocket → `ws://localhost:8765`, and set the 3D panel's display frame to `world`. The ego car then drives through a still scene, and the processor's `/perception/filtered_points` and `/perception/bounding_boxes` topics show the filtered cloud and one wireframe box per detection (each with a text label of its size and point count). Safari, and Brave with Shields on, can block the local connection.

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
| Points kept by each stage | 92.0 % after the ROI, 26.2 % after voxel downsampling, 14.7 % after ground removal, 3.5 % in the kept clusters (means 112,002 / 31,938 / 17,942 / 4,311 points per sweep) |
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
4. **Clustering:** DBSCAN on (x, y), eps 0.6 m, at least 8 points.
5. **Size filter:** a cluster stays if it has at least 10 voxels, an oriented length ≤ 7 m and width ≤ 3 m (the minimum-area rectangle of its footprint), and a height between 0.3 and 3 m. The orientation matters: the car turns about 75° during the drive, so an axis-aligned width limit would reject 37–46 % of the vans in the ground truth even with perfect segmentation.
6. **Detection:** the axis-aligned box of each kept cluster (center, extents dx/dy/dz, voxel count). Boxes are not rotated.

All parameters are in [`config/tracker.yaml`](config/tracker.yaml), read by the live node and the offline runner alike.

Run it offline (no publisher; results go to `output/offline/`, never over the live files):

```bash
docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py
```

### Results

Scored against the KITTI tracklets in the front half of the region of interest (where ground truth exists), one-to-one Hungarian matching on 2D box centers with a 2.0 m gate, over all 154 sweeps. Source: [`docs/samples/detection_recall.json`](docs/samples/detection_recall.json); two runs give byte-identical files.

| Class | Matched / ground-truth boxes | Recall | Tracklets |
|---|---|---|---|
| **Car + Van (the gated number)** | **428 / 520** | **0.823** | 12 |
| Car | 210 / 248 | 0.847 | 9 |
| Van | 218 / 272 | 0.802 | 3 |
| Pedestrian (small sample) | 6 / 22 | 0.273 | 2 |
| Cyclist (small sample) | 148 / 154 | 0.961 | 1 |

- The mean distance between matched centers is 0.56 m (90th percentile 1.44 m). Boxes are axis-aligned and built from the points the sensor sees, so a car seen from one side can have its box center offset from the true center (up to 1.9 m even for a perfect segmentation of the ground-truth boxes).
- The weakest tracklet is parked van #3, matched in 70 of 107 frames (0.654). Pedestrians are mostly missed (6 of 22 boxes, two tracklets); the cause has not been diagnosed. No parameter was tuned against these numbers.
- On average 31.6 detections per sweep in that region match no ground-truth box (maximum 53). This is not a false-positive rate: the tracklets label only a few object types in front of the car, so unlabeled objects count as unmatched. Precision is evaluated in Phase 4.
- Time per sweep for the detection stages (the sweep's file read excluded), on an Apple M5 Pro in Docker (linux/arm64): 95th percentile 49.3–52.1 ms over three runs (mean about 42 ms), against a 75 ms bar and the 100 ms live period. Live, with the publisher and bridge sharing the CPU: about 55 ms (the whole callback about 60 ms), and 2 of 678 sweeps were skipped in the recorded run. Timing varies by machine and is not part of the committed samples.
- The recall gate in `./scripts/verify.sh` is 0.8 and the measured value is 0.823 (12 box-frames of margin). Results are byte-identical per CPU architecture; on a different architecture (for example amd64) a float-level difference could in principle move a few frames, and this has not been tested there.

## Verify

```bash
./scripts/verify.sh
```

This builds both images, checks the publisher image's imports, and runs the unit tests inside the processor image, including the Car+Van recall gate (≥ 0.8) and the two-run determinism check. It needs the drive in `./data`: the real-data tests fail (they never skip) without it.

## Credits

Dataset: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/) (Geiger, Lenz, Stiller, Urtasun), licensed CC BY-NC-SA 3.0.
