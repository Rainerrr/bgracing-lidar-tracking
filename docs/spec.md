# LiDAR Obstacle Tracking: Revised Project Spec

Nov 5, 2026 · @rene

## Overview

The pipeline replays a KITTI raw drive as a live LiDAR sensor, detects obstacles, and tracks them in a world frame. It reports ego-motion-compensated velocities and scores them against KITTI's ground-truth tracklets. Two Docker containers (Ubuntu 24.04, ROS 2 Jazzy, Python 3.12) run the producer and the processor. The results are visible live in a browser through Foxglove and as static files any grader can open.

### Hard requirements from the assignment

Source: BGRacing Autonomous Department home assignment. Due **21/11/26**.

- Exactly two Docker containers, as the building blocks: one acquires and publishes data, the other reads and processes it. Both run ROS 2 nodes.
- Containers run **Ubuntu 24.04**, **ROS 2 Jazzy**, and **Python 3.12 or C++17**.
- A **GitHub repo** holds all project files and code. Reviewers must be able to open it (make it public or share it before submitting).
- The **README** shows the chosen dataset's properties and statistics, plus the project: chosen task, pipeline blocks, results, and how to run.
- The project must be **reproducible by the reviewers**: they download the project files, run the containers, and reproduce the results.
- Any relevant dataset is allowed; KITTI is one of the suggested ones. All tools are valid as long as you understand your work and can answer for it.
- Submit through the form: https://forms.gle/q4teuV2GQBwLSESB8
- Questions go to Nikolai Krokhmal: nikolaik@post.bgu.ac.il

### What changed from v1

- **Velocities are in the world frame.** The publisher also streams the car's pose from KITTI's GPS/IMU (OXTS) data. Parked cars now read about 0 m/s instead of minus the ego speed.
- **Time comes from the dataset.** Δt uses KITTI's sensor timestamps carried in message headers, never wall-clock or timer time.
- **Tracking is a real tracker.** Hungarian assignment with gating, track IDs with a confirm/delete lifecycle, and a constant-velocity Kalman filter per track replace greedy nearest neighbor.
- **The pipeline keeps up with the data.** Region-of-interest cropping, voxel downsampling and 2D clustering keep each frame under 100 ms. Sensor-data QoS drops stale frames instead of queueing them.
- **Ground removal uses RANSAC.** A numpy plane fit replaces the fixed Z cut, which fails on slopes and leaves curbs.
- **Results are measured.** An evaluation script reports detection recall, velocity error and ID switches against tracklets. A stats script produces the dataset statistics the assignment asks for.
- **Grading is reproducible on any OS.** Foxglove replaces RViz2-in-Docker as the default viewer, so there is no X11. A default bridge network replaces host networking. Dependencies come from apt rather than pip.

## Tech stack and key decisions

Every choice below favors a grader running `docker compose up` on an unknown machine and seeing results with no setup.

| Area | Choice | Why |
|---|---|---|
| Base image | `ros:jazzy-ros-base` (Ubuntu 24.04, amd64 + arm64) | Meets the OS and ROS requirements; multi-arch, so it runs on Apple Silicon too |
| Language | Python 3.12 (system Python in the image) | Required by the assignment; no version juggling |
| Python deps | apt: `python3-numpy`, `python3-scipy`, `python3-sklearn`, `python3-matplotlib`, `ros-jazzy-sensor-msgs-py`, `ros-jazzy-tf2-ros` | Ubuntu 24.04 blocks system-wide pip installs; apt packages match ROS's Python |
| Clustering | scikit-learn DBSCAN | Standard, well documented, easy to defend |
| Ground removal | RANSAC plane fit in numpy (~30 lines) | No open3d dependency (wheel and ARM issues); you can explain every line |
| Assignment | `scipy.optimize.linear_sum_assignment` | Optimal one-to-one matching in one call |
| Viewer (default) | Foxglove via `ros-jazzy-foxglove-bridge`, port 8765 | Browser-based on any OS; no X11 or GUI in Docker |
| Viewer (optional) | RViz2 config file for Linux users | Some reviewers prefer it; costs nothing to include |
| Network | Default compose bridge network, shared `ROS_DOMAIN_ID` | DDS discovery works inside one compose network; host mode behaves differently on Docker Desktop |
| Config | ROS 2 parameter YAML per node | Tuning without rebuilding; parameters documented in one place |

The `requirements.txt` files from v1 are dropped in favor of apt lines in each Dockerfile. If a pip-only package becomes necessary, install it into a venv created with `--system-site-packages`.

## Dataset

Use one short KITTI raw "synced + rectified" drive that ships with tracklet labels, for example `2011_09_26_drive_0005_sync` (city, moving and parked cars, about 150 frames). Confirm on the KITTI raw page that the drive you pick has a tracklets download before committing to it.

### Files used per drive

| Path | Contents | Used for |
|---|---|---|
| `velodyne_points/data/*.bin` | float32 `[x, y, z, reflectance]` per point, one file per sweep | The point clouds |
| `velodyne_points/timestamps.txt` | One timestamp per sweep | Message header stamps and Δt |
| `oxts/data/*.txt` | GPS/IMU: lat, lon, alt, roll, pitch, yaw, velocities | Ego pose per frame |
| `calib_imu_to_velo.txt` (date folder) | Rigid transform IMU → Velodyne | Placing the LiDAR in the world frame |
| `tracklet_labels.xml` | Per-object boxes, class and pose per frame | Evaluation ground truth |

### Getting the data

`scripts/download_kitti.sh` downloads and unpacks the drive, calibration and tracklets into `./data`. KITTI may require registration and a license acceptance, so the script checks for the files and prints exact manual-download steps if fetching fails. `./data` is git-ignored. KITTI is licensed CC BY-NC-SA 3.0; credit it in the README.

### Statistics for the README

Computed by `scripts/dataset_stats.py`, saved to `output/`:

- Frames, duration, and mean sensor rate from the timestamps
- Points per frame (mean, min, max) and the share removed by each pipeline stage
- Range histogram (points vs distance) and reflectance histogram
- Ego speed over the drive from OXTS
- Ground-truth objects per frame, by class (Car, Pedestrian, Cyclist, Van…)

## System architecture

Two containers on compose's default bridge network exchange ROS 2 topics, and every message from the publisher carries the dataset timestamp of its sweep.

The tracker is the only stateful part; everything a grader inspects flows out of it, live to the browser and as files to `./output`. Two coordinate frames exist: `world`, with the first OXTS pose as origin, and `velodyne`. Clouds are published in `velodyne`; boxes, arrows and labels in `world`.

### Container A: publisher

Node `kitti_lidar_publisher` replays the drive as a live sensor: each tick publishes one sweep, the matching ego pose, and the `world → velodyne` transform, all with the same dataset timestamp.

**Published:**

| Topic | Type | QoS | Contents |
|---|---|---|---|
| `/sensor/lidar/raw_points` | `sensor_msgs/msg/PointCloud2` | Sensor data (best effort, depth 5) | Fields x, y, z, intensity (float32); `frame_id: velodyne` |
| `/sensor/ego/odom` | `nav_msgs/msg/Odometry` | Reliable, depth 10 | Pose of the velodyne frame in world, plus OXTS velocity |
| `/tf` | `tf2_msgs/msg/TFMessage` | Default | `world → velodyne` per frame |

**Logic:**

1. On start, load the file list, `timestamps.txt`, all OXTS rows and the IMU → Velodyne calibration.
2. Convert OXTS to poses the way the KITTI devkit does: Mercator projection of lat/lon with the first frame as origin, plus roll/pitch/yaw. Compute `T_world_velo = T_world_imu · inverse(T_velo_imu)`.
3. A ROS timer fires at `rate_hz` (default 10). Each tick reads the next `.bin` with `np.fromfile(..., dtype=np.float32).reshape(-1, 4)`.
4. Build the cloud with `sensor_msgs_py.point_cloud2.create_cloud`, set `header.stamp` from the dataset timestamp, and publish cloud, odometry and TF.
5. At the end, loop to frame 0 if `loop: true`. Timestamps then jump backward, which the processor detects and uses to reset its tracker.

**Parameters:** `data_dir`, `drive`, `rate_hz`, `loop`, `start_frame`, `end_frame`.

Lowering `rate_hz` (for example to 2) lets the processor handle every frame for evaluation runs. Velocities stay correct at any rate because Δt comes from stamps.

### Container B: processor and tracker

Node `obstacle_tracker_node` turns each sweep into tracked obstacles with world-frame velocities in under 100 ms per frame. It subscribes to `/sensor/lidar/raw_points` and `/sensor/ego/odom`, pairing them by header stamp.

**Pipeline per frame:**

1. **Reset check.** If the stamp went backward or jumped more than 1 s, clear all tracks. This handles the dataset loop.
2. **Region of interest.** Keep x in [-40, 40] m, y in [-20, 20] m and z in [-3, 2] m. Drop the box around the ego car (`|x| < 2.5 m`, `|y| < 1.5 m`).
3. **Voxel downsample.** Keep one point per 0.15 m voxel by rounding to a grid and using `np.unique`. This cuts about 120k points to roughly 15–25k.
4. **Ground removal (RANSAC).** Fit a plane from 3 random points, 100 iterations. Count inliers within 0.2 m and accept only near-horizontal planes (`|normal z| > 0.9`). Remove the inliers of the best plane.
5. **Clustering.** Run DBSCAN on the (x, y) coordinates only, with eps 0.6 m and min_samples 8. Clustering in 2D is faster and avoids splitting objects vertically.
6. **Cluster filtering.** Keep clusters with at least 10 points, length ≤ 7 m, width ≤ 3 m and height between 0.3 and 3 m. This removes walls, buildings and vegetation.
7. **Detections.** For each cluster, compute the box center (more stable than the mean of points) and the axis-aligned extents. Transform centers to the world frame with the paired ego pose.
8. **Prediction.** Advance each track's Kalman filter by Δt from the header stamps.
9. **Association.** Build a cost matrix of distances between predicted track positions and detections. Set costs above a 2.0 m gate to infinity, then solve with `linear_sum_assignment`.
10. **Update and lifecycle.** Matched tracks get a Kalman update. Unmatched detections start tentative tracks. A track is confirmed after 3 hits and deleted after 5 consecutive misses.
11. **Motion state.** A confirmed track with filtered speed above 0.7 m/s is "moving"; otherwise it is "static".
12. **Publish and log.** Send markers and the filtered cloud, then append confirmed tracks to `output/tracks.csv` (stamp, id, x, y, vx, vy, speed, state, box size).

**Kalman filter:** state `[x, y, vx, vy]` in the world frame, constant-velocity motion model, measurement `[x, y]`. Process noise `q_accel` and measurement noise `r_pos` are parameters. Velocity comes from the filter state, not from raw centroid differences, so partial views don't make arrows jump.

**Published:**

| Topic | Type | Frame | Contents |
|---|---|---|---|
| `/perception/filtered_points` | `PointCloud2` | velodyne | Points after ROI, downsampling and ground removal |
| `/perception/bounding_boxes` | `MarkerArray` | world | `LINE_LIST` wireframe per confirmed track, colored by state |
| `/perception/obstacle_vectors` | `MarkerArray` | world | `ARROW` from box center along (vx, vy), length = speed × 0.5 s |
| `/perception/track_labels` | `MarkerArray` | world | `TEXT_VIEW_FACING`: track ID and speed in km/h |

Every `MarkerArray` begins with a `DELETEALL` marker so deleted tracks don't leave ghosts. Per-frame processing time is logged and summarized in `output/metrics.json`.

**Parameters** (defaults, in `config/tracker.yaml`):

- `roi_*`
- `ego_box`
- `voxel_size: 0.15`
- `ransac_iters: 100`
- `ransac_dist: 0.2`
- `dbscan_eps: 0.6`
- `dbscan_min_samples: 8`
- `cluster_min_points: 10`
- `max_length: 7.0`
- `max_width: 3.0`
- `gate_m: 2.0`
- `confirm_hits: 3`
- `max_misses: 5`
- `q_accel`
- `r_pos`
- `moving_speed: 0.7`
- `report_every_n: 50`
- `output_dir: /output`

## Visualization and outputs

Graders get a live 3D view in their browser and a set of static files that need no ROS at all.

### Live view (Foxglove)

`foxglove_bridge` runs inside Container B and exposes `ws://localhost:8765`. Open Foxglove in a browser, connect to that address, and import the committed layout `viz/foxglove_layout.json`. The layout sets the fixed frame to `world`, so the ego car visibly drives through a static scene and parked cars stay still. It shows the raw and filtered clouds, boxes, arrows and labels, plus a plot panel of ego speed. Linux users can instead run RViz2 with `viz/tracker.rviz`.

### Static outputs

Written to the mapped `./output` volume every `report_every_n` frames and on shutdown:

| File | Contents |
|---|---|
| `obstacle_tracking_report.png` | Top-down world-frame map: the ego path, each confirmed track's trajectory colored by ID, and velocity arrows at the last position. Static tracks are grey. |
| `tracks.csv` | Every confirmed track state per frame |
| `metrics.json` | Frames processed, mean and p95 processing time, track counts |
| `dataset_stats.png` | Histograms and per-frame counts from the stats script |
| `evaluation.md` | Metrics table from the evaluation script |

Writing the report periodically, not only on exit, matters because `docker compose down` can kill the node before a clean shutdown handler runs. Also catch SIGTERM and write one final report.

## Evaluation against ground truth

`scripts/evaluate.py` compares `tracks.csv` with `tracklet_labels.xml` offline and writes `output/evaluation.md`. It turns "it looks right in the viewer" into numbers you can defend.

### Method

1. Parse tracklets into per-frame boxes (center, size, yaw, class) in the velodyne frame. Map frames to stamps with `timestamps.txt`.
2. Transform ground-truth centers to the world frame with the same ego poses the publisher uses.
3. Compute ground-truth velocity by differencing world-frame centers over consecutive frames, smoothed over 3 frames.
4. Keep only ground truth inside the processing ROI, since objects outside it can't be detected by design.
5. Per frame, match confirmed tracks to ground truth with Hungarian assignment on center distance, using a 2.0 m gate.

### Metrics

Overall and per class:

| Metric | Definition |
|---|---|
| Recall | Matched ground-truth objects ÷ all ground-truth objects in ROI |
| Precision | Matched tracks ÷ all confirmed tracks |
| Center error | Mean distance between matched track and ground-truth centers (m) |
| Velocity RMSE | RMSE of (vx, vy) for matched pairs (m/s), reported separately for moving and static objects |
| Static accuracy | Share of truly static objects (GT speed < 0.5 m/s) that the tracker labels "static" |
| ID switches | Times a ground-truth object's matched track ID changes |

### Ablations

Include in the README: run once without ego-motion compensation and once with greedy matching instead of Hungarian plus Kalman. The jump in static-object velocity error between those runs is the strongest single result in the project.

### DBSCAN sweep

Run a sweep over `eps` in {0.4, 0.5, 0.6, 0.8} and `min_samples` in {5, 8, 12}. Report recall, precision and runtime per setting, and use it to justify the chosen defaults.

## Repository layout and reproducibility

A grader runs three commands: download the data, `docker compose up`, then open Foxglove or the `output/` folder.

```text
bgracing-lidar-tracking/
├── docker-compose.yml
├── README.md
├── publisher/
│   ├── Dockerfile.publisher
│   ├── kitti_lidar_publisher.py
│   └── kitti_utils.py          # bin/timestamp/OXTS/calib loading, pose math
├── processor/
│   ├── Dockerfile.processor
│   ├── obstacle_tracker_node.py
│   ├── ground.py               # RANSAC
│   ├── tracker.py              # Kalman + Hungarian + lifecycle
│   ├── markers.py
│   ├── report.py               # matplotlib PNG
│   └── processor.launch.py     # tracker + foxglove_bridge
├── config/
│   ├── publisher.yaml
│   └── tracker.yaml
├── scripts/
│   ├── download_kitti.sh
│   ├── dataset_stats.py
│   └── evaluate.py
├── viz/
│   ├── foxglove_layout.json
│   └── tracker.rviz
├── tests/                      # pytest: RANSAC, tracker, pose math
├── data/                       # git-ignored
└── output/                     # git-ignored, except sample results
```

### docker-compose.yml (sketch)

```yaml
services:
  publisher:
    build: { context: ., dockerfile: publisher/Dockerfile.publisher }
    environment: [ROS_DOMAIN_ID=42]
    volumes: ["./data:/data:ro", "./config:/config:ro"]
  processor:
    build: { context: ., dockerfile: processor/Dockerfile.processor }
    environment: [ROS_DOMAIN_ID=42]
    volumes: ["./data:/data:ro", "./config:/config:ro", "./output:/output"]
    ports: ["8765:8765"]
    depends_on: [publisher]
```

No `network_mode` is set, so both services share compose's default bridge network. Scripts run directly with `python3` after sourcing ROS. The launch file starts them with `ExecuteProcess`, which avoids building colcon packages.

### Commands for the README

```bash
./scripts/download_kitti.sh                     # fetch the drive into ./data
docker compose up --build                       # run the pipeline
# open https://app.foxglove.dev → Open connection → ws://localhost:8765
docker compose run --rm processor python3 /app/scripts/evaluate.py
docker compose run --rm processor python3 /app/scripts/dataset_stats.py
```

### Reproducibility checks before submitting

- [ ] Fresh clone on a second machine (or a clean VM) runs end to end with only the commands above
- [ ] Works on both amd64 and arm64 images
- [ ] `output/` files are readable by the host user (set `ROS_HOME=/tmp/ros` and, if needed, `user: "${UID}:${GID}"`)
- [ ] Sample outputs committed to `docs/` so reviewers see results without running anything
- [ ] Pinned base image tag in both Dockerfiles
- [ ] Both containers verified as Ubuntu 24.04 with Python 3.12 (`cat /etc/os-release`, `python3 --version`)
- [ ] GitHub repo is public, or shared with the reviewers, before submitting

## Build plan

The MVP lands over the weekend, correctness by Sunday night, and the evaluation work on weekday evenings, leaving Wednesday for the README and a fresh-clone test before the internal 12/11 target. The official deadline is 21/11/26, which leaves about nine days of buffer. Each phase ends with something that runs, so a slip never leaves you with nothing to submit.

### Phase 0: plumbing (Fri evening, ~3 h)

Done when Foxglove shows a test cloud sent from Container A.

- [ ] Download the drive and tracklets; open one `.bin` in numpy to check its shape
- [ ] Both Dockerfiles build; compose starts both services
- [ ] Publisher sends one cloud; `ros2 topic echo` in Container B sees it
- [ ] `foxglove_bridge` connects from the browser

### Phase 1: MVP (Saturday, ~7 h)

Done when boxes appear around cars in the viewer.

- [ ] Publisher loops the full drive at 10 Hz with dataset timestamps
- [ ] ROI crop, voxel downsampling and RANSAC ground removal; filtered cloud published
- [ ] DBSCAN plus cluster filtering; box markers with `DELETEALL`
- [ ] Per-frame timing logged; confirm it stays under 100 ms
- [ ] First version of the PNG report

### Phase 2: correct tracking (Sunday, ~7 h)

Done when parked cars show about 0 m/s and moving cars show plausible speeds.

- [ ] OXTS → poses, IMU → velodyne calibration, `/tf` and `/sensor/ego/odom` published
- [ ] Detections transformed to the world frame
- [ ] Kalman filter, Hungarian matching with gating, track lifecycle, loop reset
- [ ] Arrows, labels and static/moving coloring; Foxglove layout saved
- [ ] Unit tests for RANSAC, pose math and tracker

### Phase 3: evidence (Mon–Tue evenings, ~5 h)

Done when `evaluation.md` has real numbers.

- [ ] `dataset_stats.py` and its plots
- [ ] `evaluate.py` with recall, precision, velocity RMSE and ID switches
- [ ] Ablations: no ego compensation, greedy matching
- [ ] DBSCAN parameter sweep table

### Phase 4: ship (Wednesday, ~3 h)

Done when a fresh clone reproduces everything.

- [ ] README with all sections below, an architecture diagram and a GIF of Foxglove
- [ ] Fresh-clone test on a second machine; fix whatever breaks
- [ ] Submit through the form (https://forms.gle/q4teuV2GQBwLSESB8); keep Thursday morning as buffer

### If you fall behind

Cut in this order: the DBSCAN sweep, then the ablations, then the RViz config. Never cut ego-motion compensation or the basic evaluation; they are what make the project stand out.

## Risks, stretch goals and README

### Risks

| Risk | Sign | Fallback |
|---|---|---|
| KITTI download needs a login | Script can't fetch files | Manual steps in the README; check whether the license allows committing a few sample frames with attribution |
| DDS discovery fails between containers | `ros2 topic list` in B misses A's topics | Same `ROS_DOMAIN_ID`; try `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`; last resort `network_mode: host` on Linux |
| Processing exceeds 100 ms | Timing log, dropped frames | Larger voxels, tighter ROI, fewer RANSAC iterations; lower `rate_hz` |
| Ego pose looks wrong | Static objects drift in the world view | Plot the OXTS path alone first; check the calibration direction (IMU → velo vs velo → IMU) |
| Velocities are noisy | Arrows flicker | Raise `r_pos`, require more hits before confirming, use box center rather than mean |
| Walls or bushes tracked as obstacles | Large or oddly shaped boxes | Tighten size filters; require a minimum height |

### Stretch goals

Only after Phase 4 is done:

- **Cluster classifier.** Train a small scikit-learn model on geometric features (extents, point count, height profile) using tracklet classes as labels. It adds a learned component you can measure.
- **Formula Student variant.** If the team's provided LiDAR sample contains cones, swap the cluster filter for cone-sized clusters and estimate track boundaries and a centerline. It sits closest to BGRacing's actual domain.
- **Time-to-collision warnings** for tracks on a path crossing the ego car's.
- **Oriented boxes** from a PCA of each cluster's (x, y) points.

### README outline

1. One-paragraph summary plus a GIF of the Foxglove view
2. Architecture diagram of containers, nodes, topics and frames
3. Dataset: drive used, file formats, statistics plots
4. Pipeline: each stage with its parameters and why those values
5. Results: evaluation table, ablations, DBSCAN sweep, the PNG report
6. How to run: the exact commands, expected outputs, troubleshooting
7. Limitations and next steps, stated honestly
8. Credits: KITTI citation and license
