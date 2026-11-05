# LiDAR Obstacle Tracking: Project Spec

Nov 5, 2026 · @rene · v3 (phased)

This is the main project spec: what the system does, why, and the six phases that deliver it. Each phase is worked on its own. When a phase starts, it gets its own spec and plan (see [How phases are worked](#how-phases-are-worked)). This document stays the single source for scope, end goals and exit criteria.

## Overview

The pipeline replays a KITTI raw drive as a live LiDAR sensor, detects obstacles, and tracks them in a world frame. It reports ego-motion-compensated velocities and scores them against KITTI's ground-truth tracklets. Two Docker containers (Ubuntu 24.04, ROS 2 Jazzy, Python 3.12) run the producer and the processor. The results are visible live in a browser through Foxglove and as static files any grader can open.

### Hard requirements from the assignment

Source: BGRacing Autonomous Department home assignment. Due **21/11/26**.

- Exactly two Docker containers, as the building blocks: one acquires and publishes data, the other reads and processes it. Both run ROS 2 nodes.
- Containers run **Ubuntu 24.04**, **ROS 2 Jazzy**, and **Python 3.12 or C++17**.
- A **GitHub repo** holds all project files and code. Reviewers must be able to open it, so make it public or share it before submitting.
- The **README** shows the chosen dataset's properties and statistics, plus the project: the chosen task, the pipeline blocks, the results, and how to run it.
- The project must be **reproducible by the reviewers**: they download the project files, run the containers, and reproduce the results.
- Any relevant dataset is allowed; KITTI is one of the suggested ones. All tools are valid as long as you understand your work and can answer for it.
- Submit through the form: https://forms.gle/q4teuV2GQBwLSESB8
- Questions go to Nikolai Krokhmal: nikolaik@post.bgu.ac.il

### Who it's for

- **Reviewers (primary).** BGRacing Autonomous reviewers clone the repo, run the containers, inspect Foxglove and `output/`, and decide whether the submission is reproducible, correct and well understood.
- **The author (secondary).** The author must defend every design choice. Each phase therefore ends in evidence, not impressions.
- **Success looks like this:**
  - Every claim in the README is backed by a number that one command regenerates identically on the reviewer's machine.
  - The project is submittable at the end of Phase 5, before any optional work.

### What changed from v1

- **Velocities are in the world frame.** The publisher also streams the car's pose from KITTI's GPS/IMU (OXTS) data. Parked cars now read about 0 m/s instead of minus the ego speed.
- **Time comes from the dataset.** Δt uses KITTI's sensor timestamps carried in message headers, never wall-clock or timer time.
- **Tracking is a real tracker.** Hungarian assignment with gating, track IDs with a confirm/delete lifecycle, and a constant-velocity Kalman filter per track replace greedy nearest neighbor.
- **The pipeline keeps up with the data.** Region-of-interest cropping, voxel downsampling and 2D clustering keep each frame under 100 ms. Sensor-data QoS drops stale frames instead of queueing them.
- **Ground removal uses RANSAC.** A numpy plane fit replaces the fixed Z cut, which fails on slopes and leaves curbs.
- **Results are measured.** An evaluation script reports detection recall, velocity error and ID switches against tracklets. A stats script produces the dataset statistics the assignment asks for.
- **Grading is reproducible on any OS.** Foxglove replaces RViz2-in-Docker as the viewer, so there is no X11. A default bridge network replaces host networking. Dependencies come from apt rather than pip.

### What changed in v3

- **Work is split into six phases**, each with objective exit criteria and its own spec and plan. The old subjective lines ("boxes appear around cars", "plausible speeds") are gone.
- **Foundations come first.** The test harness, ground-truth loading and ego-pose math come before detection, so a wrong pose can't masquerade as a bad tracker.
- **Reported numbers come from an offline runner.** The same ROS-free core runs over all 154 frames, and one live-vs-offline parity check proves the ROS path matches it. A live replay drops frames and isn't repeatable.
- **Evaluation covers the front half only**, because KITTI ground truth exists only in front of the car.
- **The no-ego-compensation ablation is core**, not cuttable. It is the project's strongest single result.
- **The RViz2 config is dropped.** It can't be verified on the dev Mac, and Foxglove plus committed screenshots cover every reviewer.

## Tech stack and key decisions

Every choice below favors a grader running `docker compose up` on an unknown machine and seeing results with no setup.

| Area | Choice | Why |
|---|---|---|
| Base image | `ros:jazzy-ros-base` (Ubuntu 24.04, amd64 + arm64), pinned by digest | Meets the OS and ROS requirements; multi-arch, so it runs on Apple Silicon too |
| Language | Python 3.12 (system Python in the image) | Required by the assignment; no version juggling |
| Python deps | apt: `python3-numpy`, `python3-scipy`, `python3-sklearn`, `python3-matplotlib`, `ros-jazzy-sensor-msgs-py`, `ros-jazzy-tf2-ros` | Ubuntu 24.04 blocks system-wide pip installs; apt packages match ROS's Python |
| Clustering | scikit-learn DBSCAN | Standard, well documented, easy to defend |
| Ground removal | RANSAC plane fit in numpy (~30 lines), seeded | No open3d dependency (wheel and ARM issues); you can explain every line; seeding makes runs repeatable |
| Assignment | `scipy.optimize.linear_sum_assignment` | Optimal one-to-one matching in one call |
| Viewer | Foxglove web app (app.foxglove.dev) connected to `ros-jazzy-foxglove-bridge`, port 8765 | Browser-based on any OS; no X11, GUI or viewer install; needs a free Foxglove login |
| Network | Default compose bridge network, shared `ROS_DOMAIN_ID` | DDS discovery works inside one compose network; host mode behaves differently on Docker Desktop |
| Config | ROS 2 parameter YAML per node, also read by the offline runner | Tuning without rebuilding; parameters documented in one place |
| Evaluation runs | Offline runner over the shared ROS-free core, plus one live parity check | Deterministic and fast; the sweep and ablations are loops over configs |
| Tests | pytest from apt (`python3-pytest`) in the processor image, no ROS imports (Phase 1) | Runs inside the same Python as the pipeline; no third container |

The `requirements.txt` files from v1 are dropped in favor of apt lines in each Dockerfile. If a pip-only package becomes necessary, install it into a venv created with `--system-site-packages`.

## Dataset

The project uses KITTI raw drive `2011_09_26_drive_0005_sync`, "synced + rectified" (city, with moving and parked cars). This drive ships with tracklet labels.

### Files used per drive

| Path | Contents | Used for |
|---|---|---|
| `velodyne_points/data/*.bin` | float32 `[x, y, z, reflectance]` per point, one file per sweep | The point clouds |
| `velodyne_points/timestamps.txt` | One timestamp per sweep | Message header stamps, Δt, and the project's only time key |
| `oxts/data/*.txt` | GPS/IMU: lat, lon, alt, roll, pitch, yaw, velocities | Ego pose per frame |
| `calib_imu_to_velo.txt` (date folder) | Rigid transform IMU → Velodyne | Placing the LiDAR in the world frame |
| `tracklet_labels.xml` | Per-object boxes, class and pose per frame (velodyne frame) | Evaluation ground truth |

The drive also contains `oxts/timestamps.txt`, which is never used as a time key.

### Measured facts

Measured on the local data on Nov 5. The phases' exit criteria rely on these values.

| Fact | Value |
|---|---|
| Sweeps | 154, frame 0 = 123,397 points, mean ≈ 121.7k |
| Sensor period | 0.1027–0.1037 s (≈ 9.7 Hz, not 10 Hz). The 100 ms budget leaves ~3 ms of margin. |
| Drive | ≈ 15.8 s, ego speed 3.1–6.5 m/s |
| Ground truth | 15 tracklets, 717 poses: 9 Car, 3 Van, 2 Pedestrian, 1 Cyclist |
| Ground-truth coverage | Every pose has x ≥ 0.2 m (front half only). Bearings −86° to +57°. Nothing behind the car. |
| Parked objects (world frame, full pose math) | 10 tracklets (#3, #5–#11, #13, #14). Net speed 0.12–0.22 m/s. Max deviation from mean position: median 0.34 m, worst 1.09 m (van #3, 111 frames). |
| Moving objects | Lead van #0 ≈ 5.7 m/s and cyclist #1 ≈ 4.4 m/s (whole drive), pedestrians #2 and #12 ≈ 1.1 m/s, car #4 ≈ 2.7 m/s (11 frames) |

### Getting the data

`scripts/download_kitti.sh` downloads and unpacks the drive, calibration and tracklets into `./data`. KITTI may require registration and a license acceptance, so the script checks for the files and prints exact manual-download steps if fetching fails. `./data` is git-ignored. KITTI is licensed CC BY-NC-SA 3.0; credit it in the README.

### Statistics for the README

Computed by `scripts/dataset_stats.py`, saved to `output/`:

- Frames, duration, and mean sensor rate from the timestamps
- Points per frame (mean, min, max) and the share removed by each pipeline stage (the share is added in Phase 2)
- Range histogram (points vs distance) and reflectance histogram
- Ego speed over the drive from OXTS
- Ground-truth objects per frame, by class (Car, Pedestrian, Cyclist, Van…)

## System architecture

Two containers on compose's default bridge network exchange ROS 2 topics. Every message from the publisher carries the dataset timestamp of its sweep.

The tracker is the only stateful part. Everything a grader inspects flows out of it, live to the browser and as files to `./output`. There are two coordinate frames:

- `world`: ENU axes (x east, y north, z up, gravity-aligned) with the first OXTS position as its origin. This is the pykitti convention, frozen at the end of Phase 1;
- `velodyne`.

Clouds are published in `velodyne`. Boxes, arrows and labels are published in `world`.

### One core, two runners

- **ROS-free core.** The data layer, pose math, perception and tracker are plain numpy/scipy with no ROS imports. The ROS node only converts messages, pairs cloud and odom by stamp, publishes, and writes files.
- **Offline runner.** It calls the same core directly over all frames inside the processor image, so there is no third container. Evaluation, ablations and the sweep use it. It writes to a separate output folder and never overwrites live outputs.
- **Shared code.** The publisher, the processor and the scripts all need the data layer and the pose math. One copy goes into both images, so the riskiest math can't drift between them, and each image still runs on its own.
- **Time key.** Every cloud, odom message, TF, CSV row and evaluation match is keyed by frame index plus `stamp_ns` from `velodyne_points/timestamps.txt`.
- **Determinism.** RANSAC is seeded per frame from a fixed seed and `stamp_ns`, so a dropped frame doesn't shift the random stream. `metrics.json` records the seed and the parameters.

### Container A: publisher

Node `kitti_lidar_publisher` replays the drive as a live sensor. Each tick publishes one sweep, the matching ego pose, and the `world → velodyne` transform, all with the same dataset timestamp.

**Published:**

| Topic | Type | QoS | Contents |
|---|---|---|---|
| `/sensor/lidar/raw_points` | `sensor_msgs/msg/PointCloud2` | Sensor data (best effort, depth 5) | Fields x, y, z, intensity (float32); `frame_id: velodyne` |
| `/sensor/ego/odom` | `nav_msgs/msg/Odometry` | Reliable, depth 10 | `frame_id: world`, `child_frame_id: velodyne`. Pose of the velodyne frame in world. Twist in the velodyne frame: OXTS (ve, vn, vu) rotated into velodyne plus the lever-arm term ω × r, and OXTS (wx, wy, wz) rotated into velodyne |
| `/tf` | `tf2_msgs/msg/TFMessage` | Default | `world → velodyne` per frame |

**Logic:**

1. On start, load and validate the file list, `timestamps.txt`, all OXTS rows and the IMU → Velodyne calibration through the shared data layer. Any inconsistency stops the node before it publishes.
2. Convert OXTS to poses the pykitti way: Mercator projection of lat/lon with the scale from the first latitude, translated so the first OXTS position is the origin, and absolute roll/pitch/yaw (`R = Rz(yaw)·Ry(pitch)·Rx(roll)`). The world axes are therefore ENU. Compute `T_world_velo = T_world_imu · inverse(T_velo_imu)` for all frames at startup. The world origin is OXTS frame 0 whatever `start_frame` is.
3. A ROS timer fires at `rate_hz` (default 10, wall clock). Each tick reads the next `.bin` with `np.fromfile(..., dtype=np.float32).reshape(-1, 4)`. The stamps keep the sensor's own ≈ 9.7 Hz spacing.
4. Build the cloud with `sensor_msgs_py.point_cloud2.create_cloud` and set `header.stamp` from the dataset timestamp. Publish the TF, then the odometry, then the cloud, all with the same stamp.
5. At the end, loop to frame 0 if `loop: true`. Timestamps then jump backward, which the processor detects and uses to reset its tracker.

**Parameters:** `data_dir`, `drive`, `rate_hz`, `loop`, `start_frame`, `end_frame`.

Velocities stay correct at any rate because Δt comes from stamps.

### Container B: processor and tracker

Node `obstacle_tracker_node` turns each sweep into tracked obstacles with world-frame velocities in under 100 ms per frame. It subscribes to `/sensor/lidar/raw_points` and `/sensor/ego/odom` and pairs them by header stamp.

**Pipeline per frame:**

1. **Reset check.** If the stamp went backward, clear all tracks and increment `pass`. This handles the dataset loop. Phase 3 decides how a large forward gap is handled: the v2 rule of resetting on gaps over 1 s would also fire after a processing stall.
2. **Region of interest.** Keep x in [-40, 40] m, y in [-20, 20] m and z in [-3, 2] m. Drop the box around the ego car (`|x| < 2.5 m`, `|y| < 1.5 m`).
3. **Voxel downsample.** Keep one point per 0.15 m voxel by rounding to a grid and using `np.unique`. This cuts about 120k points to roughly 15–25k.
4. **Ground removal (RANSAC, seeded).** Fit a plane from 3 random points, 100 iterations. Count inliers within 0.2 m and accept only near-horizontal planes (`|normal z| > 0.9`, about 25.8° of tilt or less). Remove the inliers of the best plane.
5. **Clustering.** Run DBSCAN on the (x, y) coordinates only, with eps 0.6 m and min_samples 8. Clustering in 2D is faster and avoids splitting objects vertically.
6. **Cluster filtering.** Keep clusters with at least 10 points, length ≤ 7 m, width ≤ 3 m and height between 0.3 and 3 m. This removes walls, buildings and vegetation.
7. **Detections.** For each cluster, compute the box center (more stable than the mean of points) and the axis-aligned extents. Transform centers to the world frame with the paired ego pose, unless `ego_compensation` is off.
8. **Prediction.** Advance each track's Kalman filter by Δt from the header stamps.
9. **Association.** Build a cost matrix of distances between predicted track positions and detections. Set costs above a 2.0 m gate to infinity, then solve with `linear_sum_assignment`. With `association: greedy`, use nearest neighbor instead (ablation only).
10. **Update and lifecycle.** Matched tracks get a Kalman update. Unmatched detections start tentative tracks. A track is confirmed after 3 hits and deleted after 5 consecutive misses.
11. **Motion state.** A confirmed track with filtered speed above 0.7 m/s is "moving"; otherwise it is "static".
12. **Publish and log.** Send markers and the filtered cloud, then append confirmed tracks to `tracks.csv`.

**Kalman filter:** the state is `[x, y, vx, vy]` in the world frame, with a constant-velocity motion model and measurement `[x, y]`. Process noise `q_accel` and measurement noise `r_pos` are parameters. Velocity comes from the filter state, not from raw centroid differences, so partial views don't make arrows jump.

**`tracks.csv` contract** (frozen at the end of Phase 3): `pass, frame, stamp_ns, track_id, x, y, vx, vy, speed, state, l, w, h`. World frame, confirmed tracks only. `pass` increments on every dataset loop.

**Published:**

| Topic | Type | Frame | Contents |
|---|---|---|---|
| `/perception/filtered_points` | `PointCloud2` | velodyne | Points after ROI, downsampling and ground removal |
| `/perception/bounding_boxes` | `MarkerArray` | world | `LINE_LIST` wireframe per confirmed track, colored by state |
| `/perception/obstacle_vectors` | `MarkerArray` | world | `ARROW` from box center along (vx, vy), length = speed × 0.5 s |
| `/perception/track_labels` | `MarkerArray` | world | `TEXT_VIEW_FACING`: track ID, state as text, and speed in km/h |

Every `MarkerArray` begins with a `DELETEALL` marker, so deleted tracks don't leave ghosts. Per-frame processing time and the dropped-frame count go into `metrics.json`.

**Parameters** (defaults in `config/tracker.yaml`, read by both the node and the offline runner):

- `roi_*`
- `ego_box`
- `voxel_size: 0.15`
- `ransac_iters: 100`
- `ransac_dist: 0.2`
- `seed`
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
- `ego_compensation: true`
- `association: hungarian`
- `report_every_n: 50`
- `output_dir: /output`

## Visualization and outputs

Graders get a live 3D view in their browser and a set of static files that need no ROS at all.

### Live view (Foxglove)

`foxglove_bridge` runs inside Container B and exposes `ws://localhost:8765`. Open Foxglove in a browser, connect to that address, and import the committed layout `viz/foxglove_layout.json`. The layout sets the display frame to `world`, so the ego car visibly drives through a static scene and parked cars stay still. It shows:

- the raw and filtered clouds;
- boxes, arrows and labels;
- a plot panel of ego speed.

The viewer is the **Foxglove web app** (https://app.foxglove.dev). Nothing is installed besides Docker. The bridge is only a WebSocket server; the browser is the client.

Steps for reviewers (also in the README):

1. Run `docker compose up --build` and wait for the processor to log that `foxglove_bridge` advertised its channels.
2. Open https://app.foxglove.dev in **Chrome, Edge or Firefox** and sign in. A free Foxglove account is required.
3. Choose Open connection → Foxglove WebSocket → `ws://localhost:8765`.
4. Import the layout `viz/foxglove_layout.json` (Layout menu → Import from file).

Browser caveats to document under troubleshooting:

- Safari and Brave (with Shields on) can block a secure page from opening `ws://localhost`. Use Chrome, Edge or Firefox, or turn Shields off for the site.
- If the 3D panel is empty, set the display frame to `world`, or to `velodyne` before Phase 1 lands.
- Reviewers who don't want an account can use the Foxglove desktop app against the same address.
- The README ships sample screenshots in `docs/`, so the result is visible even if the live view can't be opened.

### Static outputs

Written to the mapped `./output` volume every `report_every_n` frames and on shutdown. Offline runs write to a separate subfolder.

| File | Contents | Phase |
|---|---|---|
| `dataset_stats.png` | Histograms and per-frame counts from the stats script | 1 (per-stage share in 2) |
| `dataset_stats.json` | The same statistics as numbers (the README quotes it; the checked artifact) | 1 (per-stage share in 2) |
| `metrics.json` | Frames expected, processed and dropped; mean and p95 processing time; track counts; seed and parameters | 2 (track counts in 3) |
| `tracks.csv` | Every confirmed track state per frame (contract above) | 3 |
| `obstacle_tracking_report.png` | Top-down world-frame map: the ego path, each confirmed track's trajectory colored by ID, and velocity arrows at the last position. Static tracks are grey and named in a legend. | 3 |
| `evaluation.md` | Metrics table and the no-ego ablation from the evaluation script | 4 |

The report is written periodically, not only on exit, because `docker compose down` can kill the node before a clean shutdown handler runs. The node also catches SIGTERM and writes one final report.

## Evaluation against ground truth

`scripts/evaluate.py` compares the offline runner's `tracks.csv` with `tracklet_labels.xml` and writes `output/evaluation.md`. It turns "it looks right in the viewer" into numbers you can defend, and two runs give identical output.

### Method

1. Parse tracklets into per-frame boxes (center, size, yaw, class) in the velodyne frame. Map frames to stamps with `velodyne_points/timestamps.txt`.
2. Transform ground-truth centers to the world frame with the same ego poses the publisher uses.
3. Compute ground-truth velocity by differencing world-frame centers over consecutive frames, smoothed over 3 frames.
4. Keep only ground truth **and tracks** inside the evaluation region: the processing ROI ∩ the front half (x ≥ 0 in the velodyne frame). KITTI tracklets exist only in front of the car, so without this filter correct tracks behind it would count as false positives.
5. Per frame, match confirmed tracks to ground truth with Hungarian assignment on center distance, using a 2.0 m gate. This is the same matching function Phase 2 uses for detection recall.

### Metrics

Overall and per class. Per-class rows always show object counts; Pedestrian (n = 2) and Cyclist (n = 1) are flagged as small samples.

| Metric | Definition |
|---|---|
| Recall | Matched ground-truth objects ÷ all ground-truth objects in the evaluation region |
| Precision | Matched tracks ÷ all confirmed tracks in the evaluation region |
| Center error | Mean distance between matched track and ground-truth centers (m) |
| Velocity RMSE | RMSE of (vx, vy) for matched pairs (m/s), reported separately for moving and static objects |
| Static accuracy | Share of truly static objects (GT speed < 0.5 m/s) that the tracker labels "static" |
| ID switches | Times a ground-truth object's matched track ID changes |

### Ablations

- **No ego compensation (core, Phase 4).** Run with `ego_compensation: false` and report static-object velocity RMSE side by side with the compensated run. The jump in that error is the strongest single result in the project.
- **Greedy matching (optional, Phase 6).** Run with `association: greedy`. The Phase 6 spec decides whether "greedy instead of Hungarian plus Kalman" also removes the Kalman filter.

### DBSCAN sweep (optional, Phase 6)

Run a sweep over `eps` in {0.4, 0.5, 0.6, 0.8} and `min_samples` in {5, 8, 12}. Report recall, precision and runtime for each of the 12 settings, and use the results to justify the chosen defaults.

## Repository layout and reproducibility

A grader runs three commands: download the data, `docker compose up`, then open Foxglove or the `output/` folder.

```text
bgracing-lidar-tracking/
├── docker-compose.yml
├── README.md
├── core/                       # shared ROS-free code, copied into both images (Phase 1)
│   ├── kitti.py                # data layer: sweeps, stamps, OXTS, calibration, tracklets
│   └── poses.py                # OXTS → world poses, transforms, ego twist
├── publisher/
│   ├── Dockerfile.publisher
│   └── kitti_lidar_publisher.py
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
│   ├── verify.sh               # build + publisher smoke import + pytest (Phase 1)
│   ├── dataset_stats.py
│   └── evaluate.py
├── viz/
│   └── foxglove_layout.json
├── tests/                      # pytest: data layer, pose math, RANSAC, perception, tracker, metrics
├── docs/                       # this spec, sample outputs (docs/samples/) and screenshots for reviewers
├── data/                       # git-ignored
└── output/                     # git-ignored
```

The layout is indicative. Each phase's plan fixes the exact files, including where the shared data layer and the offline runner live.

### docker-compose.yml (current)

```yaml
services:
  publisher:
    build: { context: ., dockerfile: publisher/Dockerfile.publisher }
    init: true
    environment: [ROS_DOMAIN_ID=42]
    volumes: ["./data:/data:ro", "./config:/config:ro"]
  processor:
    build: { context: ., dockerfile: processor/Dockerfile.processor }
    init: true
    environment: [ROS_DOMAIN_ID=42]
    volumes: ["./data:/data:ro", "./config:/config:ro", "./output:/output"]
    ports: ["127.0.0.1:8765:8765"]
    stop_signal: SIGINT  # ros2 launch shuts its nodes down cleanly on SIGINT, not on SIGTERM
    depends_on: [publisher]
```

No `network_mode` is set, so both services share compose's default bridge network. Scripts run directly with `python3` after sourcing ROS. The launch file starts them with `ExecuteProcess`, which avoids building colcon packages. Both images copy `core/` and set `PYTHONPATH=/app`. Since Phase 1 the bridge port is bound to `127.0.0.1` only, because `foxglove_bridge` has no authentication.

### Commands for the README

Final commands are fixed in Phase 5. Offline commands use `--no-deps` so they don't start the publisher.

```bash
./scripts/download_kitti.sh                     # fetch the drive into ./data
docker compose up --build                       # run the pipeline
# open https://app.foxglove.dev → Open connection → ws://localhost:8765
docker compose run --rm --no-deps processor python3 /app/scripts/dataset_stats.py
docker compose run --rm --no-deps processor python3 /app/scripts/evaluate.py
./scripts/verify.sh                             # build + unit tests (author's verify command)
```

### Reproducibility checks (owned by Phase 5)

- [ ] A fresh clone in an empty directory runs end to end with only the README commands, and `evaluation.md` matches the committed sample
- [ ] Both images build for amd64 and arm64
- [ ] `output/` files are readable and deletable by the host user (`ROS_HOME=/tmp/ros` is set; add `user: "${UID}:${GID}"` if needed)
- [ ] Sample outputs and screenshots are committed to `docs/`, so reviewers see results without running anything
- [ ] The base image is pinned by digest in both Dockerfiles (done in Phase 0)
- [ ] Both containers report Ubuntu 24.04 and Python 3.12 (`cat /etc/os-release`, `python3 --version`)
- [ ] The GitHub repo is public, or shared with the reviewers, before submitting

## Phases

### How phases are worked

- **One phase at a time, each in its own folder.** When a phase starts:
  1. `/write-feature-spec` creates `docs/specs/phase-N-<slug>/FEATURE_SPEC.md`.
  2. `/architecture-plan` creates `PLAN.md`.
  3. `/loop-engineering` builds the phase and writes `REVIEW.md` with `EVIDENCE/`.
  
  These folders are local working notes and are git-ignored.
- **The folders are:**
  - `phase-1-foundations`
  - `phase-2-detection`
  - `phase-3-tracking`
  - `phase-4-evaluation`
  - `phase-5-ship`
  - `phase-6-optional`
- **This spec fixes each phase's end goal, scope and exit criteria.** The phase spec adds detail. It also sets every threshold marked **T?** before that phase's code starts, never after seeing results. The measurements and candidate values listed here are input for those decisions.
- **A phase is done when:**
  - the verify command passes;
  - every exit criterion has evidence;
  - `change-reviewer` has reviewed the diff;
  - `docker compose up --build` still runs the pipeline.
- **Security re-check.** Phase 1 adds an apt dependency (pytest) and parses downloaded XML, so its review was triggered. It bound the bridge port to `127.0.0.1`, pinned the SHA-256 of the extracted inputs, and added an XML entity-bomb test. Later phases that add dependencies or change exposure re-check.

```text
P0 Plumbing ─► P1 Foundations ─► P2 Detection ─► P3 Tracking ─► P4 Core evaluation ─► P5 Ship-ready ─► P6 Optional evidence
  (done)        (no tracker)      (velodyne)      (world frame)   (never cut)            (submittable)    (each item cuttable)
```

### Phase 0: Plumbing (done)

**Built:**

- Both images build from a pinned base.
- Compose starts both services.
- The publisher replays raw clouds at `rate_hz` with dataset stamps and loops.
- The processor runs `foxglove_bridge` and a probe that logs received sweeps.
- The download script fetches the drive, calibration and tracklets.

**Not done:** the evidence for this phase hasn't been recorded yet. Phase 1 records it.

### Phase 1: Foundations

**End goal:** every later phase can be verified automatically, and the world frame is proven correct against ground truth before anything depends on it.

**In scope**

- A unit-test harness that runs inside the processor image, without ROS and without the publisher. The verify command grows from `docker compose build` to `scripts/verify.sh`: build, a publisher smoke import, and the unit tests.
- One shared, ROS-free KITTI data layer (`core/`) used by both containers and the scripts. It loads sweeps, timestamps, OXTS, IMU → Velodyne calibration and tracklets.
- Pose math (see Container A, logic step 2).
- The publisher adds `/sensor/ego/odom` and `/tf` (world → velodyne), with the same stamp as each cloud.
- `dataset_stats` (PNG + JSON), every part except the per-stage share, with a committed sample in `docs/samples/` and a README Dataset section.
- Binding the bridge port to `127.0.0.1` (moved forward from Phase 5).
- Recording Phase 0's evidence.

**Exit criteria**

- E1.1: verify runs the unit tests and passes. Tests cover:
  - timestamp parsing (frame 0 = `1317042272335337762` ns, the strings read as UTC);
  - sweep loading (frame 0 → 123,397 × 4);
  - tracklet parsing (15 tracklets, 717 poses, 9/3/2/1 by class; center z = tz + h/2);
  - pose math on synthetic poses;
  - the calibration direction: the velodyne origin sits at (+0.81, −0.31, +0.80) ± 0.01 m in the IMU frame, which matches KITTI's 1.73 m Velodyne and 0.93 m IMU heights. The parked check below can't detect an omitted calibration (worst 1.13 m), so this test is required;
  - the SHA-256 of the three small input files.
- E1.2 **Pose check.** All distances are 2D (x, y) box centers.
  - *Given* the 10 parked tracklets (#3, #5–#11, #13, #14: world net speed < 0.5 m/s, a fixed fixture list), *when* their centers are moved into the world frame, *then* each stays within **1.2 m** of its mean position. Measured: median 0.35 m, worst 1.088 m; an inverted calibration gives 1.34 m.
  - *Negative control:* in the velodyne frame, every one of them deviates ≥ 3.6 m (measured minimum 4.9 m), so the check can fail.
  - *Moving control:* in the world frame, van #0 reads 5.74 ± 0.3 m/s and cyclist #1 reads 4.38 ± 0.3 m/s, so a transform that freezes everything can't pass.
- E1.3: live, `/sensor/ego/odom` and `/tf` publish once per sweep with stamps equal to the cloud's, and the odom pose equals the data layer's pose within 1 mm (probe log over at least one loop). Foxglove with display frame `world` shows the ego moving through a still scene (screenshot).
- E1.4: `dataset_stats` runs without the publisher. Two runs produce byte-identical output, and `dataset_stats.json` contains every listed quantity.
- E1.5: Phase 0 evidence is recorded. The rate derived from cloud header stamps is 9.68 ± 0.05 Hz (the wall-clock publish rate, ≈ `rate_hz`, is recorded but not gated), `frame_id` is `velodyne`, and a received cloud carries the stamp from line 1 of `velodyne_points/timestamps.txt`.

**Failure paths owned:** a missing data folder gives an error that names the download script. A sweep/timestamp count mismatch, a truncated `.bin` (0 bytes or not a multiple of 16) or a missing or malformed OXTS row each give a clear error. A missing tracklet file means stats still run and report that ground truth is missing.

**Frozen at exit:** the world-frame convention (ENU, origin at the first OXTS position), the time key, and the shared data layer's behavior.

**Reviewer-visible if the project stopped here:** the dataset statistics (a hard requirement) in the README and `docs/samples/`, and an ego car driving through a static world in Foxglove.

**Resolved by the phase spec:** the seed and parameters go into `metrics.json` from Phase 2, because nothing in Phase 1 is seeded.

### Phase 2: Detection

**End goal:** each sweep becomes obstacle detections, measured against ground truth and within the time budget, identically offline and live.

**In scope**

- The ROS-free perception core: pipeline steps 2–7 in the velodyne frame, with seeded RANSAC.
- The offline runner (detections).
- The live node replaces the probe. It publishes the filtered cloud and box markers, and every `MarkerArray` starts with `DELETEALL`.
- Per-frame timing and a dropped-frame count go into `metrics.json`.
- Per-stage point counts are added to the dataset stats.
- A ground-truth matching function (Hungarian assignment with a 2.0 m gate), reused by Phase 4.

**Exit criteria**

- E2.1 **Unit tests on synthetic clouds:**
  - flat ground plus one car-sized box gives one detection at the right center;
  - ground tilted 20° is removed, while ground tilted 30° is not accepted as ground;
  - an empty cloud, or one with fewer than 8 points, gives zero detections and no crash;
  - a 20 m wall is filtered out;
  - two boxes 0.5 m apart behave as the test documents (eps 0.6 merges them).
- E2.2 **Offline recall** against ground truth: cars and vans, front half of the ROI, velodyne frame, 2.0 m gate, all 154 frames. Report per class with object counts, against **T?** (candidates: 0.7 / 0.8 / 0.9).
- E2.3: two offline runs produce identical detections.
- E2.4: offline p95 time per frame ≤ **T?** on the dev machine (this spec: 100 ms; sensor period ≈ 103 ms). A live 10 Hz run reports its dropped-frame count.
- E2.5: live, every `MarkerArray` begins with `DELETEALL`, and the box count equals the detection count for that stamp. A Foxglove screenshot shows the boxes.

**Failure paths owned:** zero points after the ROI, zero ground inliers, fewer than 3 points for RANSAC, and the processor starting before the publisher or before DDS discovery completes.

**Frozen at exit:**

- The detection record: center, l/w/h and point count, in the velodyne frame.
- The per-stage counts.
- How the offline runner is invoked and where it writes.

**Reviewer-visible if the project stopped here:** boxes around obstacles in Foxglove, plus a recall number.

### Phase 3: Tracking

**End goal:** detections become persistent tracks with world-frame velocities. Parked objects read near zero, moving objects read their speed, and live and offline runs agree.

**In scope**

- Pipeline steps 1 and 7–12: world-frame detections, Kalman filter, Hungarian association, lifecycle, reset and motion state.
- Cloud/odom pairing by stamp.
- The `ego_compensation` and `association` switches.
- Arrows, plus labels that carry the state as text, and boxes colored by state.
- Output files:
  - `tracks.csv`;
  - `metrics.json` with track counts;
  - the report PNG, with static tracks grey **and** named in a legend;
  - all written periodically and on SIGTERM.
- `viz/foxglove_layout.json`.
- The offline runner produces tracks.
- A live-vs-offline parity check.

**Exit criteria**

- E3.1 **Unit tests on synthetic tracks:**
  - a constant-velocity target's speed converges to within **T?**;
  - two crossing targets keep their IDs;
  - a track confirms on the 3rd hit and is deleted on the 5th miss;
  - a backward stamp clears all tracks;
  - a duplicate stamp (Δt = 0) and an empty detection set are handled safely.
- E3.2 **Parked objects:** offline, the tracks matched to the 10 parked tracklets have median speed ≤ **T?**. Static ground truth is defined as < 0.5 m/s. *Negative control:* with `ego_compensation: false`, the same tracks read roughly the ego speed (3.1–6.5 m/s).
- E3.3 **Moving objects:** speed error for the lead van (≈ 5.7 m/s) and the cyclist (≈ 4.4 m/s) is ≤ **T?**.
- E3.4 **Parity:** a live run with loop off and zero dropped frames produces the same `tracks.csv` as the offline run, within **T?**.
- E3.5 **Loop:** after the wrap from frame 153 to 0, the first frame of the new pass contains no track from the previous pass, and `pass` increments.
- E3.6 **Shutdown:** `docker compose down` mid-run leaves complete, parseable `tracks.csv`, `metrics.json` and PNG within compose's stop timeout.
- E3.7: the layout file loads in Foxglove with display frame `world` and shows the clouds, boxes, arrows, labels and an ego-speed plot (screenshot).

**Failure paths owned:** a cloud with no matching odom stamp, zero detections, and a false reset after a processing stall.

**Frozen at exit:** the `tracks.csv` schema, the `metrics.json` fields and the config switches.

**Reviewer-visible if the project stopped here:** the core result. Parked cars read near 0, moving objects show their speed, and the PNG map shows tracks in the world frame.

**Open for the phase spec:** whether a forward stamp gap resets tracks or only ages them through misses.

### Phase 4: Core evaluation (never cut)

**End goal:** turn tracks into defensible numbers that a reviewer regenerates with one command and gets identically.

**In scope**

- `evaluate.py` per [Evaluation against ground truth](#evaluation-against-ground-truth), scoring the offline runner's tracks.
- The no-ego-compensation ablation.
- `evaluation.md`.
- Sample outputs committed to `docs/`: the report PNG, the stats PNG and `evaluation.md`.

**Exit criteria**

- E4.1: one command, with no publisher and no Foxglove, regenerates `evaluation.md`. Two runs are byte-identical.
- E4.2: every metric is finite, matched + missed equals the ground-truth count in the evaluation region, and per-class rows show object counts.
- E4.3: the metric functions are unit-tested on hand-built cases: a perfect match, one miss, one false track and one ID switch.
- E4.4: the no-ego ablation row reports static-object velocity RMSE side by side with the compensated run.
- E4.5: the sample outputs in `docs/` match a fresh regeneration.

**Failure paths owned:** a missing tracklet file gives a clear error. An empty `tracks.csv` is reported as zero recall, not a crash.

**Reviewer-visible if the project stopped here:** the results table, including the ablation that shows why ego compensation matters.

### Phase 5: Ship-ready (submittable)

**End goal:** a reviewer on an unknown machine goes from fresh clone to the same results using only the README, and the project could be submitted as-is.

**In scope**

- The README per [README outline](#readme-outline).
- Screenshots.
- The [reproducibility checks](#reproducibility-checks-owned-by-phase-5).
- The Foxglove troubleshooting section.
- The repo made public or shared.
- Form submission (the owner's action).

**Exit criteria**

- E5.1: from a fresh clone in an empty directory, using only the README's commands, the expected output files exist and `evaluation.md` matches the committed sample.
- E5.2: both images build for amd64 and arm64. Inside each, `os-release` reports Ubuntu 24.04 and `python3 --version` reports 3.12.
- E5.3: the README contains every outline section, and every command in it was run during E5.1.
- E5.4: after a run, the host user can read and delete the files in `output/`.
- E5.5: the repo can be opened by someone who isn't the owner.

**Reviewer-visible:** the complete submission.

**Open for the phase spec:** a GIF or screenshots only. (Binding the bridge to `127.0.0.1` was done in Phase 1.)

### Phase 6: Optional evidence (each item independent)

**End goal:** add evidence for parameter and design choices without risking the submittable state.

**Items, in cut order (cut the first one first):**

1. The [DBSCAN sweep](#dbscan-sweep-optional-phase-6): 12 settings, each with recall, precision and runtime.
2. The greedy-matching ablation. Its spec defines what "greedy" removes.

**Exit, per item:** its table appears in the README results and the evaluation output, and E5.1 passes again afterwards.

### Failure states by phase

| State | Where it occurs | Phase |
|---|---|---|
| No data downloaded | Publisher, scripts | 1: clear error naming the download script |
| Partial data (truncated `.bin`, missing OXTS row, no tracklets) | Data layer | 1 (stats degrade gracefully), 4 (evaluation errors clearly) |
| Processor up before publisher, or DDS discovery late | Live node | 2: no crash; missed frames show in the dropped count |
| Empty frame or zero clusters | Perception, tracker | 2, 3 |
| Processor falls behind | Live node | 2 (dropped count), 3 (no false reset) |
| Dataset loop (stamps go backward) | Tracker, `tracks.csv` | 3: reset, `pass` increments |
| `docker compose down` mid-run | Output files | 3: complete files on SIGTERM |
| Foxglove can't connect | Reviewer | 5: troubleshooting section plus committed screenshots |

### Schedule (indicative)

| Phase | Slot |
|---|---|
| 1 Foundations | Fri 6/11 |
| 2 Detection | Sat 7/11 |
| 3 Tracking | Sun 8/11 – Mon 9/11 |
| 4 Core evaluation | Tue 10/11 |
| 5 Ship-ready | Wed 11/11 → internal target Thu 12/11 |
| 6 Optional evidence | After 12/11, before the 21/11 deadline |

### If you fall behind

Cut only Phase 6 items: first the DBSCAN sweep, then the greedy ablation. Never cut anything in Phases 1–5. Ego-motion compensation, the core evaluation and the no-ego ablation are what make the project stand out.

## Risks, stretch goals and README

### Risks

| Risk | Sign | Fallback | Phase |
|---|---|---|---|
| KITTI download needs a login | Script can't fetch files | Manual steps in the README; check whether the license allows committing a few sample frames with attribution | 1, 5 |
| Ego pose looks wrong | Parked ground truth drifts in the world frame (E1.2 fails) | Plot the OXTS path alone; check the calibration direction (IMU → velo vs velo → IMU) | 1 |
| DDS discovery fails between containers | `ros2 topic list` in B misses A's topics | Same `ROS_DOMAIN_ID`; try `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`; last resort `network_mode: host` on Linux | 1, 2 |
| Processing exceeds the budget | p95 above budget, dropped frames | Larger voxels, tighter ROI, fewer RANSAC iterations; lower `rate_hz`. The budget (100 ms) is only ~3 ms under the sensor period. | 2 |
| Walls or bushes tracked as obstacles | Large or oddly shaped boxes | Tighten size filters; require a minimum height | 2 |
| False reset after a stall | Tracks vanish mid-drive; `pass` increments without a loop | Reset only on backward stamps; let misses age out stale tracks | 3 |
| Velocities are noisy | Arrows flicker | Raise `r_pos`, require more hits before confirming, use box center rather than mean | 3 |
| Live and offline disagree | Parity check (E3.4) fails | Check dropped frames, stamp pairing and seeding before trusting either | 3 |
| Precision looks wrong | Many false positives beside or behind the car | Evaluation region limited to the front half, where ground truth exists | 4 |
| Foxglove web app can't reach the bridge | Connection fails or the 3D panel is empty | Use Chrome/Edge/Firefox; disable Brave Shields; confirm `ws://localhost:8765` is mapped (`docker compose ps`); fall back to the desktop app or the committed screenshots | 5 |

### Stretch goals

Only after Phase 6:

- **Cluster classifier.** Train a small scikit-learn model on geometric features (extents, point count, height profile) using tracklet classes as labels. It adds a learned component you can measure.
- **Formula Student variant.** If the team's provided LiDAR sample contains cones, swap the cluster filter for cone-sized clusters and estimate track boundaries and a centerline. It sits closest to BGRacing's actual domain.
- **Time-to-collision warnings** for tracks on a path crossing the ego car's.
- **Oriented boxes** from a PCA of each cluster's (x, y) points.

### README outline

1. A one-paragraph summary plus a screenshot (or GIF) of the Foxglove view
2. An architecture diagram of containers, nodes, topics and frames
3. Dataset: the drive used, file formats, statistics plots
4. Pipeline: each stage with its parameters and why those values
5. Results: the evaluation table, the no-ego ablation, the PNG report, plus the DBSCAN sweep and greedy ablation if Phase 6 delivered them
6. How to run: the exact commands, expected outputs, troubleshooting
7. Limitations and next steps, stated honestly
8. Credits: KITTI citation and license

## Decisions and open items

**Decided Nov 5, 2026.** The full rationale is in `docs/specs/PRODUCT-DIRECTION.md`, which is a local file.

| Decision | Rejected alternatives |
|---|---|
| Six phases, foundations first | Detection first; keeping the original five phases with rewritten criteria |
| Reported numbers come from the offline runner plus one live parity check | Every number from a live ROS replay |
| Each phase's spec sets its own thresholds before its code starts | Setting defaults now as gates; recording a baseline only |
| RViz2 config dropped | Keeping it as the last optional item |
| No-ego ablation is core | Keeping it second in the cut order |

**Corrections from the measured data:**

- The evaluation region is limited to the front half.
- The velodyne timestamps are the only time key.
- `tracks.csv` gets a `pass` column.
- RANSAC is seeded.
- The report PNG moves to Phase 3, because it's a world-frame map.
- The sensor rate is ≈ 9.7 Hz.

**Open:**

- Does the form allow resubmission? If yes, submit at the end of Phase 5 and again after Phase 6. If no, submit once, after Phase 6 or in the deadline buffer.
- The remaining open items are listed in their phases: Phase 3 (forward gaps), Phase 5 (GIF), Phase 6 (greedy definition).

**Decided in the Phase 1 spec (Nov 5, 2026):** the world frame is ENU with the first OXTS position as origin (rejected: the first full IMU pose); the E1.2 bar is 1.2 m (rejected: 1.5 m); the publisher keeps its 10 Hz timer and E1.5 checks the stamp-derived rate (rejected: pacing by stamps); the bridge port is bound to `127.0.0.1` now (rejected: waiting for Phase 5). Details are in `docs/specs/phase-1-foundations/FEATURE_SPEC.md`, which is a local file.
