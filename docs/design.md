# Design notes

Why the pipeline is built the way it is, and which alternatives were rejected. The [README](../README.md) describes what it does and how to run it. The numbers quoted here come from the committed samples in [`samples/`](samples/) or from the measurements stated in the README.

## The assignment, and where each requirement is met

| Requirement | Where |
|---|---|
| Two Docker containers with ROS 2 nodes: one acquires and publishes, one reads and processes | `publisher/` (KITTI drive → `/sensor/lidar/raw_points`, `/sensor/ego/odom`, `/tf`); `processor/` (detection, tracking, Foxglove bridge, output files); `docker-compose.yml` |
| Ubuntu 24.04, ROS 2 Jazzy, Python 3.12 | both images build from `ros:jazzy-ros-base`, pinned by digest; CI prints the OS and Python of each |
| README with dataset properties and statistics, task, pipeline blocks, results, how to run | [README](../README.md): Dataset, Pipeline, Results, How to run |
| Reproducible by the reviewers | `scripts/download_kitti.sh`, then the README commands; committed samples to compare with; CI runs the whole flow on a clean x86_64 runner |

## Key decisions

| Decision | Rejected | Why |
|---|---|---|
| Velocities in a world frame (ENU, origin at the first GPS/IMU position, pykitti's convention), with the ego pose streamed by the publisher | Velocities in the sensor frame | In the sensor frame every parked car moves at minus the ego speed. The world frame makes "parked reads about 0 m/s" a testable claim. |
| Δt from the dataset timestamps carried in every message header | Wall clock or the publisher's timer | The sensor runs at 9.68 Hz, not 10 Hz, and live delivery jitters; stamps make velocities exact and repeatable. |
| Reported numbers come from an offline runner over the same ROS-free code, plus one live-versus-offline parity check | Numbers from a live ROS replay | A live replay drops about 2 % of sweeps and is never the same twice. The parity check shows the live node gives byte-identical tracks for the sweeps it processed. |
| The pose math was proven before any detection code: parked ground truth must stay within 1.2 m in the world frame, and the check must fail without the ego pose | Building detection first | A wrong pose would otherwise look like a bad tracker. |
| Seeded RANSAC ground plane in numpy, refined by local 10 m tiles | A fixed height cut; open3d | A fixed cut fails on slopes and leaves curbs. One plane can't follow 80 m of road, and the leftover band of ground joined cars to walls in the clustering (most of the early misses). Seeding makes runs repeat. |
| DBSCAN on (x, y) only | 3D clustering | Faster, and it doesn't split an object vertically. |
| Size filter on the oriented footprint (minimum-area rectangle); boxes stay axis-aligned | Axis-aligned width limit; oriented boxes everywhere | The car turns about 75° during the drive, so an axis-aligned limit rejected 37–46 % of the real vans even with perfect segmentation. |
| Constant-velocity Kalman filter with Hungarian association, gated at 2.0 m; parameters fixed before any tracking code ran | Greedy nearest neighbour; tuning the filter against the scores | Optimal one-to-one assignment in one call, and nothing is fitted to the numbers it is judged by. |
| A track is `moving` only if it is confidently faster than 0.7 m/s (speed − 2σ > 0.7 m/s) | The plain rule speed > 0.7 m/s | The plain rule labelled 197 of 417 tracks moving, mostly poles and walls whose box center slides as the car passes. The cost is stated: a walking pedestrian reads `static`. |
| Missing sweeps count as misses; only a backward timestamp (the dataset loop) resets the tracker | Resetting on long gaps | A processing stall would otherwise fake a loop, and live would stop matching offline. |
| Scoring only in the front half of the region of interest | The whole region | KITTI tracklets exist only in front of the car. |
| One pass/fail bar for the evaluation, set before its first run: removing ego compensation must make the static-object velocity error at least 3 times larger (measured: 8.19) | Bars on recall or precision | With about 74 tracks per sweep and at most 15 labelled objects, a precision bar would be arbitrary. Every other number is reported with "no bar by design". |
| Unlabelled tracks are judged only by physics (a fixed object must stay still once ego motion is removed) | Calling unmatched tracks false positives | The tracklets label only a few object types, so "unmatched" is not "wrong". |
| Foxglove in the browser, bridge bound to `127.0.0.1` | RViz2 in Docker | No X11 or GUI setup on any OS. The bridge has no authentication, so it listens on localhost only. |
| Dependencies from apt only, base images pinned by digest | pip requirements | Ubuntu 24.04 blocks system-wide pip; apt packages match ROS's Python. |

## How reproducibility is proven

- **Determinism.** Every random step is seeded, and the compared outputs (`evaluation.md` and the JSON samples) carry no timestamps or host names (`metrics.json`, which holds timing, is not compared). Two runs give byte-identical `evaluation.md` and JSON files. The README states which CPU architectures this was verified on.
- **One command per result.**
  - `evaluate.py` regenerates the evaluation;
  - `dataset_stats.py` regenerates the dataset statistics;
  - `run_offline.py` regenerates detections, tracks and speed scores.
- **`./scripts/verify.sh`** builds both images, smoke-tests the publisher, and runs the unit and real-data tests. The key checks (pose, speeds, ablation, live parity, sample comparison, submission check) each have a negative control showing they can fail. It also compares the committed samples with a fresh run, checks that the README quotes them correctly, and checks that the repository holds only the submission.
- **CI** (`.github/workflows/verify.yml`) runs the reviewer flow on a clean x86_64 runner:
  - download, build, `verify.sh`;
  - the sample comparison;
  - a headless live run and stop, checking the topics and the output files;
  - deleting `output/` as a normal user.

## Known risks

The README's Known gaps section lists what is not solved:
- some clutter tracks are labelled `moving`;
- buses and long trucks are dropped by the size filter;
- speed accuracy did not carry over to the second drive;
- tracks carry no class, and walking pedestrians read `static`;
- the evidence is two drives, scored in front of the car only, and the live run drops about 2 % of sweeps.

Axis-aligned boxes, and byte identity verified only on the stated architectures with unpinned apt packages, are listed there as smaller caveats.
