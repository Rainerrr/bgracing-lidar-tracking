# LiDAR Obstacle Tracking

BGRacing Autonomous home assignment: a two-container ROS 2 Jazzy pipeline that replays a KITTI raw LiDAR drive, detects obstacles, and tracks them in a world frame. See [docs/spec.md](docs/spec.md) for the full spec.

> Work in progress (Phase 1: foundations). The publisher streams the point cloud, the ego odometry and the `world → velodyne` transform. Detection and tracking come in Phases 2–3.

## Run

```bash
./scripts/download_kitti.sh                     # fetch the drive into ./data
docker compose up --build                       # run the pipeline
```

Then open the Foxglove web app at <https://app.foxglove.dev> in Chrome, Edge or Firefox (a free account is required), choose Open connection → Foxglove WebSocket → `ws://localhost:8765`, and set the 3D panel's display frame to `world`. The ego car then drives through a still scene. Safari, and Brave with Shields on, can block the local connection.

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

## Verify

```bash
./scripts/verify.sh
```

This builds both images, checks the publisher image's imports, and runs the unit tests inside the processor image. It needs the drive in `./data`: the real-data tests fail (they never skip) without it.

## Credits

Dataset: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/) (Geiger, Lenz, Stiller, Urtasun), licensed CC BY-NC-SA 3.0.
