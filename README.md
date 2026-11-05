# LiDAR Obstacle Tracking

BGRacing Autonomous home assignment: a two-container ROS 2 Jazzy pipeline that replays a KITTI raw LiDAR drive, detects obstacles, and tracks them in a world frame. See [docs/spec.md](docs/spec.md) for the full spec.

> Work in progress (Phase 0: plumbing).

## Run

```bash
./scripts/download_kitti.sh                     # fetch the drive into ./data
docker compose up --build                       # run the pipeline
# Foxglove desktop app → Open connection → ws://localhost:8765
```

## Credits

Dataset: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/) (Geiger, Lenz, Stiller, Urtasun), licensed CC BY-NC-SA 3.0.
