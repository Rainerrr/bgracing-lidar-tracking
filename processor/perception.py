"""Perception core, docs/spec.md steps 2-7 (velodyne frame). ROS-free: numpy, scipy, sklearn.

detect(points, stamp_ns, params) turns one sweep into axis-aligned detections. The live node and the
offline runner both call it, so a number can only differ between them if the input differs.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

STAGES = ("raw", "roi", "voxel", "non_ground", "clustered")
GROUND_OK, NO_PLANE, TOO_FEW = "ok", "no_plane", "too_few_points"

_OFFSET = 1 << 20  # voxel indices are shifted to be non-negative before packing 3 x 21 bits


@dataclass(frozen=True)
class Detection:
    """Axis-aligned box in the velodyne frame: center, extents along the velodyne axes, voxel count."""
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    n_points: int


@dataclass(frozen=True)
class FrameResult:
    detections: tuple
    stages: dict
    clusters_found: int
    clusters_kept: int
    ground_status: str
    filtered_xyz: np.ndarray  # (M, 3) float32, non-ground voxel centers


# ---------------------------------------------------------------- step 2: region of interest

def crop_roi(xyz, p):
    """Keep points inside the ROI box and outside the ego box."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    keep = ((x >= p.roi_x[0]) & (x <= p.roi_x[1]) & (y >= p.roi_y[0]) & (y <= p.roi_y[1])
            & (z >= p.roi_z[0]) & (z <= p.roi_z[1]))
    keep &= ~((np.abs(x) < p.ego_box[0]) & (np.abs(y) < p.ego_box[1]))
    return xyz[keep]


# ---------------------------------------------------------------- step 3: voxel downsample

def voxel_downsample(xyz, size):
    """One point per occupied voxel, at the voxel's grid center. Independent of input order.

    Returns (centers (M, 3) float64, ijk (M, 3) int64 voxel indices), sorted by packed key.
    """
    if len(xyz) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    ijk = np.floor(xyz / size).astype(np.int64)
    shifted = ijk + _OFFSET
    key = (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]
    uniq = np.unique(key)
    ijk = np.stack([(uniq >> 42) & 0x1FFFFF, (uniq >> 21) & 0x1FFFFF, uniq & 0x1FFFFF], axis=1) - _OFFSET
    return (ijk + 0.5) * size, ijk


# ---------------------------------------------------------------- step 4: ground (seeded RANSAC)

def fit_ground(pts, stamp_ns, p):
    """Return (ground_mask, status). Seeded by [seed, stamp_ns]; all samples are drawn up front."""
    n = len(pts)
    if n < 3:
        return np.zeros(n, dtype=bool), TOO_FEW
    rng = np.random.default_rng([p.seed, int(stamp_ns)])
    idx = rng.integers(0, n, size=(p.ransac_iters, 3))
    a, b, c = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal, axis=1)
    valid = norm > 1e-9
    normal = normal / np.where(valid, norm, 1.0)[:, None]
    if not valid.any():
        return np.zeros(n, dtype=bool), NO_PLANE
    d = -np.einsum("ij,ij->i", normal, a)
    # elementwise (no BLAS matmul): the same bits on every CPU path
    dist = np.abs(pts[None, :, 0] * normal[:, None, 0] + pts[None, :, 1] * normal[:, None, 1]
                  + pts[None, :, 2] * normal[:, None, 2] + d[:, None])
    inliers = dist < p.ransac_dist
    counts = np.where(valid, inliers.sum(axis=1), -1)
    best = int(np.argmax(counts))  # the best plane overall; ties go to the first
    # Accept it only if it is near-horizontal. Filtering candidates by tilt first would let a noisy
    # three-point normal of a steep surface pass (measured: 42-73 % of a 30 degree plane removed).
    if abs(normal[best, 2]) <= p.ransac_min_normal_z:
        return np.zeros(n, dtype=bool), NO_PLANE
    return inliers[best], GROUND_OK


# ---------------------------------------------------------------- step 5: clustering

def cluster_xy(ijk, centers, p):
    """DBSCAN on (x, y). Voxels sharing a column are collapsed into one weighted sample, which gives
    exactly the labels of DBSCAN over every voxel (same x, y) but runs on far fewer samples.
    Returns labels (M,), -1 = noise."""
    m = len(centers)
    if m == 0:
        return np.empty(0, dtype=np.int64)
    key = ((ijk[:, 0] + _OFFSET) << 21) | (ijk[:, 1] + _OFFSET)
    uniq, first, inverse, counts = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    cols = centers[first, :2]
    labels = DBSCAN(eps=p.dbscan_eps, min_samples=p.dbscan_min_samples).fit(
        cols, sample_weight=counts.astype(np.float64)).labels_
    return labels[inverse.reshape(-1)]


# ---------------------------------------------------------------- step 6: oriented extents and the size filter

def oriented_extents(xy, size):
    """(length, width) of the minimum-area rectangle around occupied cells, length >= width.

    xy are voxel-center columns; one voxel size is added to both sides to cover the cell edges.
    Degenerate hulls (fewer than 3 points, collinear) fall back to the extents along the SVD axes.
    """
    xy = np.unique(np.asarray(xy, dtype=np.float64), axis=0)
    ext = None
    if len(xy) >= 3:
        try:
            hull = xy[ConvexHull(xy).vertices]
            edges = np.roll(hull, -1, axis=0) - hull
            u = edges / np.linalg.norm(edges, axis=1)[:, None]
            v = np.stack([-u[:, 1], u[:, 0]], axis=1)
            along = hull @ u.T
            across = hull @ v.T
            e1, e2 = np.ptp(along, axis=0), np.ptp(across, axis=0)
            best = int(np.argmin(e1 * e2))
            ext = (e1[best], e2[best])
        except QhullError:
            ext = None
    if ext is None:
        if len(xy) < 2:
            ext = (0.0, 0.0)
        else:
            centered = xy - xy.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            ext = tuple(np.ptp(centered @ vt.T, axis=0))
    length, width = max(ext) + size, min(ext) + size
    return float(length), float(width)


def passes_size_filter(n, length, width, height, p):
    return (n >= p.cluster_min_points and length <= p.max_length and width <= p.max_width
            and p.min_height <= height <= p.max_height)


# ---------------------------------------------------------------- step 7 and the whole frame

def detect(points, stamp_ns, p):
    """One sweep (N, >=3) -> FrameResult. Never raises on empty or small input."""
    xyz = np.asarray(points, dtype=np.float64)
    xyz = xyz.reshape(-1, xyz.shape[1])[:, :3] if xyz.size else np.empty((0, 3))
    stages = dict.fromkeys(STAGES, 0)
    stages["raw"] = len(xyz)
    empty = np.empty((0, 3), dtype=np.float32)

    roi = crop_roi(xyz, p)
    stages["roi"] = len(roi)
    centers, ijk = voxel_downsample(roi, p.voxel_size)
    stages["voxel"] = len(centers)

    ground, status = fit_ground(centers, stamp_ns, p)
    if status == TOO_FEW:  # nothing to fit: no ground, no detections, empty filtered cloud
        return FrameResult((), stages, 0, 0, status, empty)
    keep = ~ground
    centers, ijk = centers[keep], ijk[keep]
    stages["non_ground"] = len(centers)
    if len(centers) == 0:
        return FrameResult((), stages, 0, 0, status, empty)

    labels = cluster_xy(ijk, centers, p)
    detections, found = [], 0
    for label in range(labels.max() + 1 if len(labels) else 0):
        member = labels == label
        if not member.any():
            continue
        found += 1
        cell = ijk[member]
        lo, hi = cell.min(axis=0), cell.max(axis=0)
        height = (hi[2] - lo[2] + 1) * p.voxel_size
        length, width = oriented_extents(centers[member][:, :2], p.voxel_size)
        if not passes_size_filter(len(cell), length, width, height, p):
            continue
        mid = (lo + hi + 1) * 0.5 * p.voxel_size
        ext = (hi - lo + 1) * p.voxel_size
        detections.append(Detection(float(mid[0]), float(mid[1]), float(mid[2]),
                                    float(ext[0]), float(ext[1]), float(ext[2]), int(len(cell))))
    detections.sort(key=lambda d: (d.x, d.y))
    stages["clustered"] = sum(d.n_points for d in detections)
    return FrameResult(tuple(detections), stages, found, len(detections), status,
                       centers.astype(np.float32))
