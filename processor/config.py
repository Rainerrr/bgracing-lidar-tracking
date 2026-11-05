"""Parameter contract shared by the live node and the offline runner. ROS-free."""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """The parameter set is incomplete or malformed. The message names the key."""


@dataclass(frozen=True)
class Params:
    data_dir: str
    drive: str
    output_dir: str
    report_every_n: int
    seed: int
    roi_x: tuple
    roi_y: tuple
    roi_z: tuple
    ego_box: tuple
    voxel_size: float
    ransac_iters: int
    ransac_dist: float
    ransac_min_normal_z: float
    dbscan_eps: float
    dbscan_min_samples: int
    cluster_min_points: int
    max_length: float
    max_width: float
    min_height: float
    max_height: float


_TYPES = {f.name: f.type for f in fields(Params)}


def _coerce(key, value):
    typ = _TYPES[key]
    if typ is tuple:
        if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            raise ConfigError(f"key '{key}' must be a list of 2 numbers, got {value!r}")
        return tuple(float(v) for v in value)
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"key '{key}' must be a string, got {value!r}")
        return value
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"key '{key}' must be an integer, got {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"key '{key}' must be a number, got {value!r}")
    return float(value)


def params_from_dict(d):
    unknown = sorted(set(d) - set(_TYPES))
    if unknown:
        raise ConfigError(f"unknown key '{unknown[0]}'" + (f" (and {len(unknown) - 1} more)" if len(unknown) > 1 else ""))
    missing = [k for k in _TYPES if k not in d]
    if missing:
        raise ConfigError(f"missing key '{missing[0]}'" + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""))
    return Params(**{k: _coerce(k, d[k]) for k in _TYPES})


def load_params(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text())
        block = doc["obstacle_tracker_node"]["ros__parameters"]
    except (yaml.YAMLError, KeyError, TypeError) as exc:
        raise ConfigError(f"{path}: needs obstacle_tracker_node.ros__parameters ({exc})") from exc
    return params_from_dict(block)
