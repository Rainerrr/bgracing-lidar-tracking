"""Phase 2 parameter contract: one YAML, strict keys."""
import pytest

from processor import config

from conftest import TRACKER_CONFIG


def valid():
    return dict(config.load_params(TRACKER_CONFIG).__dict__)


def test_repo_yaml_loads():
    p = config.load_params(TRACKER_CONFIG)
    assert p.voxel_size == 0.15 and p.dbscan_min_samples == 8 and p.roi_x == (-40.0, 40.0)


def test_missing_key_names_it():
    d = valid()
    del d["seed"]
    with pytest.raises(config.ConfigError, match="seed"):
        config.params_from_dict(d)


def test_unknown_key_names_it():
    d = valid()
    d["voxel"] = 1
    with pytest.raises(config.ConfigError, match="voxel"):
        config.params_from_dict(d)


@pytest.mark.parametrize("key,bad", [("seed", "x"), ("voxel_size", "big"), ("roi_x", [1.0]), ("seed", True)])
def test_wrong_type_names_key(key, bad):
    d = valid()
    d[key] = bad
    with pytest.raises(config.ConfigError, match=key):
        config.params_from_dict(d)


def test_missing_file_and_block(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load_params(tmp_path / "nope.yaml")
    f = tmp_path / "a.yaml"
    f.write_text("other: 1\n")
    with pytest.raises(config.ConfigError, match="ros__parameters"):
        config.load_params(f)


# ---------------------------------------------------------------- Phase 3 keys

def test_tracking_defaults_are_the_approved_ones():
    p = config.load_params(TRACKER_CONFIG)
    assert (p.gate_m, p.confirm_hits, p.max_misses, p.q_accel, p.r_pos, p.v0_sigma, p.moving_speed, p.moving_sigma) == \
        (2.0, 3, 5, 1.0, 0.5, 5.0, 0.7, 2.0)
    assert p.ego_compensation is True and p.association == "hungarian" and p.warmup_frames == 5


@pytest.mark.parametrize("key,bad", [("ego_compensation", 1), ("ego_compensation", "yes"),
                                     ("association", "nearest"), ("association", 3), ("confirm_hits", 3.0)])
def test_tracking_key_types_name_the_key(key, bad):
    d = valid()
    d[key] = bad
    with pytest.raises(config.ConfigError, match=key):
        config.params_from_dict(d)


def test_overrides_parse_and_check():
    p = config.load_params(TRACKER_CONFIG)
    q = config.apply_overrides(p, ["ego_compensation=false", "association=greedy", "gate_m=2.5"])
    assert q.ego_compensation is False and q.association == "greedy" and q.gate_m == 2.5
    assert p.ego_compensation is True  # the original is untouched
    for bad, name in (["nope=1"], "nope"), (["association=nearest"], "association"), (["gate_m"], "key=value"):
        with pytest.raises(config.ConfigError, match=name):
            config.apply_overrides(p, bad)
