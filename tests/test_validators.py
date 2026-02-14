"""Tests for ValidatorMixin: dataset validation per policy type."""

import json
import pytest
from pathlib import Path

from app.core.training.service import TrainingService


# ── Helpers ──


def _make_dataset(base_path, repo_id, info):
    """Create a fake dataset directory with meta/info.json."""
    ds_path = base_path / repo_id
    meta = ds_path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info))
    return ds_path


def _make_stats(base_path, repo_id, stats):
    """Create a meta/stats.json for a dataset."""
    stats_path = base_path / repo_id / "meta" / "stats.json"
    stats_path.write_text(json.dumps(stats))


def _full_features(**overrides):
    """Return a features dict with images, state, and action."""
    features = {
        "observation.images.camera1": {"shape": [3, 480, 640], "dtype": "image"},
        "observation.state": {"shape": [7]},
        "action": {"shape": [7]},
    }
    features.update(overrides)
    return features


# ── Basic validation ──


def test_path_traversal_rejected(training_service):
    """Repo IDs with .. are rejected."""
    result = training_service.validate_dataset("../etc/passwd", "smolvla")
    assert result.valid is False
    assert any("path traversal" in e.lower() for e in result.errors)


def test_missing_dataset(training_service):
    """Non-existent dataset returns error."""
    result = training_service.validate_dataset("nonexistent/dataset", "smolvla")
    assert result.valid is False
    assert any("not found" in e.lower() for e in result.errors)


def test_missing_info_json(training_service, tmp_path):
    """Dataset without info.json returns error."""
    ds = training_service.datasets_path / "test_repo"
    ds.mkdir(parents=True)
    result = training_service.validate_dataset("test_repo", "smolvla")
    assert result.valid is False
    assert any("info.json" in e for e in result.errors)


def test_zero_episodes(training_service):
    """Dataset with 0 episodes returns error."""
    _make_dataset(training_service.datasets_path, "empty_ds", {
        "total_episodes": 0,
        "total_frames": 100,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("empty_ds", "smolvla")
    assert result.valid is False
    assert any("no episodes" in e.lower() for e in result.errors)


def test_zero_frames(training_service):
    """Dataset with 0 frames returns error."""
    _make_dataset(training_service.datasets_path, "no_frames", {
        "total_episodes": 5,
        "total_frames": 0,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("no_frames", "smolvla")
    assert result.valid is False
    assert any("no frames" in e.lower() for e in result.errors)


def test_unknown_policy_type(training_service):
    """Unknown policy type produces warning, not error."""
    _make_dataset(training_service.datasets_path, "ds1", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("ds1", "unknown_policy")
    assert result.valid is True
    assert any("unknown" in w.lower() for w in result.warnings)


# ── SmolVLA validation ──


def test_smolvla_valid(training_service):
    """SmolVLA passes with images, state, and action."""
    _make_dataset(training_service.datasets_path, "smolvla_ok", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("smolvla_ok", "smolvla")
    assert result.valid is True


def test_smolvla_missing_images(training_service):
    """SmolVLA requires at least one camera."""
    _make_dataset(training_service.datasets_path, "no_img", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": {
            "observation.state": {"shape": [7]},
            "action": {"shape": [7]},
        },
    })
    result = training_service.validate_dataset("no_img", "smolvla")
    assert result.valid is False
    assert any("camera" in e.lower() or "images" in e.lower() for e in result.errors)


def test_smolvla_missing_state(training_service):
    """SmolVLA requires observation.state."""
    features = _full_features()
    del features["observation.state"]
    _make_dataset(training_service.datasets_path, "no_state", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": features,
    })
    result = training_service.validate_dataset("no_state", "smolvla")
    assert result.valid is False
    assert any("state" in e.lower() for e in result.errors)


def test_smolvla_missing_action(training_service):
    """SmolVLA requires action feature."""
    features = _full_features()
    del features["action"]
    _make_dataset(training_service.datasets_path, "no_action", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": features,
    })
    result = training_service.validate_dataset("no_action", "smolvla")
    assert result.valid is False
    assert any("action" in e.lower() for e in result.errors)


def test_smolvla_high_dim_warning(training_service):
    """SmolVLA warns on state dimension > 32."""
    features = _full_features()
    features["observation.state"] = {"shape": [64]}
    _make_dataset(training_service.datasets_path, "high_dim", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": features,
    })
    result = training_service.validate_dataset("high_dim", "smolvla")
    assert result.valid is True
    assert any("32" in w for w in result.warnings)


# ── Diffusion validation ──


def test_diffusion_valid(training_service):
    """Diffusion passes with images and action."""
    _make_dataset(training_service.datasets_path, "diff_ok", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("diff_ok", "diffusion")
    assert result.valid is True


def test_diffusion_missing_action(training_service):
    """Diffusion requires action."""
    features = _full_features()
    del features["action"]
    _make_dataset(training_service.datasets_path, "diff_no_act", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": features,
    })
    result = training_service.validate_dataset("diff_no_act", "diffusion")
    assert result.valid is False


def test_diffusion_warns_missing_state(training_service):
    """Diffusion warns (not errors) on missing state."""
    features = _full_features()
    del features["observation.state"]
    _make_dataset(training_service.datasets_path, "diff_no_state", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": features,
    })
    result = training_service.validate_dataset("diff_no_state", "diffusion")
    assert result.valid is True
    assert any("state" in w.lower() for w in result.warnings)


def test_diffusion_multi_shape_warning(training_service):
    """Diffusion warns when cameras have different shapes."""
    features = _full_features()
    features["observation.images.camera2"] = {"shape": [3, 720, 1280]}
    _make_dataset(training_service.datasets_path, "diff_multi", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": features,
    })
    result = training_service.validate_dataset("diff_multi", "diffusion")
    assert result.valid is True
    assert any("shape" in w.lower() or "resize" in w.lower() for w in result.warnings)


# ── ACT validation ──


def test_act_valid(training_service):
    """ACT passes with images, action, and state."""
    _make_dataset(training_service.datasets_path, "act_ok", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("act_ok", "act")
    assert result.valid is True


def test_act_requires_state(training_service):
    """ACT requires observation.state (not just a warning)."""
    features = _full_features()
    del features["observation.state"]
    _make_dataset(training_service.datasets_path, "act_no_state", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": features,
    })
    result = training_service.validate_dataset("act_no_state", "act")
    assert result.valid is False


# ── Pi0.5 validation ──


def test_pi05_valid(training_service):
    """Pi0.5 passes with images, action, state."""
    _make_dataset(training_service.datasets_path, "pi05_ok", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("pi05_ok", "pi05")
    assert result.valid is True


def test_pi05_warns_missing_quantiles(training_service):
    """Pi0.5 warns when quantile stats are missing."""
    _make_dataset(training_service.datasets_path, "pi05_noq", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": _full_features(),
    })
    result = training_service.validate_dataset("pi05_noq", "pi05")
    assert result.valid is True
    assert any("quantile" in w.lower() for w in result.warnings)


def test_pi05_no_warning_with_quantiles(training_service):
    """Pi0.5 doesn't warn when quantile stats are present."""
    _make_dataset(training_service.datasets_path, "pi05_q", {
        "total_episodes": 5,
        "total_frames": 100,
        "features": _full_features(),
    })
    _make_stats(training_service.datasets_path, "pi05_q", {
        "action": {"q01": [0.0], "q99": [1.0], "mean": [0.5], "std": [0.1]},
        "observation.state": {"q01": [0.0], "q99": [1.0], "mean": [0.5], "std": [0.1]},
    })
    result = training_service.validate_dataset("pi05_q", "pi05")
    assert result.valid is True
    assert not any("quantile" in w.lower() for w in result.warnings)


# ── has_quantile_stats ──


def test_has_quantile_stats_true(training_service):
    """has_quantile_stats returns True when q01/q99 present."""
    _make_dataset(training_service.datasets_path, "qs_ok", {
        "total_episodes": 5, "total_frames": 100, "features": {},
    })
    _make_stats(training_service.datasets_path, "qs_ok", {
        "action": {"q01": [0], "q99": [1]},
        "observation.state": {"q01": [0], "q99": [1]},
    })
    result = training_service.has_quantile_stats("qs_ok")
    assert result["has_quantiles"] is True


def test_has_quantile_stats_missing(training_service):
    """has_quantile_stats returns False when stats.json missing."""
    _make_dataset(training_service.datasets_path, "qs_no", {
        "total_episodes": 5, "total_frames": 100, "features": {},
    })
    result = training_service.has_quantile_stats("qs_no")
    assert result["has_quantiles"] is False


# ── Rename map ──


def test_smolvla_rename_map(training_service):
    """Rename map normalizes camera names to camera1, camera2."""
    features = {
        "observation.images.wrist": {"shape": [3, 480, 640]},
        "observation.images.head": {"shape": [3, 480, 640]},
        "observation.state": {"shape": [7]},
        "action": {"shape": [7]},
    }
    rename = training_service._build_smolvla_rename_map(features)
    # Should map the two cameras to camera1 and camera2
    assert len(rename) == 2
    target_values = set(rename.values())
    assert "observation.images.camera1" in target_values
    assert "observation.images.camera2" in target_values
