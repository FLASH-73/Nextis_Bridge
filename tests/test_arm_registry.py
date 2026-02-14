"""Tests for ArmRegistryService: CRUD, config loading, pairings."""

import yaml
import pytest
from unittest.mock import MagicMock, patch

from app.core.hardware.arm_registry import ArmRegistryService
from app.core.hardware.types import ArmRole, MotorType, ConnectionStatus


# ── Fixtures ──


@pytest.fixture
def registry_config(tmp_path):
    """Create a temp YAML config and return its path."""
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.dump({
        "arms": {
            "leader1": {
                "name": "Leader 1",
                "role": "leader",
                "motor_type": "dynamixel_xl330",
                "port": "/dev/ttyUSB0",
                "enabled": True,
            },
            "follower1": {
                "name": "Follower 1",
                "role": "follower",
                "motor_type": "sts3215",
                "port": "/dev/ttyUSB1",
                "enabled": True,
            },
        },
        "pairings": [
            {"leader": "leader1", "follower": "follower1", "name": "Pair A"},
        ],
    }))
    return str(config)


@pytest.fixture
def registry(registry_config):
    """An ArmRegistryService loaded from test config."""
    return ArmRegistryService(config_path=registry_config)


@pytest.fixture
def empty_registry(tmp_path):
    """An ArmRegistryService with no config file."""
    config = tmp_path / "nonexistent.yaml"
    return ArmRegistryService(config_path=str(config))


# ── Config loading ──


def test_loads_arms_from_yaml(registry):
    """Arms are loaded from config file."""
    assert "leader1" in registry.arms
    assert "follower1" in registry.arms
    assert registry.arms["leader1"].name == "Leader 1"
    assert registry.arms["leader1"].role == ArmRole.LEADER
    assert registry.arms["follower1"].motor_type == MotorType.STS3215


def test_loads_pairings_from_yaml(registry):
    """Pairings are loaded from config file."""
    assert len(registry.pairings) == 1
    assert registry.pairings[0].leader_id == "leader1"
    assert registry.pairings[0].follower_id == "follower1"


def test_empty_config_no_crash(empty_registry):
    """Missing config file doesn't crash, just produces empty registry."""
    assert len(empty_registry.arms) == 0
    assert len(empty_registry.pairings) == 0


def test_arm_status_initialized(registry):
    """All arms start as DISCONNECTED."""
    for arm_id in registry.arms:
        assert registry.arm_status[arm_id] == ConnectionStatus.DISCONNECTED


# ── CRUD ──


def test_get_all_arms(registry):
    """get_all_arms() returns all arms with status."""
    arms = registry.get_all_arms()
    assert len(arms) == 2
    ids = {a["id"] for a in arms}
    assert ids == {"leader1", "follower1"}


def test_get_arm_found(registry):
    """get_arm() returns arm data when found."""
    arm = registry.get_arm("leader1")
    assert arm is not None
    assert arm["name"] == "Leader 1"


def test_get_arm_not_found(registry):
    """get_arm() returns None for unknown ID."""
    assert registry.get_arm("nonexistent") is None


def test_add_arm(registry):
    """add_arm() adds a new arm to the registry."""
    result = registry.add_arm({
        "id": "follower2",
        "name": "Follower 2",
        "role": "follower",
        "motor_type": "damiao",
        "port": "/dev/can0",
    })
    assert result["success"] is True
    assert "follower2" in registry.arms
    assert registry.arms["follower2"].motor_type == MotorType.DAMIAO


def test_add_arm_duplicate_id(registry):
    """add_arm() rejects duplicate IDs."""
    result = registry.add_arm({
        "id": "leader1",
        "name": "Duplicate",
        "role": "leader",
        "motor_type": "dynamixel_xl330",
        "port": "/dev/ttyUSB2",
    })
    assert result["success"] is False


def test_update_arm(registry):
    """update_arm() modifies arm fields."""
    result = registry.update_arm("follower1", name="Updated Follower")
    assert result["success"] is True
    assert registry.arms["follower1"].name == "Updated Follower"


def test_update_arm_not_found(registry):
    """update_arm() returns error for unknown ID."""
    result = registry.update_arm("nonexistent", name="X")
    assert result["success"] is False


def test_remove_arm(registry):
    """remove_arm() removes arm and cleans up pairings."""
    result = registry.remove_arm("follower1")
    assert result["success"] is True
    assert "follower1" not in registry.arms
    # Pairing should be cleaned up too
    for p in registry.pairings:
        assert p.follower_id != "follower1"


def test_remove_arm_not_found(registry):
    """remove_arm() returns error for unknown ID."""
    result = registry.remove_arm("nonexistent")
    assert result["success"] is False


# ── Pairings ──


def test_get_pairings(registry):
    """get_pairings() returns all pairing dicts."""
    pairings = registry.get_pairings()
    assert len(pairings) == 1
    assert pairings[0]["leader_id"] == "leader1"


def test_create_pairing(registry):
    """create_pairing() adds a new leader-follower pair."""
    # Add another follower first
    registry.add_arm({
        "id": "follower2",
        "name": "F2",
        "role": "follower",
        "motor_type": "sts3215",
        "port": "/dev/ttyUSB2",
    })
    result = registry.create_pairing("leader1", "follower2", "Pair B")
    assert result["success"] is True
    assert len(registry.pairings) == 2


def test_create_pairing_duplicate(registry):
    """create_pairing() rejects duplicate pairs."""
    result = registry.create_pairing("leader1", "follower1", "Duplicate")
    assert result["success"] is False


def test_remove_pairing(registry):
    """remove_pairing() removes a pair."""
    result = registry.remove_pairing("leader1", "follower1")
    assert result["success"] is True
    assert len(registry.pairings) == 0


def test_get_active_pairings_all(registry):
    """get_active_pairings(None) returns all pairings."""
    active = registry.get_active_pairings(None)
    assert len(active) == 1


def test_get_active_pairings_filtered(registry):
    """get_active_pairings() filters by active arm IDs."""
    active = registry.get_active_pairings(["leader1", "follower1"])
    assert len(active) == 1

    active = registry.get_active_pairings(["leader1"])  # follower1 missing
    assert len(active) == 0


# ── Status ──


def test_get_status_summary(registry):
    """get_status_summary() returns counts."""
    summary = registry.get_status_summary()
    assert summary["total_arms"] == 2
    assert summary["leaders"] >= 1
    assert summary["followers"] >= 1


# ── Leaders / Followers ──


def test_get_leaders(registry):
    """get_leaders() returns only leader arms."""
    leaders = registry.get_leaders()
    assert len(leaders) == 1
    assert leaders[0]["role"] == "leader"


def test_get_followers(registry):
    """get_followers() returns only follower arms."""
    followers = registry.get_followers()
    assert len(followers) == 1
    assert followers[0]["role"] == "follower"


# ── Config persistence ──


def test_save_config_roundtrip(registry, registry_config):
    """Saving and reloading config preserves arms and pairings."""
    registry.add_arm({
        "id": "extra",
        "name": "Extra",
        "role": "follower",
        "motor_type": "damiao",
        "port": "/dev/can0",
    })
    registry._save_config()

    # Reload
    new_reg = ArmRegistryService(config_path=registry_config)
    assert "extra" in new_reg.arms
    assert new_reg.arms["extra"].motor_type == MotorType.DAMIAO
