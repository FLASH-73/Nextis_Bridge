"""Tests for hardware type definitions: enums, dataclasses, and serialization."""

from app.core.hardware.types import (
    ArmDefinition,
    ArmRole,
    ConnectionStatus,
    MotorType,
    Pairing,
)


# ── Enum value tests ──


def test_motor_type_values():
    assert MotorType.STS3215.value == "sts3215"
    assert MotorType.DAMIAO.value == "damiao"
    assert MotorType.DYNAMIXEL_XL330.value == "dynamixel_xl330"
    assert MotorType.DYNAMIXEL_XL430.value == "dynamixel_xl430"


def test_arm_role_values():
    assert ArmRole.LEADER.value == "leader"
    assert ArmRole.FOLLOWER.value == "follower"


def test_connection_status_values():
    assert ConnectionStatus.DISCONNECTED.value == "disconnected"
    assert ConnectionStatus.CONNECTING.value == "connecting"
    assert ConnectionStatus.CONNECTED.value == "connected"
    assert ConnectionStatus.ERROR.value == "error"


def test_enum_from_string():
    """Enums can be constructed from string values (used in YAML config loading)."""
    assert MotorType("sts3215") == MotorType.STS3215
    assert ArmRole("leader") == ArmRole.LEADER
    assert ConnectionStatus("connected") == ConnectionStatus.CONNECTED


# ── ArmDefinition tests ──


def test_arm_definition_defaults():
    arm = ArmDefinition(
        id="arm1",
        name="Arm 1",
        role=ArmRole.FOLLOWER,
        motor_type=MotorType.STS3215,
        port="/dev/ttyUSB0",
    )
    assert arm.enabled is True
    assert arm.structural_design is None
    assert arm.config == {}
    assert arm.calibrated is False


def test_arm_definition_to_dict():
    arm = ArmDefinition(
        id="test_arm",
        name="Test Arm",
        role=ArmRole.LEADER,
        motor_type=MotorType.DYNAMIXEL_XL330,
        port="/dev/ttyUSB1",
        enabled=False,
        structural_design="umbra_7dof",
        calibrated=True,
    )
    d = arm.to_dict()
    assert d["id"] == "test_arm"
    assert d["name"] == "Test Arm"
    assert d["role"] == "leader"  # Serialized as string value
    assert d["motor_type"] == "dynamixel_xl330"
    assert d["port"] == "/dev/ttyUSB1"
    assert d["enabled"] is False
    assert d["structural_design"] == "umbra_7dof"
    assert d["calibrated"] is True


def test_arm_definition_to_dict_has_all_fields():
    arm = ArmDefinition(
        id="a", name="n", role=ArmRole.FOLLOWER,
        motor_type=MotorType.DAMIAO, port="/dev/can0",
    )
    d = arm.to_dict()
    expected_keys = {"id", "name", "role", "motor_type", "port", "enabled",
                     "structural_design", "config", "calibrated"}
    assert set(d.keys()) == expected_keys


# ── Pairing tests ──


def test_pairing_to_dict():
    p = Pairing(leader_id="l1", follower_id="f1", name="Pair A")
    d = p.to_dict()
    assert d == {"leader_id": "l1", "follower_id": "f1", "name": "Pair A"}


def test_pairing_fields():
    p = Pairing(leader_id="leader1", follower_id="follower1", name="Test")
    assert p.leader_id == "leader1"
    assert p.follower_id == "follower1"
    assert p.name == "Test"
