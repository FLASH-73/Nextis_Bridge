"""Tests for PairingContext, build_pairing_context, and joint mapping."""

from unittest.mock import MagicMock

from app.core.hardware.types import ArmDefinition, ArmRole, MotorType
from app.core.teleop.pairing import (
    DYNAMIXEL_TO_DAMIAO_JOINT_MAP,
    PairingContext,
    build_pairing_context,
    get_arm_prefix,
)


# ── DYNAMIXEL_TO_DAMIAO_JOINT_MAP ──


def test_joint_map_completeness():
    """Map covers all 7 joints (6 DOF + gripper)."""
    assert len(DYNAMIXEL_TO_DAMIAO_JOINT_MAP) == 7


def test_joint_map_entries():
    """Verify specific joint mappings."""
    assert DYNAMIXEL_TO_DAMIAO_JOINT_MAP["joint_1"] == "base"
    assert DYNAMIXEL_TO_DAMIAO_JOINT_MAP["joint_2"] == "link1"
    assert DYNAMIXEL_TO_DAMIAO_JOINT_MAP["joint_3"] == "link2"
    assert DYNAMIXEL_TO_DAMIAO_JOINT_MAP["joint_4"] == "link3"
    assert DYNAMIXEL_TO_DAMIAO_JOINT_MAP["joint_5"] == "link4"
    assert DYNAMIXEL_TO_DAMIAO_JOINT_MAP["joint_6"] == "link5"
    assert DYNAMIXEL_TO_DAMIAO_JOINT_MAP["gripper"] == "gripper"


# ── PairingContext dataclass ──


def test_pairing_context_defaults():
    """PairingContext mutable fields default correctly."""
    ctx = PairingContext(
        pairing_id="test→test",
        active_leader=None,
        active_robot=None,
        joint_mapping={"a": "b"},
        follower_value_mode="float",
        has_damiao_follower=True,
        leader_cal_ranges={},
    )
    assert ctx.follower_start_pos == {}
    assert ctx.leader_start_rad == {}
    assert ctx.rad_to_percent_scale == {}
    assert ctx.blend_start_time is None
    assert ctx.filtered_gripper_torque == 0.0


# ── get_arm_prefix ──


def test_arm_prefix_left():
    svc = MagicMock()
    assert get_arm_prefix(svc, "left_follower") == "left_"


def test_arm_prefix_right():
    svc = MagicMock()
    assert get_arm_prefix(svc, "right_leader") == "right_"


def test_arm_prefix_damiao():
    svc = MagicMock()
    assert get_arm_prefix(svc, "damiao_follower") == ""


def test_arm_prefix_custom_arm():
    """Custom arm IDs use registry lookup."""
    svc = MagicMock()
    svc.arm_registry.get_arm.return_value = {"id": "custom_arm"}
    assert get_arm_prefix(svc, "custom_arm") == "custom_arm_"


# ── build_pairing_context ──


def _make_svc(leader_type, follower_type):
    """Create a mock TeleoperationService with arm registry."""
    svc = MagicMock()
    svc.joint_names_template = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]

    leader = ArmDefinition(
        id="leader1", name="L", role=ArmRole.LEADER,
        motor_type=leader_type, port="/dev/ttyUSB0",
    )
    follower = ArmDefinition(
        id="follower1", name="F", role=ArmRole.FOLLOWER,
        motor_type=follower_type, port="/dev/ttyUSB1",
    )
    svc.arm_registry.arms = {"leader1": leader, "follower1": follower}
    return svc


def test_dynamixel_to_damiao_context():
    """Dynamixel→Damiao produces float mode with correct mapping."""
    svc = _make_svc(MotorType.DYNAMIXEL_XL330, MotorType.DAMIAO)
    pairing = {"leader_id": "leader1", "follower_id": "follower1"}

    ctx = build_pairing_context(svc, pairing, MagicMock(), MagicMock())

    assert ctx.follower_value_mode == "float"
    assert ctx.has_damiao_follower is True
    assert len(ctx.joint_mapping) == 7
    assert ctx.joint_mapping["joint_1.pos"] == "base.pos"
    assert ctx.joint_mapping["gripper.pos"] == "gripper.pos"


def test_dynamixel_to_feetech_context():
    """Dynamixel→Feetech produces rad_to_percent mode."""
    svc = _make_svc(MotorType.DYNAMIXEL_XL330, MotorType.STS3215)
    pairing = {"leader_id": "leader1", "follower_id": "follower1"}

    # Leader without calibration
    leader_inst = MagicMock()
    leader_inst.calibration = None

    ctx = build_pairing_context(svc, pairing, leader_inst, MagicMock())

    assert ctx.follower_value_mode == "rad_to_percent"
    assert ctx.has_damiao_follower is False
    assert len(ctx.joint_mapping) == 7


def test_dynamixel_to_feetech_with_calibration():
    """Dynamixel→Feetech populates leader_cal_ranges from calibration."""
    svc = _make_svc(MotorType.DYNAMIXEL_XL330, MotorType.STS3215)
    pairing = {"leader_id": "leader1", "follower_id": "follower1"}

    # Leader with calibration for all joints
    leader_inst = MagicMock()
    cal = {}
    for joint in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
        cal[joint] = MagicMock()
        cal[joint].range_min = -3.14
        cal[joint].range_max = 3.14
    leader_inst.calibration = cal

    ctx = build_pairing_context(svc, pairing, leader_inst, MagicMock())

    assert ctx.follower_value_mode == "rad_to_percent"
    # 6 joints (gripper excluded from cal ranges)
    assert len(ctx.leader_cal_ranges) == 6


def test_same_type_legacy_context():
    """Same-type arms use legacy prefix mapping."""
    svc = _make_svc(MotorType.STS3215, MotorType.STS3215)
    pairing = {"leader_id": "leader1", "follower_id": "follower1"}

    # For same-type, get_arm_prefix is called. Mock it to return custom prefix.
    # The leader1/follower1 IDs don't start with left_/right_ so they'll use registry lookup.
    svc.arm_registry.get_arm.return_value = {"id": "leader1"}

    ctx = build_pairing_context(svc, pairing, MagicMock(), MagicMock())

    assert ctx.follower_value_mode == "int"
    assert ctx.has_damiao_follower is False
    assert len(ctx.joint_mapping) == 7


def test_context_isolation():
    """Two contexts from different pairings have independent state."""
    svc = _make_svc(MotorType.DYNAMIXEL_XL330, MotorType.DAMIAO)

    # Add another follower (Feetech)
    svc.arm_registry.arms["follower2"] = ArmDefinition(
        id="follower2", name="F2", role=ArmRole.FOLLOWER,
        motor_type=MotorType.STS3215, port="/dev/ttyUSB2",
    )

    p1 = {"leader_id": "leader1", "follower_id": "follower1"}
    p2 = {"leader_id": "leader1", "follower_id": "follower2"}

    ctx1 = build_pairing_context(svc, p1, MagicMock(), MagicMock())
    leader_inst = MagicMock()
    leader_inst.calibration = None
    ctx2 = build_pairing_context(svc, p2, leader_inst, MagicMock())

    # Critical: these must be independent (the Feb 2026 multi-pair bug)
    assert ctx1.follower_value_mode == "float"
    assert ctx2.follower_value_mode == "rad_to_percent"
    assert ctx1.joint_mapping is not ctx2.joint_mapping
