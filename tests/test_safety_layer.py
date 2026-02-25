"""Tests for SafetyLayer: violation counting, debounce, and E-STOP."""

from unittest.mock import MagicMock, PropertyMock

from app.core.hardware.safety import SafetyLayer

# ── Basic edge cases ──


def test_check_limits_no_robot(safety_layer):
    """Returns True (safe) when robot is None."""
    assert safety_layer.check_limits(None) is True


def test_check_limits_disconnected(safety_layer):
    """Returns True when robot is not connected."""
    robot = MagicMock()
    robot.is_connected = False
    assert safety_layer.check_limits(robot) is True


def test_check_damiao_limits_no_robot(safety_layer):
    """Returns True when robot is None."""
    assert safety_layer.check_damiao_limits(None) is True


def test_check_damiao_limits_no_torque_api(safety_layer):
    """Returns True when robot doesn't have get_torques method."""
    robot = MagicMock(spec=[])  # No attributes
    robot.is_connected = True
    assert safety_layer.check_damiao_limits(robot) is True


def test_check_all_limits_no_robot(safety_layer):
    """Returns True when robot is None."""
    assert safety_layer.check_all_limits(None) is True


# ── Violation counting (Feetech) ──


def _make_feetech_robot(load_value):
    """Create a mock robot with one Feetech motor bus returning a fixed load."""
    bus = MagicMock()
    bus.motors = {"motor1": MagicMock()}
    bus.read.return_value = load_value

    robot = MagicMock()
    robot.is_connected = True
    # SafetyLayer checks for left_arm, right_arm, bus attributes
    robot.left_arm = MagicMock()
    robot.left_arm.bus = bus
    # Remove right_arm so it doesn't add duplicate bus
    del robot.right_arm
    return robot


def test_feetech_normal_load(safety_layer):
    """Normal load does not trigger violation."""
    robot = _make_feetech_robot(100)  # 100 < threshold 500
    assert safety_layer.check_limits(robot) is True
    assert safety_layer.violation_counts.get("motor1", 0) == 0


def test_feetech_overload_increments_count(safety_layer):
    """Overload increments violation count."""
    robot = _make_feetech_robot(600)  # 600 % 1024 = 600 > 500
    safety_layer.check_limits(robot)  # First call initializes + checks
    assert safety_layer.violation_counts.get("motor1", 0) >= 1


def test_feetech_good_read_resets_count(safety_layer):
    """A good reading after violations resets the counter."""
    robot_overload = _make_feetech_robot(600)
    safety_layer.check_limits(robot_overload)  # violation count = 1

    # Now change to normal load
    robot_overload.left_arm.bus.read.return_value = 100
    safety_layer.check_limits(robot_overload)
    assert safety_layer.violation_counts.get("motor1", 0) == 0


def test_feetech_estop_after_limit(safety_layer):
    """E-STOP triggers after VIOLATION_LIMIT consecutive overloads."""
    robot = _make_feetech_robot(600)
    for _ in range(safety_layer.VIOLATION_LIMIT + 1):
        result = safety_layer.check_limits(robot)

    # After enough violations, check_limits returns False (E-STOP)
    assert result is False
    robot.disconnect.assert_called()


# ── Violation counting (Damiao) ──


def test_damiao_normal_torque(safety_layer):
    """Normal torque does not trigger violation."""
    robot = MagicMock()
    robot.is_connected = True
    robot.get_torques.return_value = {"base": 2.0}
    robot.get_torque_limits.return_value = {"base": 35.0}

    assert safety_layer.check_damiao_limits(robot) is True
    assert safety_layer.violation_counts.get("base", 0) == 0


def test_damiao_overload_increments_count(safety_layer):
    """Torque exceeding limit increments violation count."""
    robot = MagicMock()
    robot.is_connected = True
    robot.get_torques.return_value = {"base": 40.0}
    robot.get_torque_limits.return_value = {"base": 35.0}

    safety_layer.check_damiao_limits(robot)
    assert safety_layer.violation_counts.get("base", 0) == 1


def test_damiao_estop_after_limit(safety_layer):
    """check_damiao_limits returns False after VIOLATION_LIMIT — no disconnect (graceful stop)."""
    robot = MagicMock()
    robot.is_connected = True
    robot.get_torques.return_value = {"link1": 50.0}
    robot.get_torque_limits.return_value = {"link1": 35.0}

    for _ in range(safety_layer.VIOLATION_LIMIT):
        result = safety_layer.check_damiao_limits(robot)

    assert result is False
    robot.disconnect.assert_not_called()  # Graceful: control loop → stop() → homing → disable


def test_damiao_good_read_resets(safety_layer):
    """Good torque reading resets violation count."""
    robot = MagicMock()
    robot.is_connected = True

    # First: overload
    robot.get_torques.return_value = {"base": 50.0}
    robot.get_torque_limits.return_value = {"base": 35.0}
    safety_layer.check_damiao_limits(robot)
    assert safety_layer.violation_counts["base"] == 1

    # Then: normal
    robot.get_torques.return_value = {"base": 10.0}
    safety_layer.check_damiao_limits(robot)
    assert safety_layer.violation_counts["base"] == 0


# ── Emergency stop ──


def test_emergency_stop_disconnects(safety_layer, mock_robot):
    """emergency_stop() calls robot.disconnect()."""
    safety_layer.emergency_stop(mock_robot)
    mock_robot.disconnect.assert_called_once()


def test_emergency_stop_none_robot(safety_layer):
    """emergency_stop(None) doesn't crash."""
    safety_layer.emergency_stop(None)  # Should not raise


# ── Configuration ──


def test_default_thresholds(safety_layer):
    """Verify default safety thresholds."""
    assert safety_layer.LOAD_THRESHOLD == 500
    assert safety_layer.VIOLATION_LIMIT == 3
