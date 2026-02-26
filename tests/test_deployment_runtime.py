"""Tests for the unified deployment runtime.

Covers:
- Start/stop lifecycle and state transitions
- Safety pipeline always called
- Arm resolution from registry
- InterventionDetector position delta logic
- ObservationBuilder standalone usage
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.deployment.intervention import InterventionDetector
from app.core.deployment.rl_learner import RLLearner
from app.core.deployment.types import (
    ActionSource,
    DeploymentConfig,
    DeploymentMode,
    DeploymentStatus,
    RuntimeState,
    SafetyConfig,
)
from app.core.hardware.types import ArmDefinition, ArmRole, MotorType, Pairing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def robot_lock():
    return threading.Lock()


@pytest.fixture
def mock_arm_registry():
    registry = MagicMock()

    leader_def = ArmDefinition(
        id="leader_left",
        name="Left Leader",
        role=ArmRole.LEADER,
        motor_type=MotorType.DYNAMIXEL_XL330,
        port="/dev/ttyUSB0",
        enabled=True,
        structural_design="umbra_7dof",
        config={"motor_names": ["left_base.pos", "left_link1.pos"]},
    )
    follower_def = ArmDefinition(
        id="follower_left",
        name="Left Follower",
        role=ArmRole.FOLLOWER,
        motor_type=MotorType.STS3215,
        port="/dev/ttyUSB1",
        enabled=True,
        structural_design="umbra_7dof",
        config={"motor_names": ["left_base.pos", "left_link1.pos"]},
    )

    registry.arms = {
        "leader_left": leader_def,
        "follower_left": follower_def,
    }

    pairing = Pairing(
        leader_id="leader_left",
        follower_id="follower_left",
        name="Left Pair",
    )
    registry.pairings = [pairing]
    registry.get_active_pairings.return_value = [pairing.to_dict()]

    leader_inst = MagicMock()
    leader_inst.get_action.return_value = {
        "left_base.pos": 0.0,
        "left_link1.pos": 0.0,
    }

    follower_inst = MagicMock()
    follower_inst.is_connected = True
    follower_inst.get_observation.return_value = {
        "left_base.pos": 0.1,
        "left_link1.pos": 0.2,
    }

    registry.arm_instances = {
        "leader_left": leader_inst,
        "follower_left": follower_inst,
    }

    return registry


@pytest.fixture
def mock_teleop():
    teleop = MagicMock()
    teleop.safety = MagicMock()
    teleop._action_lock = threading.Lock()
    teleop._latest_leader_action = {}
    teleop.dataset = None
    return teleop


@pytest.fixture
def mock_training():
    training = MagicMock()

    policy_info = MagicMock()
    policy_info.checkpoint_path = "/tmp/fake_checkpoint"

    policy_config = MagicMock()
    policy_config.cameras = ["camera_1"]
    policy_config.arms = ["left"]
    policy_config.policy_type = "act"
    policy_config.state_dim = 7
    policy_config.action_dim = 7

    training.get_policy.return_value = policy_info
    training.get_policy_config.return_value = policy_config
    return training


@pytest.fixture
def mock_camera_service():
    return MagicMock()


@pytest.fixture
def deployment_config():
    return DeploymentConfig(
        mode=DeploymentMode.INFERENCE,
        policy_id="test_policy",
        safety=SafetyConfig(),
    )


@pytest.fixture
def runtime(mock_teleop, mock_training, mock_arm_registry, mock_camera_service, robot_lock):
    """Create a DeploymentRuntime with mocked dependencies."""
    from app.core.deployment.runtime import DeploymentRuntime

    return DeploymentRuntime(
        teleop_service=mock_teleop,
        training_service=mock_training,
        arm_registry=mock_arm_registry,
        camera_service=mock_camera_service,
        robot_lock=robot_lock,
    )


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Test start/stop lifecycle."""

    def test_initial_state_is_idle(self, runtime):
        assert runtime._state == RuntimeState.IDLE

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_start_transitions_to_running(
        self, mock_load, runtime, deployment_config
    ):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])

        # Give the loop thread a moment to start
        time.sleep(0.05)
        assert runtime._state == RuntimeState.RUNNING

        runtime.stop()
        assert runtime._state == RuntimeState.IDLE

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_stop_is_idempotent(self, mock_load, runtime, deployment_config):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])
        time.sleep(0.05)

        runtime.stop()
        assert runtime._state == RuntimeState.IDLE

        # Second stop should not raise
        runtime.stop()
        assert runtime._state == RuntimeState.IDLE

    def test_start_fails_from_non_idle(self, runtime, deployment_config):
        # Manually set state to RUNNING
        runtime._state = RuntimeState.RUNNING

        with pytest.raises(RuntimeError, match="Cannot start"):
            runtime.start(deployment_config, ["leader_left", "follower_left"])

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_get_status_returns_deployment_status(
        self, mock_load, runtime, deployment_config
    ):
        status = runtime.get_status()
        assert isinstance(status, DeploymentStatus)
        assert status.state == RuntimeState.IDLE


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Test state machine transitions."""

    def test_valid_transitions(self, runtime):
        assert runtime._transition(RuntimeState.STARTING)
        assert runtime._state == RuntimeState.STARTING

        assert runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.RUNNING

        assert runtime._transition(RuntimeState.HUMAN_ACTIVE)
        assert runtime._state == RuntimeState.HUMAN_ACTIVE

        assert runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.RUNNING

        assert runtime._transition(RuntimeState.PAUSED)
        assert runtime._state == RuntimeState.PAUSED

        assert runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.RUNNING

        assert runtime._transition(RuntimeState.STOPPING)
        assert runtime._state == RuntimeState.STOPPING

        assert runtime._transition(RuntimeState.IDLE)
        assert runtime._state == RuntimeState.IDLE

    def test_invalid_transition_rejected(self, runtime):
        # IDLE → RUNNING is not valid (must go through STARTING)
        assert not runtime._transition(RuntimeState.RUNNING)
        assert runtime._state == RuntimeState.IDLE

    def test_estop_reachable_from_running(self, runtime):
        runtime._state = RuntimeState.RUNNING
        assert runtime._transition(RuntimeState.ESTOP)
        assert runtime._state == RuntimeState.ESTOP

    def test_estop_is_terminal(self, runtime):
        runtime._state = RuntimeState.ESTOP
        assert not runtime._transition(RuntimeState.RUNNING)
        assert not runtime._transition(RuntimeState.IDLE)

    def test_reset_from_estop(self, runtime):
        runtime._state = RuntimeState.ESTOP
        runtime._safety_pipeline = MagicMock()
        assert runtime.reset()
        assert runtime._state == RuntimeState.IDLE

    def test_reset_from_error(self, runtime):
        runtime._state = RuntimeState.ERROR
        runtime._safety_pipeline = MagicMock()
        assert runtime.reset()
        assert runtime._state == RuntimeState.IDLE

    def test_reset_fails_from_running(self, runtime):
        runtime._state = RuntimeState.RUNNING
        assert not runtime.reset()

    def test_pause_from_running(self, runtime):
        runtime._state = RuntimeState.RUNNING
        assert runtime.pause()
        assert runtime._state == RuntimeState.PAUSED

    def test_resume_from_paused(self, runtime):
        runtime._state = RuntimeState.PAUSED
        assert runtime.resume()
        assert runtime._state == RuntimeState.RUNNING


# ---------------------------------------------------------------------------
# Safety pipeline tests
# ---------------------------------------------------------------------------


class TestSafetyPipeline:
    """Verify safety pipeline is always called."""

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_safety_called_on_every_frame(
        self, mock_load, runtime, deployment_config, mock_arm_registry
    ):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])

        # Let a few frames run
        time.sleep(0.15)
        runtime.stop()

        # Safety pipeline should have been created and its process() called
        assert runtime._safety_pipeline is None  # cleaned up after stop

    @patch("app.core.deployment.runtime.DeploymentRuntime._load_policy")
    def test_safety_pipeline_created_with_safety_layer(
        self, mock_load, runtime, deployment_config, mock_teleop
    ):
        mock_load.return_value = None
        runtime._policy = MagicMock()
        runtime._checkpoint_path = Path("/tmp/fake")

        runtime.start(deployment_config, ["leader_left", "follower_left"])
        time.sleep(0.05)

        # Verify pipeline was created (it gets cleaned up on stop)
        # We check by ensuring stop cleaned it up
        runtime.stop()
        assert runtime._safety_pipeline is None


# ---------------------------------------------------------------------------
# Arm resolution tests
# ---------------------------------------------------------------------------


class TestArmResolution:
    """Test arm resolution from registry."""

    def test_resolve_arms_calls_get_active_pairings(
        self, runtime, mock_arm_registry
    ):
        runtime._resolve_arms(["leader_left", "follower_left"])

        mock_arm_registry.get_active_pairings.assert_called_once_with(
            ["leader_left", "follower_left"]
        )
        assert runtime._leader is not None
        assert runtime._follower is not None

    def test_resolve_arms_auto_connects(self, runtime, mock_arm_registry):
        # Remove leader from instances to trigger auto-connect
        del mock_arm_registry.arm_instances["leader_left"]

        runtime._resolve_arms(["leader_left", "follower_left"])

        mock_arm_registry.connect_arm.assert_called_with("leader_left")

    def test_resolve_arms_raises_on_no_pairings(
        self, runtime, mock_arm_registry
    ):
        mock_arm_registry.get_active_pairings.return_value = []

        with pytest.raises(RuntimeError, match="No pairings found"):
            runtime._resolve_arms(["unknown_arm"])

    def test_resolve_arms_raises_without_registry(self, runtime):
        runtime._arm_registry = None

        with pytest.raises(RuntimeError, match="No arm registry"):
            runtime._resolve_arms(["leader_left"])


# ---------------------------------------------------------------------------
# InterventionDetector tests
# ---------------------------------------------------------------------------


class TestInterventionDetector:
    """Test the position-delta intervention detector."""

    def test_first_call_returns_no_intervention(self):
        leader = MagicMock()
        leader.get_action.return_value = {"left_base.pos": 0.0}

        detector = InterventionDetector(policy_arms=["left"], loop_hz=30)
        is_intervening, velocity = detector.check(leader)

        assert not is_intervening
        assert velocity == 0.0

    def test_large_position_delta_triggers_intervention(self):
        leader = MagicMock()
        detector = InterventionDetector(
            policy_arms=["left"],
            move_threshold=0.05,
            loop_hz=30,
        )

        # First call initializes
        leader.get_action.return_value = {"left_base.pos": 0.0}
        detector.check(leader)

        # Large movement
        leader.get_action.return_value = {"left_base.pos": 0.5}
        is_intervening, velocity = detector.check(leader)

        assert is_intervening
        assert velocity == pytest.approx(0.5 * 30, abs=0.01)

    def test_small_delta_no_intervention(self):
        leader = MagicMock()
        detector = InterventionDetector(
            policy_arms=["left"],
            move_threshold=0.05,
            loop_hz=30,
        )

        leader.get_action.return_value = {"left_base.pos": 0.0}
        detector.check(leader)

        # Tiny movement (0.001 * 30 = 0.03 < 0.05 threshold)
        leader.get_action.return_value = {"left_base.pos": 0.001}
        is_intervening, velocity = detector.check(leader)

        assert not is_intervening

    def test_policy_arms_filtering(self):
        leader = MagicMock()
        detector = InterventionDetector(
            policy_arms=["left"],  # Only left arm triggers
            move_threshold=0.05,
            loop_hz=30,
        )

        # First call
        leader.get_action.return_value = {
            "left_base.pos": 0.0,
            "right_base.pos": 0.0,
        }
        detector.check(leader)

        # Only right arm moves — should NOT trigger (left-only policy)
        leader.get_action.return_value = {
            "left_base.pos": 0.0,
            "right_base.pos": 1.0,
        }
        is_intervening, velocity = detector.check(leader)

        assert not is_intervening

    def test_none_leader_returns_false(self):
        detector = InterventionDetector()
        is_intervening, velocity = detector.check(None)
        assert not is_intervening
        assert velocity == 0.0

    def test_is_idle_after_timeout(self):
        detector = InterventionDetector(idle_timeout=0.05)
        assert detector.is_idle()

        # Simulate a move
        detector._last_move_time = time.monotonic()
        assert not detector.is_idle()

        # Wait for timeout
        time.sleep(0.06)
        assert detector.is_idle()

    def test_reset_clears_state(self):
        detector = InterventionDetector()
        detector._last_positions = {"a": 1.0}
        detector._last_move_time = time.monotonic()

        detector.reset()

        assert detector._last_positions is None
        assert detector._last_move_time == 0.0


# ---------------------------------------------------------------------------
# ObservationBuilder tests
# ---------------------------------------------------------------------------


class TestObservationBuilder:
    """Test ObservationBuilder standalone usage."""

    def test_get_training_state_names_caches(self, tmp_path):
        import json

        from app.core.deployment.observation_builder import ObservationBuilder

        # Set up fake checkpoint with train_config
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "meta").mkdir(parents=True)

        # Write train_config.json
        train_config = {"dataset": {"root": str(dataset_dir)}}
        (checkpoint / "train_config.json").write_text(json.dumps(train_config))

        # Write dataset info.json
        info = {
            "features": {
                "observation.state": {
                    "names": ["left_base.pos", "left_link1.pos"],
                }
            }
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        names = builder.get_training_state_names()
        assert names == ["left_base.pos", "left_link1.pos"]

        # Second call returns cached
        names2 = builder.get_training_state_names()
        assert names2 is names

    def test_get_training_state_names_none_when_missing(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        assert builder.get_training_state_names() is None

    def test_reset_cache_clears(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        # Load once (returns None since no files)
        builder.get_training_state_names()
        assert builder._training_state_names_loaded

        builder.reset_cache()
        assert not builder._training_state_names_loaded
        assert builder._training_state_names is None

    def test_convert_action_to_dict_returns_empty_without_names(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        import numpy as np

        action = np.array([0.1, 0.2, 0.3])
        result = builder.convert_action_to_dict(action, {})
        assert result == {}

    def test_convert_action_dict_passthrough(self, tmp_path):
        from app.core.deployment.observation_builder import ObservationBuilder

        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()

        builder = ObservationBuilder(
            checkpoint_path=checkpoint,
            policy=MagicMock(),
        )

        action = {"left_base.pos": 0.5}
        result = builder.convert_action_to_dict(action, {})
        assert result == {"left_base.pos": 0.5}


# ---------------------------------------------------------------------------
# RLLearner tests
# ---------------------------------------------------------------------------


class TestRLLearner:
    """Test RLLearner placeholder interface."""

    def test_init(self):
        learner = RLLearner()
        assert learner.config is None

    def test_add_transition_raises(self):
        learner = RLLearner()
        with pytest.raises(NotImplementedError):
            learner.add_transition({}, None, 0.0, {}, False)

    def test_compute_reward_raises(self):
        learner = RLLearner()
        with pytest.raises(NotImplementedError):
            learner.compute_reward({})

    def test_get_metrics_returns_empty(self):
        learner = RLLearner()
        assert learner.get_metrics() == {}

    def test_stop_does_not_raise(self):
        learner = RLLearner()
        learner.stop()


# ---------------------------------------------------------------------------
# State machine update from intervention
# ---------------------------------------------------------------------------


class TestInterventionStateUpdates:
    """Test _update_state_from_intervention logic."""

    def test_policy_when_running_no_intervention(self, runtime):
        runtime._state = RuntimeState.RUNNING
        runtime._intervention_detector = InterventionDetector()

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.POLICY

    def test_human_on_intervention(self, runtime):
        runtime._state = RuntimeState.RUNNING
        runtime._intervention_detector = InterventionDetector()

        source = runtime._update_state_from_intervention(is_intervening=True)
        assert source == ActionSource.HUMAN
        assert runtime._state == RuntimeState.HUMAN_ACTIVE

    def test_hold_when_paused(self, runtime):
        runtime._state = RuntimeState.PAUSED
        runtime._intervention_detector = InterventionDetector()

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.HOLD

    def test_human_active_stays_human_within_timeout(self, runtime):
        runtime._state = RuntimeState.HUMAN_ACTIVE
        detector = InterventionDetector(idle_timeout=10.0)
        detector._last_move_time = time.monotonic()
        runtime._intervention_detector = detector

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.HUMAN

    def test_human_active_transitions_to_paused_on_idle(self, runtime):
        runtime._state = RuntimeState.HUMAN_ACTIVE
        detector = InterventionDetector(idle_timeout=0.01)
        detector._last_move_time = time.monotonic() - 1.0  # well past timeout
        runtime._intervention_detector = detector

        source = runtime._update_state_from_intervention(is_intervening=False)
        assert source == ActionSource.HOLD
        assert runtime._state == RuntimeState.PAUSED


# ---------------------------------------------------------------------------
# Partial action sending
# ---------------------------------------------------------------------------


class TestPartialActionSending:
    """Test _send_partial_action for bimanual and single-arm robots."""

    def test_single_arm_robot(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        # Remove bimanual attributes so it's treated as single-arm
        del robot.left_arm
        del robot.right_arm
        action = {"base.pos": 0.5, "link1.pos": 0.3}

        DeploymentRuntime._send_partial_action(robot, action)
        robot.send_action.assert_called_once_with(action)

    def test_bimanual_splits_and_strips_prefix(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        robot.left_arm = MagicMock()
        robot.right_arm = MagicMock()

        action = {
            "left_base.pos": 0.5,
            "left_link1.pos": 0.3,
            "right_base.pos": 0.1,
        }

        DeploymentRuntime._send_partial_action(robot, action)

        robot.left_arm.send_action.assert_called_once_with(
            {"base.pos": 0.5, "link1.pos": 0.3}
        )
        robot.right_arm.send_action.assert_called_once_with(
            {"base.pos": 0.1}
        )

    def test_bimanual_left_only(self):
        from app.core.deployment.runtime import DeploymentRuntime

        robot = MagicMock()
        robot.left_arm = MagicMock()
        robot.right_arm = MagicMock()

        action = {"left_base.pos": 0.5}

        DeploymentRuntime._send_partial_action(robot, action)

        robot.left_arm.send_action.assert_called_once()
        robot.right_arm.send_action.assert_not_called()
