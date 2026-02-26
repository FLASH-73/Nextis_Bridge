"""Shared test fixtures for Nextis Bridge backend tests.

CRITICAL: lerobot mocks MUST be installed at module level (before any app imports).
This file is loaded by pytest before any test module, so the mocks are in place globally.
"""
import sys
from unittest.mock import MagicMock

# ── lerobot sys.modules mocks (must happen BEFORE any app imports) ──
# The app imports lerobot at module level in several places.
# We mock every module in the import chain so Python never tries to load the real packages.

_LEROBOT_MODULES = [
    # Core
    "lerobot",
    # Robots
    "lerobot.robots",
    "lerobot.robots.utils",
    "lerobot.robots.bi_umbra_follower",
    "lerobot.robots.bi_umbra_follower.bi_umbra_follower",
    "lerobot.robots.umbra_follower",
    "lerobot.robots.umbra_follower.umbra_follower",
    "lerobot.robots.damiao_follower",
    "lerobot.robots.damiao_follower.damiao_follower",
    # Datasets
    "lerobot.datasets",
    "lerobot.datasets.lerobot_dataset",
    "lerobot.datasets.aggregate",
    "lerobot.datasets.v30",
    "lerobot.datasets.v30.augment_dataset_quantile_stats",
    # Motors
    "lerobot.motors",
    "lerobot.motors.feetech",
    "lerobot.motors.feetech.feetech",
    "lerobot.motors.dynamixel",
    "lerobot.motors.dynamixel.dynamixel",
    "lerobot.motors.damiao",
    "lerobot.motors.damiao.damiao",
    # Cameras
    "lerobot.cameras",
    "lerobot.cameras.opencv",
    "lerobot.cameras.opencv.camera_opencv",
    "lerobot.cameras.realsense",
    "lerobot.cameras.realsense.camera_realsense",
    # Model
    "lerobot.model",
    "lerobot.model.kinematics",
]

for _mod in _LEROBOT_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make DamiaoMotorsBus a real class so isinstance() checks in safety.py work
_damiao_mod = sys.modules["lerobot.motors.damiao.damiao"]
_damiao_mod.DamiaoMotorsBus = type("DamiaoMotorsBus", (), {})

# Also mock optional heavy deps that may not be present in CI
for _mod in [
    "accelerate",
    "torch",
    "transformers",
    "can",
    "cv2",
    "serial",
    "serial.tools",
    "serial.tools.list_ports",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make torch.Tensor a real class so isinstance() checks work
_torch_mod = sys.modules["torch"]
_torch_mod.Tensor = type("Tensor", (), {})

# ── Now safe to import app code ──

import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from app.core.hardware.types import ArmDefinition, ArmRole, MotorType, Pairing  # noqa: E402

# ── Fixtures ──


@pytest.fixture
def robot_lock():
    """A threading.Lock for tests that need one."""
    return threading.Lock()


@pytest.fixture
def safety_layer(robot_lock):
    """A fresh SafetyLayer instance."""
    from app.core.hardware.safety import SafetyLayer
    return SafetyLayer(robot_lock)


@pytest.fixture
def mock_robot():
    """A MagicMock robot with standard attributes."""
    robot = MagicMock()
    robot.is_connected = True
    robot.is_mock = False
    robot.is_calibrated = True
    return robot


@pytest.fixture
def sample_arm_definition():
    """A sample ArmDefinition for testing."""
    return ArmDefinition(
        id="test_follower",
        name="Test Follower",
        role=ArmRole.FOLLOWER,
        motor_type=MotorType.STS3215,
        port="/dev/ttyUSB0",
        enabled=True,
        structural_design="umbra_7dof",
    )


@pytest.fixture
def sample_pairing():
    """A sample Pairing for testing."""
    return Pairing(
        leader_id="test_leader",
        follower_id="test_follower",
        name="Test Pair",
    )


@pytest.fixture
def training_service(tmp_path):
    """A TrainingService pointing to temporary directories."""
    from app.core.training.service import TrainingService
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return TrainingService(datasets_path=datasets, outputs_path=outputs)


@pytest.fixture
def app_client():
    """FastAPI TestClient with fully mocked SystemState.

    Patches app.dependencies.get_state so routes get a controlled mock
    instead of trying to initialize real hardware.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    mock_state = MagicMock()
    mock_state.is_initializing = False
    mock_state.init_error = None
    mock_state.robot = MagicMock()
    mock_state.robot.is_connected = False
    mock_state.robot.is_mock = True
    mock_state.arm_registry = None
    mock_state.teleop_service = MagicMock()
    mock_state.teleop_service.is_running = False
    mock_state.training_service = MagicMock()
    mock_state.dataset_service = MagicMock()
    mock_state.dataset_service.list_datasets.return_value = []
    mock_state.camera_service = MagicMock()
    mock_state.hil_service = MagicMock()
    mock_state.hil_service.get_status.return_value = {
        "active": False, "mode": "idle", "policy_id": "",
        "intervention_dataset": "", "task": "", "episode_active": False,
        "episode_count": 0, "intervention_count": 0,
        "current_episode_interventions": 0, "autonomous_frames": 0,
        "human_frames": 0, "policy_config": {"cameras": [], "arms": [], "type": ""},
        "movement_scale": 1.0,
    }
    mock_state.orchestrator = MagicMock()
    mock_state.orchestrator.active_policy = None
    mock_state.orchestrator.intervention_engine = MagicMock()
    mock_state.orchestrator.intervention_engine.is_human_controlling = False
    mock_state.reward_classifier_service = MagicMock()
    mock_state.gvl_reward_service = MagicMock()
    mock_state.sarm_reward_service = MagicMock()
    mock_state.rl_service = None
    mock_state.calibration_service = MagicMock()
    mock_state.recorder = MagicMock()
    mock_state.leader_assists = {}

    # Patch both the get_state function AND the module-level state reference
    # that get_state() returns (from app.state import state captures a reference).
    with patch("app.dependencies.get_state", return_value=mock_state), \
         patch("app.dependencies.state", mock_state), \
         patch("app.state.state", mock_state):
        client = TestClient(app, raise_server_exceptions=False)
        client._mock_state = mock_state  # expose for test assertions
        yield client
