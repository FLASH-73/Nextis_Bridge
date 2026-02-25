"""Tests for HILService: session lifecycle, episode management, intervention."""

import threading
from unittest.mock import MagicMock, PropertyMock

import pytest

from app.core.hil.service import HILService
from app.core.hil.types import HILMode, HILSessionState

# ── Fixtures ──


@pytest.fixture
def mock_teleop():
    teleop = MagicMock()
    teleop.episode_count = 0
    teleop.start_recording_session = MagicMock()
    teleop.stop_recording_session = MagicMock()
    teleop.start_episode = MagicMock()
    teleop.stop_episode = MagicMock()
    return teleop


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.deployed_policy = None
    orch.deployed_policy_path = None
    orch.robot = MagicMock()
    orch.deploy_policy = MagicMock()
    return orch


@pytest.fixture
def mock_training():
    training = MagicMock()
    policy = MagicMock()
    policy.checkpoint_path = "/tmp/checkpoint"
    policy.policy_type = "smolvla"
    training.get_policy.return_value = policy

    policy_config = MagicMock()
    policy_config.cameras = ["camera1"]
    policy_config.arms = ["left"]
    policy_config.policy_type = "smolvla"
    training.get_policy_config.return_value = policy_config

    job = MagicMock()
    job.id = "job_123"
    training.create_job.return_value = job
    training.start_job = MagicMock()
    return training


@pytest.fixture
def hil_service(mock_teleop, mock_orchestrator, mock_training):
    return HILService(
        teleop_service=mock_teleop,
        orchestrator=mock_orchestrator,
        training_service=mock_training,
        robot_lock=threading.Lock(),
    )


# ── Session lifecycle ──


def test_start_session(hil_service, mock_orchestrator, mock_teleop):
    """start_session() deploys policy and starts recording."""
    result = hil_service.start_session(
        policy_id="policy_1",
        intervention_dataset="user/interventions",
        task="pick up block",
    )
    assert result["status"] == "started"
    assert hil_service.state.active is True
    mock_orchestrator.deploy_policy.assert_called_once()
    mock_teleop.start_recording_session.assert_called_once()


def test_start_session_already_active(hil_service):
    """start_session() raises when session is already active."""
    hil_service.start_session("p1", "ds", "task")
    with pytest.raises(Exception, match="already active"):
        hil_service.start_session("p2", "ds2", "task2")


def test_start_session_policy_not_found(hil_service, mock_training):
    """start_session() raises when policy is not found."""
    mock_training.get_policy.return_value = None
    with pytest.raises(Exception, match="not found"):
        hil_service.start_session("nonexistent", "ds", "task")


def test_start_session_no_checkpoint(hil_service, mock_training):
    """start_session() raises when policy has no checkpoint."""
    mock_training.get_policy.return_value.checkpoint_path = None
    with pytest.raises(Exception, match="no checkpoint"):
        hil_service.start_session("p1", "ds", "task")


def test_start_session_movement_scale_clamped(hil_service):
    """movement_scale is clamped to [0.1, 1.0]."""
    hil_service.start_session("p1", "ds", "task", movement_scale=5.0)
    assert hil_service.state.movement_scale == 1.0

    # Reset for next test
    hil_service.stop_session()
    hil_service.start_session("p1", "ds", "task", movement_scale=0.01)
    assert hil_service.state.movement_scale == 0.1


def test_stop_session(hil_service):
    """stop_session() cleans up state."""
    hil_service.start_session("p1", "ds", "task")
    result = hil_service.stop_session()
    assert result["status"] == "stopped"
    assert hil_service.state.active is False


def test_stop_session_not_active(hil_service):
    """stop_session() returns not_active when no session."""
    result = hil_service.stop_session()
    assert result["status"] == "not_active"


# ── Episode lifecycle ──


def test_start_episode(hil_service, mock_teleop):
    """start_episode() transitions to AUTONOMOUS mode."""
    hil_service.start_session("p1", "ds", "task")
    result = hil_service.start_episode()
    assert result["status"] == "started"
    assert hil_service.state.episode_active is True
    assert hil_service.state.mode == HILMode.AUTONOMOUS
    mock_teleop.start_episode.assert_called_once()


def test_start_episode_no_session(hil_service):
    """start_episode() raises when no active session."""
    with pytest.raises(Exception, match="No active"):
        hil_service.start_episode()


def test_stop_episode(hil_service, mock_teleop):
    """stop_episode() saves episode and transitions to IDLE."""
    hil_service.start_session("p1", "ds", "task")
    hil_service.start_episode()
    result = hil_service.stop_episode()
    assert result["status"] == "saved"
    assert hil_service.state.episode_active is False
    assert hil_service.state.mode == HILMode.IDLE


def test_stop_episode_not_recording(hil_service):
    """stop_episode() returns not_recording when no active episode."""
    hil_service.start_session("p1", "ds", "task")
    result = hil_service.stop_episode()
    assert result["status"] == "not_recording"


# ── Resume autonomous ──


def test_resume_from_paused(hil_service):
    """resume_autonomous() transitions from PAUSED to AUTONOMOUS."""
    hil_service.start_session("p1", "ds", "task")
    hil_service.state.mode = HILMode.PAUSED
    result = hil_service.resume_autonomous()
    assert result["status"] == "resumed"
    assert hil_service.state.mode == HILMode.AUTONOMOUS


def test_resume_already_autonomous(hil_service):
    """resume_autonomous() returns already_autonomous if already running."""
    hil_service.start_session("p1", "ds", "task")
    hil_service.start_episode()  # Sets mode to AUTONOMOUS
    result = hil_service.resume_autonomous()
    assert result["status"] == "already_autonomous"


def test_resume_from_idle(hil_service):
    """resume_autonomous() returns error from IDLE state."""
    hil_service.start_session("p1", "ds", "task")
    result = hil_service.resume_autonomous()
    assert result["status"] == "error"


# ── Status ──


def test_get_status_fields(hil_service):
    """get_status() returns all expected fields."""
    status = hil_service.get_status()
    expected_keys = {
        "active", "mode", "policy_id", "intervention_dataset", "task",
        "episode_active", "episode_count", "intervention_count",
        "current_episode_interventions", "autonomous_frames", "human_frames",
        "policy_config", "movement_scale",
    }
    assert expected_keys.issubset(set(status.keys()))


def test_get_status_after_session(hil_service):
    """get_status() reflects session state."""
    hil_service.start_session("p1", "ds_name", "my_task")
    status = hil_service.get_status()
    assert status["active"] is True
    assert status["policy_id"] == "p1"
    assert status["intervention_dataset"] == "ds_name"
    assert status["task"] == "my_task"


# ── Retrain ──


def test_trigger_retrain(hil_service, mock_training):
    """trigger_retrain() creates a fine-tuning job."""
    hil_service.start_session("p1", "ds", "task")
    result = hil_service.trigger_retrain()
    assert result["status"] == "started"
    assert result["job_id"] == "job_123"
    mock_training.create_job.assert_called_once()
    mock_training.start_job.assert_called_once()


def test_trigger_retrain_no_dataset(hil_service):
    """trigger_retrain() raises when no intervention dataset."""
    with pytest.raises(Exception, match="No intervention dataset"):
        hil_service.trigger_retrain()
