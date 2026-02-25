"""FastAPI route smoke tests using TestClient with mocked SystemState."""

from unittest.mock import MagicMock

import pytest

# ── System routes ──


def test_root(app_client):
    """GET / returns online status."""
    response = app_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "nextis-robotics"


def test_ping(app_client):
    """GET /test/ping returns pong."""
    response = app_client.get("/test/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "pong"


def test_status_mock_mode(app_client):
    """GET /status returns MOCK connection when robot is mock."""
    app_client._mock_state.robot.is_mock = True
    app_client._mock_state.robot.is_connected = False
    response = app_client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["connection"] == "MOCK"


def test_status_initializing(app_client):
    """GET /status returns INITIALIZING during startup."""
    app_client._mock_state.is_initializing = True
    response = app_client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["connection"] == "INITIALIZING"


def test_status_has_required_fields(app_client):
    """GET /status returns connection, execution, and error fields."""
    response = app_client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "connection" in data
    assert "execution" in data
    assert "error" in data


def test_config(app_client):
    """GET /config returns config data."""
    response = app_client.get("/config")
    assert response.status_code == 200


# ── Emergency stop ──


def test_emergency_stop(app_client):
    """POST /emergency/stop returns success."""
    response = app_client.post("/emergency/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("success", "partial_success")


# ── Arms routes ──


def test_arms_no_registry(app_client):
    """GET /arms works even with arm_registry=None."""
    app_client._mock_state.arm_registry = None
    response = app_client.get("/arms")
    assert response.status_code == 200


def test_arms_with_registry(app_client):
    """GET /arms returns arm data from registry."""
    mock_reg = MagicMock()
    mock_reg.get_all_arms.return_value = [
        {"id": "arm1", "name": "Arm 1", "role": "follower", "status": "disconnected"},
    ]
    mock_reg.get_status_summary.return_value = {"total_arms": 1, "leaders": 0, "followers": 1}
    app_client._mock_state.arm_registry = mock_reg
    response = app_client.get("/arms")
    assert response.status_code == 200


def test_arms_pairings(app_client):
    """GET /arms/pairings returns pairings from registry."""
    mock_reg = MagicMock()
    mock_reg.get_pairings.return_value = []
    app_client._mock_state.arm_registry = mock_reg
    response = app_client.get("/arms/pairings")
    assert response.status_code == 200


# ── Teleop routes ──


def test_teleop_status(app_client):
    """GET /teleop/status returns teleop state."""
    response = app_client.get("/teleop/status")
    assert response.status_code == 200


# ── Camera routes ──


def test_cameras_status(app_client):
    """GET /cameras/status returns camera status."""
    app_client._mock_state.camera_service.get_camera_status.return_value = {}
    response = app_client.get("/cameras/status")
    assert response.status_code == 200


# ── Dataset routes ──


def test_datasets_list(app_client):
    """GET /datasets returns dataset list."""
    app_client._mock_state.dataset_service.list_datasets.return_value = []
    response = app_client.get("/datasets")
    assert response.status_code == 200
    assert response.json() == [] or isinstance(response.json(), list)


# ── Training routes ──


def test_training_hardware(app_client):
    """GET /training/hardware returns hardware info."""
    app_client._mock_state.training_service.detect_hardware.return_value = {"cpu": True}
    response = app_client.get("/training/hardware")
    assert response.status_code == 200


def test_training_jobs(app_client):
    """GET /training/jobs returns job list."""
    app_client._mock_state.training_service.list_jobs.return_value = []
    response = app_client.get("/training/jobs")
    assert response.status_code == 200


# ── HIL routes ──


def test_hil_status(app_client):
    """GET /hil/status returns HIL session status."""
    response = app_client.get("/hil/status")
    assert response.status_code == 200
    data = response.json()
    assert "active" in data


# ── Recording routes ──


def test_recording_status(app_client):
    """GET /recording/status returns recording state."""
    response = app_client.get("/recording/status")
    assert response.status_code == 200


# ── Policy routes ──


def test_policies_list(app_client):
    """GET /policies returns policy list."""
    app_client._mock_state.training_service.list_policies.return_value = []
    response = app_client.get("/policies")
    assert response.status_code == 200
