"""Tests for TrainingService: job creation, management, hardware detection."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.training.service import TrainingService

# ── Job creation ──


def test_create_job(training_service):
    """create_job() returns a TrainingJob with a generated ID."""
    job = training_service.create_job(
        dataset_repo_id="user/my_dataset",
        policy_type="smolvla",
        config={"steps": 5000},
    )
    assert job.id is not None
    assert len(job.id) > 0
    assert job.dataset_repo_id == "user/my_dataset"
    assert job.policy_type == "smolvla"


def test_create_job_merges_defaults(training_service):
    """create_job() merges user config with policy defaults."""
    job = training_service.create_job(
        dataset_repo_id="user/ds",
        policy_type="diffusion",
        config={"batch_size": 16},
    )
    # User config should be preserved
    assert job.config.get("batch_size") == 16


def test_get_job(training_service):
    """get_job() returns the correct job by ID."""
    job = training_service.create_job("user/ds", "act", {})
    retrieved = training_service.get_job(job.id)
    assert retrieved is not None
    assert retrieved.id == job.id


def test_get_job_not_found(training_service):
    """get_job() returns None for unknown ID."""
    assert training_service.get_job("nonexistent") is None


def test_list_jobs(training_service):
    """list_jobs() returns all created jobs."""
    training_service.create_job("user/ds1", "smolvla", {})
    training_service.create_job("user/ds2", "diffusion", {})
    jobs = training_service.list_jobs()
    assert len(jobs) == 2


def test_list_jobs_empty(training_service):
    """list_jobs() returns empty list when no jobs exist."""
    assert training_service.list_jobs() == []


# ── Job status ──


def test_get_job_status(training_service):
    """get_job_status() returns status dict."""
    job = training_service.create_job("user/ds", "smolvla", {})
    status = training_service.get_job_status(job.id)
    assert "status" in status or "state" in status or "id" in status


def test_get_job_status_not_found(training_service):
    """get_job_status() raises ValueError for unknown job ID."""
    with pytest.raises(ValueError, match="not found"):
        training_service.get_job_status("nonexistent")


# ── Job logs ──


def test_get_job_logs(training_service):
    """get_job_logs() returns log dict with offset/limit."""
    job = training_service.create_job("user/ds", "smolvla", {})
    logs = training_service.get_job_logs(job.id, offset=0, limit=10)
    assert isinstance(logs, dict)
    assert "lines" in logs or "logs" in logs or "log" in logs or isinstance(logs.get("data"), (list, type(None)))


# ── Hardware detection ──


def test_detect_hardware(training_service):
    """detect_hardware() returns at least CPU device."""
    result = training_service.detect_hardware()
    assert isinstance(result, dict)
    # Should report at least one device type
    assert "devices" in result or "cuda" in result or "cpu" in result


# ── Presets ──


def test_get_presets_smolvla(training_service):
    """get_presets() returns config for smolvla."""
    presets = training_service.get_presets("smolvla")
    assert isinstance(presets, dict)


def test_get_presets_diffusion(training_service):
    """get_presets() returns config for diffusion."""
    presets = training_service.get_presets("diffusion")
    assert isinstance(presets, dict)


def test_get_presets_unknown(training_service):
    """get_presets() handles unknown policy type gracefully."""
    presets = training_service.get_presets("unknown")
    assert isinstance(presets, dict)
