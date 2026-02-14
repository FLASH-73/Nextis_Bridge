"""Tests for InterventionEngine and TaskOrchestrator.

Refactored from the original script-style tests to pytest conventions.
lerobot mocking is handled by conftest.py.
"""
import time
import threading

import numpy as np

from app.core.intervention import InterventionEngine
from app.core.recorder import DataRecorder


# ── Test-specific mocks ──


class MockRobot:
    def __init__(self):
        self.is_connected = True
        self.velocity = 0.0
        self.velocity_left = 0.0
        self.velocity_right = 0.0

    def capture_observation(self):
        return {
            "observation.velocity": np.array([self.velocity]),
            "observation.velocity_left": np.array([self.velocity_left]),
            "observation.velocity_right": np.array([self.velocity_right]),
            "observation.state": np.array([0.1, 0.2]),
            "observation.images.cam": np.zeros((3, 224, 224)),
        }

    def get_observation(self):
        return self.capture_observation()

    def capture_action(self):
        return np.array([0.5, 0.5])


class MockRecorder(DataRecorder):
    def __init__(self):
        self.recording = False
        self.episodes = 0

    def start_new_episode(self, desc):
        self.recording = True

    def save_frame(self, obs, action):
        pass

    def stop_episode(self):
        self.recording = False
        self.episodes += 1


# ── Tests ──


def test_intervention_no_movement():
    """Engine starts with no human control when robot is still."""
    robot = MockRobot()
    recorder = MockRecorder()
    engine = InterventionEngine(robot, recorder)

    t = threading.Thread(target=engine.start, daemon=True)
    t.start()
    try:
        time.sleep(0.5)
        assert not engine.is_human_controlling
    finally:
        engine.stop()
        t.join(timeout=2.0)


def test_intervention_detects_movement():
    """Engine detects human control when velocity exceeds threshold."""
    robot = MockRobot()
    recorder = MockRecorder()
    engine = InterventionEngine(robot, recorder)

    t = threading.Thread(target=engine.start, daemon=True)
    t.start()
    try:
        time.sleep(0.3)
        robot.velocity = 1.0
        time.sleep(0.5)
        assert engine.is_human_controlling
        assert recorder.recording
    finally:
        engine.stop()
        t.join(timeout=2.0)


def test_intervention_returns_to_idle():
    """Engine returns to idle after human stops moving."""
    robot = MockRobot()
    recorder = MockRecorder()
    engine = InterventionEngine(robot, recorder)

    t = threading.Thread(target=engine.start, daemon=True)
    t.start()
    try:
        robot.velocity = 1.0
        time.sleep(0.5)
        assert engine.is_human_controlling

        robot.velocity = 0.0
        time.sleep(engine.IDLE_TIMEOUT + 0.5)
        assert not engine.is_human_controlling
        assert not recorder.recording
        assert recorder.episodes == 1
    finally:
        engine.stop()
        t.join(timeout=2.0)


def test_intervention_bimanual_trigger():
    """Bimanual: left arm velocity triggers intervention."""
    robot = MockRobot()
    recorder = MockRecorder()
    engine = InterventionEngine(robot, recorder)

    t = threading.Thread(target=engine.start, daemon=True)
    t.start()
    try:
        robot.velocity_left = 1.0
        time.sleep(0.5)
        assert engine.is_human_controlling

        robot.velocity_left = 0.0
        time.sleep(engine.IDLE_TIMEOUT + 0.5)
        assert not engine.is_human_controlling
    finally:
        engine.stop()
        t.join(timeout=2.0)


def test_orchestrator_starts_and_stops():
    """Orchestrator loads tasks, starts, and stops cleanly."""
    from app.core.orchestrator import TaskOrchestrator

    robot = MockRobot()
    recorder = MockRecorder()
    orch = TaskOrchestrator(robot, recorder)

    orch.load_task_chain(["task_A", "task_B"])
    orch.start()
    try:
        time.sleep(0.5)
        assert orch.task_chain == ["task_A", "task_B"]
    finally:
        orch.stop()
