import sys
from unittest.mock import MagicMock

# MOCK DEPENDENCIES BEFORE IMPORTING APP CODE
# This is necessary because the environment is missing 'accelerate' and other libs
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.robots"] = MagicMock()
sys.modules["lerobot.robots.utils"] = MagicMock()
sys.modules["lerobot.datasets"] = MagicMock()
sys.modules["lerobot.datasets.lerobot_dataset"] = MagicMock()

import time
import threading
import numpy as np
from app.core.intervention_controller import InterventionEngine
from app.core.recorder import DataRecorder

# Mock Robot
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
            "observation.images.cam": np.zeros((3, 224, 224))
        }
    
    def capture_action(self):
        return np.array([0.5, 0.5])

# Mock Recorder
class MockRecorder(DataRecorder):
    def __init__(self, repo_id="test", robot_type="test"):
        self.recording = False
        self.episodes = 0
    
    def start_new_episode(self, desc):
        print(f"[MockRecorder] Start Episode: {desc}")
        self.recording = True
        
    def save_frame(self, obs, action):
        if self.recording:
            pass
            
    def stop_episode(self):
        print("[MockRecorder] Stop Episode")
        self.recording = False
        self.episodes += 1

def test_intervention_logic():
    print("\n--- Testing Intervention Engine ---")
    robot = MockRobot()
    recorder = MockRecorder()
    engine = InterventionEngine(robot, recorder)
    
    # Start engine in a thread
    t = threading.Thread(target=engine.start)
    t.start()
    
    print("1. Normal operation (no movement)...")
    time.sleep(1)
    assert not engine.is_human_controlling
    
    print("2. Human moves robot (Velocity > Threshold)...")
    robot.velocity = 1.0 # High velocity
    time.sleep(0.5)
    assert engine.is_human_controlling
    assert recorder.recording
    
    print("3. Human stops moving...")
    robot.velocity = 0.0
    time.sleep(engine.IDLE_TIMEOUT + 0.5)
    
    assert not engine.is_human_controlling
    assert not recorder.recording
    assert recorder.episodes == 1
    
    print("4. Testing Bimanual Trigger (Left Arm)...")
    robot.velocity_left = 1.0
    time.sleep(0.5)
    assert engine.is_human_controlling
    
    robot.velocity_left = 0.0
    time.sleep(engine.IDLE_TIMEOUT + 0.5)
    assert not engine.is_human_controlling
    
    print("5. Stopping engine...")
    engine.stop()
    t.join()
    print("Intervention Test Passed!")

def test_orchestrator():
    print("\n--- Testing Orchestrator ---")
    from app.core.orchestrator import TaskOrchestrator
    
    robot = MockRobot()
    recorder = MockRecorder()
    orch = TaskOrchestrator(robot, recorder)
    
    orch.load_task_chain(["task_A", "task_B"])
    orch.start()
    
    print("1. Orchestrator running with Task A...")
    time.sleep(1)
    assert orch.active_policy == "dummy_model_object"
    
    print("2. Triggering Intervention...")
    robot.velocity_right = 1.0
    time.sleep(0.5)
    
    # Orchestrator should still be running, but InterventionEngine should be controlling
    assert orch.intervention_engine.is_human_controlling
    
    print("3. Releasing Intervention...")
    robot.velocity_right = 0.0
    time.sleep(2.5) # Wait for timeout
    assert not orch.intervention_engine.is_human_controlling
    
    print("4. Advancing Task...")
    orch.advance_task()
    assert orch.current_task_index == 1
    
    print("5. Stopping Orchestrator...")
    orch.stop()
    print("Orchestrator Test Passed!")

if __name__ == "__main__":
    test_intervention_logic()
    test_orchestrator()
