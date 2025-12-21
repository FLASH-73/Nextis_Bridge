
import time
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.orchestrator import TaskOrchestrator
from app.core.recorder import DataRecorder

def test_orchestrator():
    print("Initializing components...")
    mock_robot = MagicMock()
    mock_robot.is_connected = True
    mock_robot.get_observation.return_value = {"observation.state": [0.0, 0.0, 0.0]}
    
    # Mock recorder to avoid actual disk I/O in test if desired, 
    # but we also want to verify it TRIES to save.
    mock_recorder = MagicMock() 
    
    orchestrator = TaskOrchestrator(mock_robot, mock_recorder)
    orchestrator.TASK_DURATION = 1.0 # Speed up for test
    
    print("Starting Orchestrator...")
    orchestrator.start()
    
    plan = [
        {"task": "move_to_bin", "params": {"bin_id": "A"}},
        {"task": "pick_object", "params": {"object_name": "apple"}}
    ]
    
    print("Executing Plan...")
    orchestrator.execute_plan(plan)
    
    # Wait for first task
    time.sleep(1.5)
    print(f"Current Task Index: {orchestrator.current_task_index}")
    
    if orchestrator.current_task_index >= 1:
        print("✅ Task advanced successfully.")
    else:
        print("❌ Task did not advance.")
        
    # Wait for second task
    time.sleep(1.5)
    
    if not orchestrator.is_executing_plan:
        print("✅ Plan execution finished.")
    else:
        print("❌ Plan execution stuck.")
        
    orchestrator.stop()

if __name__ == "__main__":
    test_orchestrator()
