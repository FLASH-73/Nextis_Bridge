import sys
import os
import threading
import queue
import time
from unittest.mock import MagicMock

# Add app to path
sys.path.append(os.getcwd())

# Mock dependencies before importing teleop_service
sys.modules['app.core.safety_layer'] = MagicMock()

# Setup hierarchical mocks for lerobot imports
lerobot_mock = MagicMock()
sys.modules['lerobot'] = lerobot_mock
sys.modules['lerobot.common'] = MagicMock()
sys.modules['lerobot.common.robot_devices'] = MagicMock()
sys.modules['lerobot.common.robot_devices.robots'] = MagicMock()
sys.modules['lerobot.common.robot_devices.robots'] = MagicMock()
sys.modules['lerobot.common.datasets'] = MagicMock()
sys.modules['lerobot.common.datasets.lerobot_dataset'] = MagicMock()

# Mock internal utils for imports in teleop_service
sys.modules['lerobot.utils'] = MagicMock()
sys.modules['lerobot.utils.robot_utils'] = MagicMock()
sys.modules['lerobot.utils.constants'] = MagicMock()
sys.modules['lerobot.datasets'] = MagicMock()
sys.modules['lerobot.datasets.lerobot_dataset'] = MagicMock()
sys.modules['lerobot.datasets.video_utils'] = MagicMock()
sys.modules['lerobot.datasets.utils'] = MagicMock() # Crucial for build_dataset_frame

# Mock VIDEO_ENCODING_MANAGER? 
# usage: from lerobot.datasets.video_utils import VideoEncodingManager

# Mock motors/feetech specifically
sys.modules['lerobot.motors'] = MagicMock()
sys.modules['lerobot.motors.feetech'] = MagicMock()
feetech_module = MagicMock()
feetech_module.OperatingMode = MagicMock() # Mock the class/enum
sys.modules['lerobot.motors.feetech.feetech'] = feetech_module

# Ensure LeaderAssistService is mocked if imported from safety_layer or elsewhere
# Looking at the code, it seems to assume LeaderAssistService is available. 
# It is likely imported from app.core.teleop_service or safety_layer?
# Let's assume it's imported from somewhere. 
# If it's a local class, we might have issue. 
# If it is imported, we need to mock it in sys.modules or the imports of teleop_service
# We'll just patch app.core.safety_layer fully
app_core_safety = sys.modules['app.core.safety_layer']
# Use lambda to avoid MagicMock(arg) treating arg as spec
app_core_safety.SafetyLayer = lambda *args, **kwargs: MagicMock()
app_core_safety.LeaderAssistService = lambda *args, **kwargs: MagicMock()

from app.core.teleop_service import TeleoperationService

def test_threaded_recording():
    print("Initializing TeleoperationService...")
    # Pass all required args
    service = TeleoperationService(robot=MagicMock(), leader=MagicMock(), robot_lock=MagicMock())
    
    # Verify Queue exists
    if not hasattr(service, 'recording_queue'):
        print("FAILURE: recording_queue not found.")
        return
        
    print("SUCCESS: recording_queue exists.")
    
    # Verify Worker exists
    if not hasattr(service, '_recording_worker'):
        print("FAILURE: _recording_worker method not found.")
        return
        
    # Test Enqueue
    print("Testing Enqueue...")
    test_item = {
        "type": "frame",
        "observation": {},
        "action": {},
        "task": "test",
        "loop_count": 0
    }
    service.recording_queue.put(test_item)
    
    if service.recording_queue.qsize() != 1:
        print("FAILURE: Item not added to queue.")
        return
    print("SUCCESS: Item added to queue.")
    
    # Test Worker Processing
    print("Testing Worker...")
    service.dataset = MagicMock()
    service.recording_thread_running = True
    
    # Start worker in separate thread
    t = threading.Thread(target=service._recording_worker, daemon=True)
    t.start()
    
    time.sleep(1) # Give it time to process
    
    if service.recording_queue.qsize() != 0:
        print("FAILURE: Queue not drained by worker.")
    else:
        print("SUCCESS: Queue drained by worker.")
        
    # Check if dataset.add_frame was called
    if service.dataset.add_frame.called:
        print("SUCCESS: dataset.add_frame was called.")
    else:
        print("FAILURE: dataset.add_frame was NOT called.")

    # Test Stop Episode
    print("Testing Stop Episode...")
    service.recording_queue.put({"type": "stop_episode"})
    time.sleep(1) # Wait for worker
    
    if service.dataset.save_episode.called:
        print("SUCCESS: dataset.save_episode was called.")
    else:
        print("FAILURE: dataset.save_episode was NOT called.")
        
    # Cleanup
    service.recording_thread_running = False
    service.recording_queue.put(None)
    t.join()
    print("Test Complete.")

if __name__ == "__main__":
    test_threaded_recording()
