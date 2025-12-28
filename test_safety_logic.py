
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "lerobot/src"))

from app.core.safety_layer import SafetyLayer
from lerobot.robots.bi_umbra_follower.config_bi_umbra_follower import BiUmbraFollowerConfig
from lerobot.robots.utils import make_robot_from_config
from app.core.config import load_config
from unittest.mock import MagicMock

# This script mocks the robot connection just to test the logic of SafetyLayer parsing
# OR tries to connect to real robot if possible. Since backend is running, we might clash ports.
# So we mocked the bus in this test to verify the PARSING logic.

def test_safety_parsing():
    print("Testing Safety Layer Logic...")
    
    # Mock Robot and Bus
    robot = MagicMock()
    bus = MagicMock()
    robot.bus = bus
    robot.is_connected = True
    
    # Setup Safety
    safety = SafetyLayer(robot_lock=None)
    
    # CASE 1: Normal Load
    print("\nCase 1: Normal Load (100)")
    bus.sync_read.return_value = {"motor1": 100, "motor2": 200}
    result = safety.check_limits(robot)
    print(f"Result: {result} (Expected True)")
    assert result == True
    
    # CASE 2: High Load (Warning)
    print("\nCase 2: High Load (600) - count 1")
    bus.sync_read.return_value = {"motor1": 600}
    result = safety.check_limits(robot)
    print(f"Result: {result} (Expected True, Warning Logged)")
    assert result == True
    
    # CASE 3: High Load (Critical) - count 3
    print("\nCase 3: High Load (600) - count 3")
    safety.check_limits(robot) # 2
    result = safety.check_limits(robot) # 3 -> Trigger
    print(f"Result: {result} (Expected False, E-Stop Triggered)")
    assert result == False
    
    print("\nSafety Logic Verified.")

if __name__ == "__main__":
    test_safety_parsing()
