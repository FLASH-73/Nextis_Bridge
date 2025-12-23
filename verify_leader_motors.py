
import sys
import os

# Add src to path
sys.path.append("/home/roberto/nextis_app/lerobot/src")

from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig
from lerobot.robots.umbra_follower.umbra_follower import UmbraFollowerRobot
from lerobot.motors.feetech import FeetechMotorsBus

print("Verifying UmbraFollower Configuration...")

try:
    config = UmbraFollowerConfig(port="/dev/ttyUSB0") # Port doesn't matter for init check
    robot = UmbraFollowerRobot(config)
    
    motors = robot.bus.motors
    print(f"Motors on bus: {list(motors.keys())}")
    
    has_link1 = "link1" in motors
    has_link2 = "link2" in motors
    has_link1_follower = "link1_follower" in motors
    
    if has_link1 and has_link2:
        print("SUCCESS: Leader motors 'link1' and 'link2' are present on the bus.")
    else:
        print("FAILURE: Leader motors missing from bus.")
        
    if has_link1_follower:
         print("SUCCESS: Follower motors present.")

except Exception as e:
    print(f"CRITICAL ERROR during initialization: {e}")
