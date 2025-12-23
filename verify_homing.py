from lerobot.robots.umbra_follower.umbra_follower import UmbraFollowerRobot
from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig
import time

def verify_homing():
    print("Initialize...")
    config = UmbraFollowerConfig(arm_side="left")
    robot = UmbraFollowerRobot(config)
    robot.bus.connect()
    
    target_motor = "link1" # One of the leaders
    if target_motor not in robot.bus.motors:
        target_motor = "base" # Fallback
        
    print(f"Targeting motor: {target_motor}")
    
    # Pre-check
    pos_pre = robot.bus.read("Present_Position", target_motor, normalize=False)
    offset_pre = robot.bus.read("Homing_Offset", target_motor, normalize=False)
    print(f"PRE-HOMING: Pos={pos_pre}, Offset={offset_pre}")
    
    print("Performing set_half_turn_homings...")
    # This is what calibration_service calls (via perform_homing calling bus directly)
    # Note: Service calls it on a list of motors.
    robot.bus.set_half_turn_homings([target_motor])
    
    time.sleep(1.0)
    
    # Post-check
    pos_post = robot.bus.read("Present_Position", target_motor, normalize=False)
    offset_post = robot.bus.read("Homing_Offset", target_motor, normalize=False)
    print(f"POST-HOMING: Pos={pos_post}, Offset={offset_post}")
    
    expected = 2048
    diff = abs(pos_post - expected)
    print(f"Diff from 2048: {diff}")
    
    if diff < 20:
        print("SUCCESS: Motor homed correctly.")
    else:
        print("FAILURE: Motor did not home to 2048.")

    robot.bus.disconnect()

if __name__ == "__main__":
    verify_homing()
