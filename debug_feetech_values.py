from lerobot.robots.umbra_follower.umbra_follower import UmbraFollowerRobot
from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig
import time

def debug_values():
    print("Initializing UmbraFollowerRobot...")
    # Initialize with default config (assuming port is auto-detected or default)
    # Using 'left' as default
    config = UmbraFollowerConfig(arm_side="left")
    robot = UmbraFollowerRobot(config)
    
    print("Connecting to bus...")
    robot.bus.connect()
    
    print("\n--- Motor Debug Info ---")
    print(f"{'Motor':<15} | {'ID':<3} | {'Present_Pos':<12} | {'Offset':<8} | {'Min_Lim':<8} | {'Max_Lim':<8}")
    print("-" * 70)
    
    motors = robot.bus.motors.keys()
    
    # Read values
    present_positions = robot.bus.sync_read("Present_Position", normalize=False)
    
    # For registers that might not support sync_read or we want individual confirm
    for motor in motors:
        motor_obj = robot.bus.motors[motor]
        try:
            offset = robot.bus.read("Homing_Offset", motor, normalize=False)
            min_lim = robot.bus.read("Min_Position_Limit", motor, normalize=False)
            max_lim = robot.bus.read("Max_Position_Limit", motor, normalize=False)
            pos = present_positions.get(motor, "N/A")
            
            print(f"{motor:<15} | {motor_obj.id:<3} | {pos:<12} | {offset:<8} | {min_lim:<8} | {max_lim:<8}")
        except Exception as e:
            print(f"{motor:<15} | {motor_obj.id:<3} | ERROR: {e}")

    print("\nDone.")
    robot.bus.disconnect()

if __name__ == "__main__":
    debug_values()
