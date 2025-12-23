from lerobot.robots.umbra_follower.umbra_follower import UmbraFollowerRobot
from lerobot.robots.umbra_follower.config_umbra_follower import UmbraFollowerConfig
import time

def reproduce_homing():
    print("Initialize...")
    config = UmbraFollowerConfig(arm_side="left")
    robot = UmbraFollowerRobot(config)
    robot.bus.connect()
    
    # Target one motor
    motor_name = "link1"
    
    # 1. Disable Torque
    print("Disabling torque...")
    robot.bus.disable_torque([motor_name])
    
    # 2. Set Offset to 0
    print("Reseting Offset to 0...")
    robot.bus.write("Homing_Offset", motor_name, 0)
    time.sleep(0.5)
    
    # 3. Read Raw
    raw = robot.bus.read("Present_Position", motor_name, normalize=False)
    offset_read = robot.bus.read("Homing_Offset", motor_name, normalize=False)
    print(f"RAW (Offset=0): {raw}, Offset_Reg={offset_read}")
    
    # 4. Calculate Goal Offset
    # Want Pos = 2048.
    # Pos = Raw - Offset  => 2048 = Raw - Offset => Offset = Raw - 2048
    target_offset = raw - 2048
    
    # Handle wrapping ? Offset is signed 16-bit.
    print(f"Calculated Target Offset: {raw} - 2048 = {target_offset}")
    
    # 5. Write Offset
    print(f"Writing Offset {target_offset}...")
    robot.bus.write("Homing_Offset", motor_name, target_offset)
    time.sleep(0.5)
    
    # 6. Read Result
    final_pos = robot.bus.read("Present_Position", motor_name, normalize=False)
    final_offset = robot.bus.read("Homing_Offset", motor_name, normalize=False)
    print(f"FINAL: Pos={final_pos} (Target 2048), Offset={final_offset}")
    
    robot.bus.disconnect()

if __name__ == "__main__":
    reproduce_homing()
