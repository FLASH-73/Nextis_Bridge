import sys
import time
from pathlib import Path
import json

# Add lerobot/src to sys.path
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path / "lerobot" / "src"))

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE
from lerobot.motors.motors_bus import Motor, MotorNormMode

def debug_motors():
    # Hardcoded for debugging
    port = "/dev/ttyUSB3"
    print(f"Connecting to {port}...")
    
    bus = FeetechMotorsBus(port=port, motors={})
    if not bus.port_handler.openPort():
        print("Failed to open port")
        return
        
    ids = bus.broadcast_ping()
    if not ids:
        print("No motors found")
        return
        
    print(f"Found motors: {ids}")
    motor_id = list(ids.keys())[0]
    
    # Force model to sts3215
    model_name = "sts3215"
    print(f"Forcing Model Name: {model_name}")
    
    dummy_motor = Motor(id=motor_id, model=model_name, norm_mode=MotorNormMode.RANGE_0_100)
    bus.motors = {"debug_motor": dummy_motor}
    bus._id_to_model_dict = {motor_id: model_name}
    bus._id_to_name_dict = {motor_id: "debug_motor"}
    
    # Reset Calibration
    print("\n--- Resetting Calibration ---")
    print("Disabling Torque...")
    bus.write("Lock", "debug_motor", 0)
    bus.write("Torque_Enable", "debug_motor", 0)
    
    print("Resetting Homing_Offset to 0...")
    bus.write("Homing_Offset", "debug_motor", 0)
    
    print("Resetting Limits to 0 (Min) and 4095 (Max)...")
    bus.write("Min_Position_Limit", "debug_motor", 0)
    bus.write("Max_Position_Limit", "debug_motor", 4095)
    
    # Test Homing Logic
    print("\n--- Testing Homing Logic ---")
    current_pos = bus.read("Present_Position", "debug_motor", normalize=False)
    print(f"Current Pos: {current_pos}")
    
    target_pos = 2048
    delta = current_pos - target_pos
    new_offset = delta # Since old offset is 0
    print(f"Calculated Offset: {new_offset}")
    
    print(f"Writing Homing_Offset: {new_offset}")
    bus.write("Homing_Offset", "debug_motor", new_offset)
    
    time.sleep(0.5)
    new_pos = bus.read("Present_Position", "debug_motor", normalize=False)
    print(f"New Pos (should be ~2048): {new_pos}")
    
    # Dump Registers
    print("\n--- Register Dump ---")
    registers_to_check = [
        "ID", "Baud_Rate", "Min_Position_Limit", "Max_Position_Limit", 
        "Homing_Offset", "Operating_Mode", "Torque_Enable", "Lock",
        "Present_Position", "Present_Voltage", "Present_Temperature"
    ]
    
    for reg in registers_to_check:
        try:
            val = bus.read(reg, "debug_motor", normalize=False)
            print(f"{reg}: {val}")
        except Exception as e:
            print(f"{reg}: Error ({e})")
            
    print("\n--- Live Monitoring (Press Ctrl+C to stop) ---")
    try:
        while True:
            pos = bus.read("Present_Position", "debug_motor", normalize=False)
            print(f"Pos: {pos}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    debug_motors()
