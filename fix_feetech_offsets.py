import sys
import time
from pathlib import Path

# Add paths
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path / "lerobot" / "src"))

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

def fix_offsets():
    # Ports from settings.yaml
    ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"]
    
    for port in ports:
        print(f"\nScanning {port}...")
        try:
            bus = FeetechMotorsBus(port=port, motors={})
            if not bus.port_handler.openPort():
                print(f"Failed to open {port}")
                continue
                
            ids = bus.broadcast_ping()
            if not ids:
                print(f"No motors on {port}")
                continue
                
            print(f"Found motors: {ids}")
            
            for motor_id, model_num in ids.items():
                print(f"Processing Motor {motor_id}...")
                
                # Setup dummy motor for communication
                bus.motors = {
                    "temp": Motor(id=motor_id, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
                }
                bus._id_to_model_dict = {motor_id: "sts3215"}
                bus._id_to_name_dict = {motor_id: "temp"}
                
                # 1. Disable Torque
                bus.write("Lock", "temp", 0)
                bus.write("Torque_Enable", "temp", 0)
                
                # 2. Read Current Position
                pos = bus.read("Present_Position", "temp", normalize=False)
                print(f"  Current Position: {pos}")
                
                # 3. Reset Homing Offset to 0
                bus.write("Homing_Offset", "temp", 0)
                time.sleep(0.5)
                
                # 4. Read Raw Position (with Offset 0)
                raw_pos = bus.read("Present_Position", "temp", normalize=False)
                print(f"  Raw Position (Offset 0): {raw_pos}")
                
                target = 2048
                # Calculate delta assuming wrapping
                # We want: (raw_pos - offset) & 0xFFFF == target
                # offset = (raw_pos - target) & 0xFFFF
                
                offset_unsigned = (raw_pos - target) & 0xFFFF
                offset_signed = offset_unsigned
                if offset_signed >= 32768:
                    offset_signed -= 65536
                    
                print(f"  Calculated Offset to reach 2048: {offset_signed} (Hex: {offset_unsigned:04X})")
                
                bus.write("Homing_Offset", "temp", offset_signed)
                print("  Offset Written.")
                time.sleep(0.5)
                
                # Verify Offset
                read_offset = bus.read("Homing_Offset", "temp", normalize=False)
                print(f"  Read Back Offset: {read_offset}")
                
                time.sleep(0.5)
                new_pos = bus.read("Present_Position", "temp", normalize=False)
                print(f"  New Position: {new_pos} (Target: 2048)")
                
                # Also reset limits
                bus.write("Min_Position_Limit", "temp", 0)
                bus.write("Max_Position_Limit", "temp", 4095)
                print("  Limits Reset.")

        except Exception as e:
            print(f"Error on {port}: {e}")

if __name__ == "__main__":
    fix_offsets()
