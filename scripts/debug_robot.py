import sys
import time
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_path))
sys.path.append(str(root_path / "lerobot" / "src"))

from app.core.config import load_config
from scservo_sdk import PacketHandler, PortHandler, COMM_SUCCESS

# Control table address
ADDR_SCS_PRESENT_POSITION = 56  # Present Position

# Protocol version
PROTOCOL_VERSION = 2.0

# Default baudrate
BAUDRATE = 1000000

def scan_port(port, baudrate=BAUDRATE):
    print(f"Scanning {port} at {baudrate} baud...")
    
    try:
        portHandler = PortHandler(port)
        packetHandler = PacketHandler(PROTOCOL_VERSION) # Protocol 2.0

        if portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            return

        if portHandler.setBaudRate(baudrate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            return

        # Scan for IDs 1-10
        found_ids = []
        for dxl_id in range(1, 11):
            model_number, result, error = packetHandler.ping(portHandler, dxl_id)
            if result == COMM_SUCCESS: # Use COMM_SUCCESS for comparison
                print(f"[ID:{dxl_id:03d}] ping Succeeded. Model Number: {model_number}")
                found_ids.append(dxl_id)
                
                # Read Present Position
                # Address 56, Length 2
                pos, comm_res, error = packetHandler.read2ByteTxRx(portHandler, dxl_id, ADDR_SCS_PRESENT_POSITION)
                if comm_res == COMM_SUCCESS and error == 0:
                    print(f"  [ID:{dxl_id:03d}] Present Position: {pos}")
                else:
                    print(f"  [ID:{dxl_id:03d}] Failed to read position (Comm Result: {comm_res}, Error: {error})")
            else:
                # print(f"[ID:{dxl_id:03d}] ping Failed")
                pass
        
        print(f"Found IDs on {port}: {found_ids}")
        portHandler.closePort()
        
    except Exception as e:
        print(f"Error scanning {port}: {e}")

if __name__ == "__main__":
    config = load_config()
    robot_cfg = config.get("robot", {})
    
    left_port = robot_cfg.get("left_arm_port")
    right_port = robot_cfg.get("right_arm_port")
    
    if left_port:
        scan_port(left_port)
    
    if right_port and right_port != left_port:
        scan_port(right_port)
