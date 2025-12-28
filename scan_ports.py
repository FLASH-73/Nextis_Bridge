import sys
import glob
import scservo_sdk as scs
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor

# Define Protocol (STS3215 is likely Protocol 1 if it's SCS series, but check defaults)
# The default in feetech.py is 0 (which usually maps to 1.0 in Dynamixel SDK terms? or 0.0?)
# Feetech usually uses Protocol 0 (SCS) or 1 (SMS).
# FeetechMotorsBus sets default protocol_version=0
PROTOCOL_VERSION = 0 
BAUDRATE = 1000000

print("Scanning ports with Active Ping (IDs 1-10)...")
ports = sorted(glob.glob("/dev/ttyUSB*"))

if not ports:
    print("No USB ports found!")
    sys.exit(1)

for p in ports:
    print(f"Scanning {p}...")
    
    # Initialize basic handlers without the Bus class to avoid validation logic
    try:
        portHandler = scs.PortHandler(p)
        packetHandler = scs.PacketHandler(PROTOCOL_VERSION)
        
        if portHandler.openPort():
            # print("  Port Opened")
            pass
        else:
            print("  Failed to open port")
            continue
            
        if portHandler.setBaudRate(BAUDRATE):
            # print("  Baudrate Set")
            pass
        else:
            print("  Failed to set baudrate")
            continue
            
        found_ids = []
        for dxl_id in range(1, 11): # Scan 1 to 10
            model_number, result, error = packetHandler.ping(portHandler, dxl_id)
            if result == scs.COMM_SUCCESS:
                # print(f"  [ID:{dxl_id}] Ping Success. Model: {model_number}")
                found_ids.append(dxl_id)
            # else:
            #     print(f"  [ID:{dxl_id}] Failed: {packetHandler.getTxRxResult(result)}")
                
        if found_ids:
            print(f"[{p}] -> FOUND IDS: {found_ids}")
        else:
            print(f"[{p}] -> No motors found.")
            
        portHandler.closePort()
        
    except Exception as e:
        print(f"[{p}] -> Error: {e}")


