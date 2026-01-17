import sys
import glob
from dynamixel_sdk import *

# Control Table Address
ADDR_MODEL_NUMBER       = 0
ADDR_TORQUE_ENABLE      = 64

# Protocol version
PROTOCOL_VERSION        = 2.0

# Baudrate
BAUDRATE                = 1000000             # Dynamixel default is usually 57600 or 1M. LeRobot uses 1M usually.

PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB4', '/dev/ttyUSB5']

def check_port(port):
    print(f"Checking {port}...")
    portHandler = PortHandler(port)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if portHandler.openPort():
        print(f"  Succeeded to open the port {port}")
    else:
        print(f"  Failed to open the port {port}")
        return

    if portHandler.setBaudRate(BAUDRATE):
        print(f"  Succeeded to change the baudrate {BAUDRATE}")
    else:
        print(f"  Failed to change the baudrate")
        portHandler.closePort()
        return

    # Try to ping ID 1
    dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, 1)
    if dxl_comm_result != COMM_SUCCESS:
        # print(f"  Ping failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
        # Try ID 0 just in case
        dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, 0)
    
    if dxl_comm_result == COMM_SUCCESS:
        print(f"  [SUCCESS] Ping ID 1 (or 0) response. Model Number: {dxl_model_number}")
        
        # Read Model Number explicitly from EEPROM to be sure
        model_num, res, err = packetHandler.read2ByteTxRx(portHandler, 1, ADDR_MODEL_NUMBER)
        if res == COMM_SUCCESS:
             print(f"  [INFO] Read Model Number from ID 1: {model_num}")
             if model_num in [1190, 1200]:
                 print("  -> TYPE: LEADER (XL330)")
             elif model_num in [1000, 1010, 1020, 1030, 1060, 1100, 1110]:
                 print("  -> TYPE: FOLLOWER (XM/XL/XC430)")
             else:
                 print(f"  -> TYPE: UNKNOWN ({model_num})")
    else:
        print(f"  [FAIL] No response from ID 1.")

    portHandler.closePort()

if __name__ == "__main__":
    for p in PORTS:
        check_port(p)
