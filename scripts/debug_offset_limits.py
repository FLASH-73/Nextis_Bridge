from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor
from glob import glob
import time
import sys

def main():
    ports = glob('/dev/tty.usbmodem*') + glob('/dev/ttyACM*') + glob('/dev/ttyUSB*')
    print(f"Found ports: {ports}")
    
    for port in ports:
        try:
            print(f"\nScanning {port}...")
            # Try default scan
            try:
                found_map = FeetechMotorsBus.scan_port(port, protocol_version=1)
            except:
                found_map = FeetechMotorsBus.scan_port(port)

            found_ids = []
            for ids in found_map.values():
                found_ids.extend(ids)

            if not found_ids:
                print("No motors.")
                continue

            print(f"Found: {found_ids}")
            
            motors = {f"m{id}": Motor(id, "sts3215", "range_0_100") for id in found_ids}
            bus = FeetechMotorsBus(port=port, motors=motors, protocol_version=1)
            bus.connect()
            
            # Use just one motor to test
            test_motor = list(motors.keys())[0]
            print(f"Testing on {test_motor}...")
            
            # UNLOCK
            bus.write("Lock", test_motor, 0)
            bus.write("Torque_Enable", test_motor, 0)
            
            # TEST 1: Limits = 0,0 (Multi-Turn)
            print("\n--- TEST 1: Limits 0,0 (Multi-Turn) ---")
            bus.write("Min_Position_Limit", test_motor, 0)
            bus.write("Max_Position_Limit", test_motor, 0)
            bus.write("Homing_Offset", test_motor, 0)
            time.sleep(0.5)
            base_pos = bus.read("Present_Position", test_motor, normalize=False)
            print(f"  Base Pos (Offset=0): {base_pos}")
            
            # Apply Offset
            print("  Applying Offset 2048...")
            bus.write("Homing_Offset", test_motor, 2048)
            time.sleep(0.5)
            new_pos = bus.read("Present_Position", test_motor, normalize=False)
            print(f"  New Pos (Offset=2048): {new_pos}")
            diff = new_pos - base_pos
            print(f"  Delta: {diff}")
            
            if abs(diff - 2048) < 100:
                print("  => Offset WORKS in Multi-Turn.")
            else:
                print("  => Offset IGNORED in Multi-Turn.")

            # TEST 2: Limits = Standard (Single Turn?) 0, 4095
            # Or Wide Range: -32000, 32000
            print("\n--- TEST 2: Limits -32000, 32000 ---")
            bus.write("Min_Position_Limit", test_motor, -32000)
            bus.write("Max_Position_Limit", test_motor, 32000)
            bus.write("Homing_Offset", test_motor, 0)
            time.sleep(0.5)
            base_pos_2 = bus.read("Present_Position", test_motor, normalize=False)
            print(f"  Base Pos (Offset=0): {base_pos_2}")
            
            # Apply Offset
            print("  Applying Offset 2048...")
            bus.write("Homing_Offset", test_motor, 2048)
            time.sleep(0.5)
            new_pos_2 = bus.read("Present_Position", test_motor, normalize=False)
            print(f"  New Pos (Offset=2048): {new_pos_2}")
            diff_2 = new_pos_2 - base_pos_2
            print(f"  Delta: {diff_2}")
             
            if abs(diff_2 - 2048) < 100:
                print("  => Offset WORKS in Limits Mode.")
            else:
                print("  => Offset IGNORED in Limits Mode.")

            bus.disconnect()
            break # Just one port needed
            
        except Exception as e:
            print(f"Error on {port}: {e}")

if __name__ == "__main__":
    main()
