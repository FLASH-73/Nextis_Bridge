from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor
from glob import glob
import time
import sys

def main():
    ports = glob('/dev/tty.usbmodem*') + glob('/dev/ttyACM*') + glob('/dev/ttyUSB*')
    
    for port in ports:
        try:
            try:
                found_map = FeetechMotorsBus.scan_port(port, protocol_version=1)
            except:
                found_map = FeetechMotorsBus.scan_port(port)

            found_ids = []
            for ids in found_map.values():
                found_ids.extend(ids)

            if not found_ids: continue
            
            motors = {f"m{id}": Motor(id, "sts3215", "range_0_100") for id in found_ids}
            bus = FeetechMotorsBus(port=port, motors=motors, protocol_version=1)
            bus.connect()
            
            print(f"Checking {len(motors)} motors on {port}...")
            
            # Target m9 if present, or all
            targets = ["m9"] if "m9" in motors else list(motors.keys())
            
            for m in targets:
                print(f"--- Motor {m} ---")
                # Unlock
                bus.write("Lock", m, 0)
                bus.write("Torque_Enable", m, 0)
                
                # Check Resolution
                res = bus.read("Angular_Resolution", m, normalize=False)
                print(f"  Angular_Resolution: {res}")
                
                # Test Offset Write
                test_val = 2000
                print(f"  Writing Offset: {test_val}")
                bus.write("Homing_Offset", m, test_val)
                time.sleep(0.5)
                read_val = bus.read("Homing_Offset", m, normalize=False)
                print(f"  Read Offset: {read_val}")
                
                if abs(read_val - test_val) > 10:
                    print(f"  SCALING DETECTED! Ratio: {read_val/test_val if test_val!=0 else 0}")
                else:
                    print("  No scaling.")
                    
                # Test Negative
                test_neg = -2000
                print(f"  Writing Offset: {test_neg}")
                bus.write("Homing_Offset", m, test_neg)
                time.sleep(0.5)
                read_neg = bus.read("Homing_Offset", m, normalize=False)
                print(f"  Read Offset: {read_neg}")

            bus.disconnect()
            
        except Exception as e:
            print(f"Error on {port}: {e}")

if __name__ == "__main__":
    main()
