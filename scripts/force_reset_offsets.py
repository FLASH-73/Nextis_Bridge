from lerobot.motors.feetech.feetech import FeetechMotorsBus, TorqueMode
from lerobot.motors.motors_bus import Motor
from glob import glob
import time
import sys

def find_port():
    ports = glob('/dev/tty.usbmodem*') + glob('/dev/ttyACM*') + glob('/dev/ttyUSB*')
    if not ports:
        print("No ports found!")
        return None
    return ports[0]

def main():
    ports = glob('/dev/tty.usbmodem*') + glob('/dev/ttyACM*') + glob('/dev/ttyUSB*')
    if not ports:
        print("No ports found!")
        return

    print(f"Found ports: {ports}")
    
    for port in ports:
        print(f"\n=== Processing Port: {port} ===")
        try:
            # 1. Scan for motors
            params = {"protocol_version": 1}
            try:
                found_map = FeetechMotorsBus.scan_port(port, **params)
            except Exception as e:
                print(f"  Scan P1 failed: {e}")
                # Try P0
                try:
                    found_map = FeetechMotorsBus.scan_port(port)
                except Exception as e2:
                    print(f"  Scan P0 failed: {e2}")
                    continue
                
            found_ids = []
            for baud, ids in found_map.items():
                if ids:
                    found_ids.extend(ids)
            
            if not found_ids:
                print("  No motors found on this port.")
                continue

            print(f"  Found Motors: {found_ids}")

            # 2. Connect
            motors_dict = {f"motor_{id}": Motor(id=id, model="sts3215", norm_mode="range_0_100") for id in found_ids}
            bus = FeetechMotorsBus(port=port, motors=motors_dict, protocol_version=1)
            bus.connect()
            
            # 3. Aggressive Reset Loop
            for name in motors_dict:
                mid = motors_dict[name].id
                print(f"  [Motor {mid}]")
                try:
                    # A. Unlock
                    bus.write("Torque_Enable", name, 0)
                    bus.write("Lock", name, 0)
                    time.sleep(0.02)
                    
                    # B. Clear Offset
                    bus.write("Homing_Offset", name, 0)
                    
                    # C. Reset Limits
                    bus.write("Min_Position_Limit", name, 0)
                    bus.write("Max_Position_Limit", name, 0)
                    
                    # D. Read
                    pos = bus.read("Present_Position", name, normalize=False)
                    print(f"    Reset Done. Raw Pos: {pos}")
                except Exception as e:
                    print(f"    Failed: {e}")

            bus.disconnect()
            
        except Exception as e:
            print(f"  Port Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- GLOBAL RESET COMPLETE ---")

if __name__ == "__main__":
    main()
