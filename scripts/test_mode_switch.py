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
            
            for m in motors:
                print(f"[{m}]")
                p_raw = bus.read("Present_Position", m, normalize=False)
                print(f"  Start Pos: {p_raw}")
                
                # UNLOCK
                bus.write("Lock", m, 0)
                bus.write("Torque_Enable", m, 0)
                
                # MODE SWITCH SEQUENCE
                print("  -> Switching to VELOCITY Mode (Wheel)...")
                bus.write("Operating_Mode", m, 1) # Velocity
                time.sleep(0.1)
                
                print("  -> Switching back to POSITION Mode...")
                bus.write("Operating_Mode", m, 0) # Position
                time.sleep(0.1)
                
                # Force Single-Turn Limits (Just in case)
                bus.write("Min_Position_Limit", m, 0) 
                bus.write("Max_Position_Limit", m, 4095)
                
                # Clear Offset
                bus.write("Homing_Offset", m, 0)
                time.sleep(0.5)

                p_new = bus.read("Present_Position", m, normalize=False)
                print(f"  End Pos  : {p_new}")
                
                if abs(p_new) > 4100:
                    print("  FAIL: Still huge.")
                else:
                    print("  SUCCESS: Reset to range.")
                
            bus.disconnect()
            
        except Exception as e:
            print(f"Error on {port}: {e}")

if __name__ == "__main__":
    main()
