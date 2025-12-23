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
            # Try default scan (usually P0) which works for detection even if P1 is used later?
            # Or just catch the error and assume ids.
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
                # Read Current
                p_raw = bus.read("Present_Position", m, normalize=False)
                lim_min = bus.read("Min_Position_Limit", m, normalize=False)
                lim_max = bus.read("Max_Position_Limit", m, normalize=False)
                print(f"  Before: Pos={p_raw}, Limits=({lim_min}, {lim_max})")
                
                # UNLOCK
                bus.write("Lock", m, 0)
                bus.write("Torque_Enable", m, 0)
                
                # SET SINGLE TURN LIMITS (0 - 4095)
                # This should reset the multi-turn counter logic in the motor firmware
                print("  -> Setting Limits to 0 - 4095...")
                bus.write("Min_Position_Limit", m, 0)
                bus.write("Max_Position_Limit", m, 4095)
                time.sleep(0.1)
                
                # Read After
                p_new = bus.read("Present_Position", m, normalize=False)
                print(f"  After : Pos={p_new}")
                
            bus.disconnect()
            
        except Exception as e:
            print(f"Error on {port}: {e}")

if __name__ == "__main__":
    main()
