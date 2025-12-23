from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor
from glob import glob
import sys
import time

def find_port():
    ports = glob('/dev/tty.usbmodem*') + glob('/dev/ttyACM*') + glob('/dev/ttyUSB*')
    if not ports:
        print("No ports found!")
        return None
    print(f"Found ports: {ports}")
    return ports[0]

def main():
    port = find_port()
    if not port:
        return

    
    # Assume Umbra IDs - typically 1-6 for Leader, etc.

    # We'll just scan or guess. Let's try to 'scan' by pinging range 1-20.
    print(f"Connecting to {port}...")
    
    # We need to construct a bus to scan.
    # We can use the class method scan_port but it creates a bus internally.
    # Let's just manually instantiate with empty motors and ping.
    
    # Actually, let's look for known IDs if possible.
    # But simpler: use FeetechMotorsBus.scan_port
    try:
        found_map = FeetechMotorsBus.scan_port(port)
        print("Scan result:", found_map)
        
        found_ids = []
        for baud, ids in found_map.items():
            if ids:
                found_ids.extend(ids)
                
        if not found_ids:
            print("No motors found.")
            return

        # Create bus with found motors
        motors_dict = {f"motor_{id}": Motor(id=id, model="sts3215", norm_mode="range_0_100") for id in found_ids}
        bus = FeetechMotorsBus(port=port, motors=motors_dict, protocol_version=1)
        bus.connect()
        
        print("\n--- Reading Registers ---")
        for name in motors_dict:
            # Read Raw
            pos = bus.read("Present_Position", name, normalize=False)
            offset = bus.read("Homing_Offset", name, normalize=False)
            lock = bus.read("Lock", name, normalize=False)
            min_lim = bus.read("Min_Position_Limit", name, normalize=False)
            max_lim = bus.read("Max_Position_Limit", name, normalize=False)
            
            print(f"{name}: Pos={pos}, Offset={offset}, Lock={lock}, Limit={min_lim}-{max_lim}")
            
        print("\n--- Sync Read Check (Raw) ---")
        sync_pos = bus.sync_read("Present_Position", normalize=False)
        print("Sync Pos:", sync_pos)
        
        bus.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
