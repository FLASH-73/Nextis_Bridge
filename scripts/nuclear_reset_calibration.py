from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode, TorqueMode
from lerobot.motors.motors_bus import Motor
from glob import glob
import time
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
            # NAMES Mapping
            names = list(motors.keys())
            
            # 1. NUCLEAR RESET (Global Sync Write)
            print("\n--- NUCLEAR RESET ---")
            
            # Lock = 0
            print("Unlocking...")
            bus.sync_write("Lock", 0)
            bus.sync_write("Torque_Enable", 0)
            bus.sync_write("Operating_Mode", 0) # Position
            bus.sync_write("Min_Position_Limit", 0)
            bus.sync_write("Max_Position_Limit", 4095)
            bus.sync_write("Homing_Offset", 0)
            
            print("Waiting 1.5s for reset...")
            time.sleep(1.5)
            
            # 2. READ RAW
            print("Reading Raw...")
            raw_pos = bus.sync_read("Present_Position", normalize=False)
            
            # 3. CALCULATE OFFSETS (Standard Logic: Offset = Raw - 2048)
            # Check if Doubling is needed?
            # Based on m9 data: Needed -20476, Got -10226.
            # Implies we need to Write 2 * Calculated?
            
            offsets_to_write = {}
            for m, raw in raw_pos.items():
                calc_offset = raw - 2048
                
                # EXPERIMENTAL: DOUBLE THE OFFSET?
                # bus.write applies encoding.
                # If I just write calc_offset, it fails (halved).
                # Let's try writing calc_offset normally first, but debug the value.
                
                # Actually, let's stick to Standard Logic but ensure it's written correctly.
                # Maybe sync_write handles it better?
                
                # Wrap calc_offset to 16-bit signed
                wrapped_offset = calc_offset % 65536
                if wrapped_offset >= 32768:
                    wrapped_offset -= 65536
                    
                offsets_to_write[m] = wrapped_offset
                print(f"  {m}: Raw={raw} -> Target Offset={wrapped_offset}")

            # 4. WRITE OFFSETS (Sync Write)
            print("Writing Offsets via Sync Write...")
            try:
                bus.sync_write("Homing_Offset", offsets_to_write)
            except Exception as e:
                print(f"Sync Write Failed: {e}. Falling back to individual.")
                for m, off in offsets_to_write.items():
                    bus.write("Homing_Offset", m, off)
            
            print("Waiting 1.0s...")
            time.sleep(1.0)
            
            # 5. VERIFY
            print("Verifying...")
            final = bus.sync_read("Present_Position", normalize=False)
            final_offs = bus.sync_read("Homing_Offset", normalize=False)
            
            for m in names:
                print(f"  {m}: Pos={final.get(m)} (Offsets={final_offs.get(m)})")
                
            bus.disconnect()
            
        except Exception as e:
            print(f"Error on {port}: {e}")

if __name__ == "__main__":
    main()
