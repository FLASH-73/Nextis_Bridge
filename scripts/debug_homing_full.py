from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode, TorqueMode
from lerobot.motors.motors_bus import Motor
from glob import glob
import time
import sys
import logging

# Configure logging
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
            
            print(f"Connected to {len(motors)} motors.")
            
            # 1. READ RAW STATE
            print("\n--- INITIAL STATE ---")
            raw = bus.sync_read("Present_Position", normalize=False)
            offsets = bus.sync_read("Homing_Offset", normalize=False)
            for m in motors:
                print(f"Motor {m}: Raw={raw.get(m)}, Offset={offsets.get(m)}")

            # 2. PERFORM HOMING (Logic from CalibrationService)
            print("\n--- EXECUTING HOMING SEQUENCE ---")
            
            # Global Torque Disable
            try:
                bus.disable_torque(None)
            except Exception as e:
                print(f"Global Disable Torque Error: {e}")

            for motor in motors:
                print(f"Configuring {motor}...")
                bus.write("Lock", motor, 0)
                bus.write("Torque_Enable", motor, 0)
                bus.write("Operating_Mode", motor, OperatingMode.POSITION.value) # 0
                bus.write("Min_Position_Limit", motor, 0)
                bus.write("Max_Position_Limit", motor, 4095)
                # Clear Offset to prepare for calculation
                bus.write("Homing_Offset", motor, 0)
            
            print("Waiting 1.0s for registers to settle...")
            time.sleep(1.0)
            
            print("Calculating and Writing Homing Offsets (CUSTOM SUBTRACTIVE LOGIC)...")
            # READ RAW
            positions = bus.sync_read("Present_Position", normalize=False)
            
            # CALCULATE OFFSET = 2048 - RAW
            # Because Present = Raw + Offset. We want Present = 2048.
            # So Offset = 2048 - Raw.
            for motor in motors:
                if motor not in positions: continue
                raw = positions[motor]
                
                # Desired Offset
                offset = 2048 - raw
                
                # Wrap to 16-bit signed
                # Python Handles int overflow automatically? No, we need to ensure it fits in 2 bytes for write.
                # However, FeetechMotorsBus._encode_sign handles 'twos'.
                # But 'twos' requires value to be in range [-32768, 32767].
                # If offset is outside, we must wrap it modulo 65536.
                
                # Logic:
                # 2048 - (-17407) = 19455. (Fits).
                # 2048 - (30000) = -27952. (Fits).
                # 2048 - (-40000) = 42048. (Overflow > 32767).
                # 42048 % 65536 = 42048.
                # If > 32767, subtract 65536 -> -23488.
                
                wrapped_offset = offset % 65536
                if wrapped_offset >= 32768:
                    wrapped_offset -= 65536
                    
                print(f"  Motor {motor}: Raw={raw}. Target=2048. Offset={offset}. Wrapped={wrapped_offset}")
                
                # Write
                bus.write("Homing_Offset", motor, wrapped_offset)

            
            print("Waiting 0.5s for write...")
            time.sleep(0.5)
            
            # 3. VERIFY
            print("\n--- VERIFICATION ---")
            final_pos = bus.sync_read("Present_Position", normalize=False)
            final_offsets = bus.sync_read("Homing_Offset", normalize=False)
            
            success_count = 0
            for m in motors:
                pos = final_pos.get(m)
                off = final_offsets.get(m)
                delta = abs(pos - 2048)
                status = "PASS" if delta < 100 else "FAIL"
                if status == "PASS": success_count += 1
                
                print(f"Motor {m}: Pos={pos} (expect 2048), Offset={off}. [{status}]")

            print(f"\nResult: {success_count}/{len(motors)} Homed Successfully.")

            bus.disconnect()
            
        except Exception as e:
            print(f"Error on {port}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
