
import time
import logging
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_arm(port, name="Arm"):
    print(f"\n=== FIXING {name} on {port} ===")
    
    # 1. Connect
    motors = {
        "base": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "link1": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "link2": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "link3": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "link4": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "link5": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    }
    
    try:
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()
        print("Connected.")
        
        # 2. Reset Homing Offsets to 0 AND Limits to Defaults (0-4095)
        print("Resetting Homing Offsets, Min/Max Limits...")
        bus.disable_torque(None)
        
        # Explicitly UNLOCK to ensure writes succeed
        for m in motors:
            bus.write("Lock", m, 0)
            
        for m in motors:
            bus.write("Homing_Offset", m, 0)
            # RESET LIMITS to ensure 'mid' calculation is correct (Center=2048)
            bus.write("Min_Position_Limit", m, 0)
            bus.write("Max_Position_Limit", m, 4095)
            
        time.sleep(0.5)
        
        # 3. Read Raw (True Raw)
        raw = bus.sync_read("Present_Position", normalize=False)
        print(f"True Raw Positions: {raw}")
        
        # 4. Interactive Calibration
        inp = input(f"--> Please move {name} to the CENTER/STRAIGHT pose (Physically). Press ENTER when ready.")
        
        # 5. Calculate New Offsets (Target 2048)
        print("Calibrating to 2048 (Zero Degrees)...")
        
        # MANUAL IMPLEMENTATION of set_half_turn_homings to ensure robust writing
        # (Standard method might be failing on USB2?)
        current_pos = bus.sync_read("Present_Position", normalize=False)
        new_offsets = {}
        target = 2048
        
        for name, m_obj in motors.items():
            if name not in current_pos:
                print(f"Skipping {name} (No reading)")
                continue
                
            val = current_pos[name]
            # Logic: Present = Raw - Offset ??
            # Based on logs: Raw=2257, Offset=209 => Result=2048 (Result = Raw - Offset)
            # 2048 = Raw - Offset => Offset = Raw - 2048
            # Wait, verify USB3 again.
            # Raw=2257. Target=2048. New_Offset=209.
            # 2257 - 209 = 2048. Matches.
            # So Offset = Raw - Target.
            
            # Verify USB2 log: Raw=47. Target=2048. New_Offset=-2001.
            # Offset = 47 - 2048 = -2001.
            # Result = 47 - (-2001) = 2048. Matches.
            
            offset = val - target
            new_offsets[name] = offset
            
            # FORCE WRITE
            # 1. Unlock
            bus.write("Lock", name, 0)
            # 2. Write Offset
            bus.write("Homing_Offset", name, int(offset))
            # 3. Read back to verify
            written = bus.read("Homing_Offset", name)
            if written != int(offset):
                # Try handling signed 16-bit wrapping if needed?
                # Feetech might return unsigned?
                # But let's just print warning
                print(f"WARNING: {name} Offset mismatch! Wrote {offset}, Read {written}")
            else:
                print(f"Set {name}: Raw={val} -> Offset={offset} (Verified)")
                
            # 4. Relock? (Maybe keep unlocked for now)
            
            
        print(f"New Offsets Applied (Manual Mode).")
        
        # 6. Verify
        time.sleep(0.5)
        current = bus.sync_read("Present_Position", normalize=False)
        print(f"New Calibrated Positions (Should be ~2048): {current}")
        
        bus.disconnect()
        print(f"=== {name} FIXED ===")
        
    except Exception as e:
        print(f"Failed to fix {name}: {e}")

if __name__ == "__main__":
    import glob
    # Use standard port patterns
    ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    
    if not ports:
        print("No ports found!")
        exit(1)
        
    print("\n========================================")
    print("   FIX ARM CALIBRATION (Interactive)    ")
    print("========================================")
    print("Which arm do you want to fix?")
    print("1. Left Follower")
    print("2. Right Follower")
    print("3. Left Leader")
    print("4. Right Leader")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    arm_map = {
        "1": "Left Follower",
        "2": "Right Follower",
        "3": "Left Leader",
        "4": "Right Leader"
    }
    
    target_arm = arm_map.get(choice)
    if not target_arm:
        print("Invalid choice. Exiting.")
        exit(1)
        
    print(f"\n[ Selected: {target_arm} ]")
    print("Please identify the USB Port for this arm:")
    
    for i, p in enumerate(ports):
        print(f"  {i}: {p}")
        
    p_idx = input("\nEnter Port Index: ").strip()
    
    try:
        port = ports[int(p_idx)]
    except (ValueError, IndexError):
        print("Invalid port index. Exiting.")
        exit(1)
        
    print(f"\nReady to fix {target_arm} on {port}.")
    confirm = input("Are you sure? (y/n): ").lower()
    
    if confirm == 'y':
        fix_arm(port, name=target_arm)
    else:
        print("Operation cancelled.")
