#!/usr/bin/env python3
import sys
import os
import time
import argparse
import signal
import sys

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.core.leader_assist import LeaderAssistService
from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode, Motor, MotorNormMode

def main():
    parser = argparse.ArgumentParser(description="Calibrate Gravity Compensation Model")
    parser.add_argument("--port", type=str, required=True, help="Path to leader arm port (e.g. /dev/ttyUSB0)")
    args = parser.parse_args()

    print(f"Connecting to Leader Arm at {args.port}...")
    
    # Simple Setup for Umbra Leader (7 DOF STS3215)
    # IDs: 1-7. 
    motors = {}
    ids = [1, 2, 3, 4, 5, 6, 7] # Base...Gripper
    names = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
    
    for i, mid in enumerate(ids):
        motors[names[i]] = Motor(mid, "sts3215", MotorNormMode.DEGREES) # Use Raw/Degrees
        
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect()
    
    service = LeaderAssistService()
    service.start_calibration()
    
    print("\n--- GRAVITY CALIBRATION ---")
    print("Instructions:")
    print("1. Move the arm to a STATIC pose.")
    print("2. Press ENTER to record sample.")
    print("3. Repeat for ~10-15 varied poses covering the workspace.")
    print("4. Type 'done' to finish and calculate weights.")
    print("---------------------------\n")

    # Ensure Torque OFF initially so user can move it?
    # Actually, for G-Comp calibration, we usually want to read the CURRENT required to HOLD it?
    # OR: we let user move it (Torque OFF), then we turn Torque ON (Position Mode) to hold it, measure current?
    # NO. Feetech "Present_Load" (Current) is noisy.
    
    # Better Method for Feetech:
    # 1. User moves arm (Torque OFF).
    # 2. User presses Enter.
    # 3. Code locks position (Torque ON, Position Mode).
    # 4. Code reads "Present_Load" (Current/PWM) required to hold it.
    # 5. Code turns Torque OFF again.
    
    bus.disable_torque()
    
    try:
        while True:
            cmd = input("Move arm, then press ENTER (or type 'done'): ").strip().lower()
            if cmd == 'done':
                break
                
            print("Holding position and measuring load...", end="", flush=True)
            
            # 1. Read Position
            pos_dict = bus.read("Present_Position")
            
            # 2. Switch to Position Mode & Hold
            # bus.set_operating_mode(OperatingMode.POSITION) # Already default?
            # Set Goal to Current
            bus.write("Goal_Position", pos_dict)
            bus.enable_torque()
            
            # Wait for stabilize
            time.sleep(1.0)
            
            # 3. Measure Load (Average)
            loads = []
            for _ in range(10):
                l = bus.read("Present_Load")
                loads.append(l)
                time.sleep(0.05)
            
            # Average
            avg_load = {}
            for name in names:
                vals = [sample[name] for sample in loads]
                avg = sum(vals) / len(vals)
                avg_load[name] = avg
                
            print(f" Done. Load: {avg_load}")
            
            # Helper: Extract Vectors
            q_vec = []
            tau_vec = []
            
            # Conversion
            for name in names:
                raw_pos = pos_dict[name]
                raw_load = avg_load[name]
                
                # Deg
                deg = (raw_pos - 2048.0) * (360.0/4096.0)
                q_vec.append(deg)
                
                # Load (Signed int16 usually, check Feetech)
                # Feetech Load: bit 10 is direction? Or 2's complement?
                # Usually STS: bit 10 = direction (0=CCW?, 1=CW?)
                # Actually STS3215 Present_Load is Magnitude + Direction bit? Or just signed?
                # Let's assume Signed Int16 handling in 'bus.read'? 
                # lerobot Feetech reader usually handles 2's complement if configured?
                # Checking feetech.py... it uses `_read_words` -> struct.unpack('<h') for 2 bytes?
                # If "Present_Load" is configured as 2 bytes signed in table, it's fine.
                # STS manual: Addr 60, 2 bytes. 
                # If feetech.py returns signed int, we are good.
                
                tau_vec.append(raw_load) 
                
            service.record_sample(q_vec, tau_vec)
            
            # Release
            bus.disable_torque()
            print("Relased. Move to next pose.")

    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
    finally:
        bus.disable_torque()
        
    print("\nComputing Weights...")
    service.compute_weights()
    print(f"Calibration Saved to {service.calibration_path}")

if __name__ == "__main__":
    main()
