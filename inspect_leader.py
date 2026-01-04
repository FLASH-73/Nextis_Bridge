
import time
import logging
import time
import logging
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_arm(port, name="Arm"):
    print(f"--- Inspecting {name} on {port} ---")
    try:
        # Connect using RAW mode (Range 0-100 to avoid Degree conversion hiding raw values)
        # Actually, let's use the Bus DIRECTLY to get raw register values.
        
        # Define motors (Standard Umbra Leader Map)
        motors = {
            "base": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "link1": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "link2": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "link3": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "link4": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "link5": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
        }
        
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()
        
        print("Connected.")
        
        # Read Offsets
        offsets = bus.sync_read("Homing_Offset", normalize=False)
        print("Homing Offsets (Internal):")
        for m, v in offsets.items():
            print(f"  {m}: {v}")
            
        print("\nReading RAW Positions (Live 5s)...")
        for i in range(50):
            # Read Raw (normalize=False)
            raw = bus.sync_read("Present_Position", normalize=False)
            
            # Format output
            line = f"T={i*0.1:.1f} | "
            for m in ["base", "link1", "link2", "link3", "link4", "link5"]:
                r = raw.get(m, "ERR")
                line += f"{m}={r} "
            print(line)
            time.sleep(0.1)
            
        bus.disconnect()
        
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

if __name__ == "__main__":
    # Hardcoded ports from user config or typical defaults?
    # I'll check main.py or args. 
    # Providing a generic guess first.
    # User has bi_umbra_leader.
    
    # Try to find ports
    import glob
    ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    print(f"Found Ports: {ports}")
    
    # Just run on all found ports? Safer to ask user but I'll try to probe.
    # Usually Leader is on separate port. 
    # I will rely on the ports defined in the active config if possible.
    
    # Let's inspect /dev/ttyACM0 and ACM1 if available.
    if len(ports) > 0:
        inspect_arm(ports[0], "Port_0")
    if len(ports) > 1:
        inspect_arm(ports[1], "Port_1")
