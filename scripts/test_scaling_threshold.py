from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor
from glob import glob
import time

def main():
    ports = glob('/dev/tty.usbmodem*') + glob('/dev/ttyACM*') + glob('/dev/ttyUSB*')
    for port in ports:
        try:
            try:
                found_map = FeetechMotorsBus.scan_port(port, protocol_version=1)
            except:
                found_map = FeetechMotorsBus.scan_port(port)
            found_ids = []
            for ids in found_map.values(): found_ids.extend(ids)
            if not found_ids: continue
            
            motors = {f"m{id}": Motor(id, "sts3215", "range_0_100") for id in found_ids}
            bus = FeetechMotorsBus(port=port, motors=motors, protocol_version=1)
            bus.connect()
            
            print(f"Testing Scaling on {list(motors.keys())[0]}...")
            m = list(motors.keys())[0] # Test on first motor
            
            bus.write("Lock", m, 0)
            
            test_values = [2000, 4000, 4200, 8000, 10000, 20000, 30000]
            
            for val in test_values:
                bus.write("Homing_Offset", m, val)
                time.sleep(0.2)
                read = bus.read("Homing_Offset", m, normalize=False)
                print(f"Write {val} -> Read {read}. Ratio: {read/val if val!=0 else 0:.2f}")
                
            bus.disconnect()
            break
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
