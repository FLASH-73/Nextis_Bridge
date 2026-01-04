from lerobot.motors.feetech import FeetechMotorsBus
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"]

print("Scanning ports for motors...")

for port in ports:
    print(f"\n--- Checking {port} ---")
    try:
        # Instantiate bus but DO NOT calling connect()
        bus = FeetechMotorsBus(port=port, motors={})
        
        # Manually open port
        if not bus.port_handler.openPort():
             print(f"  Failed to open port {port}")
             continue
             
        # Set Baudrate (typically 1M for these robots)
        bus.port_handler.setBaudRate(1000000)
        
        print(f"  Port Opened. Pinging IDs 1-10...")
        found = []
        import scservo_sdk as scs
        
        for mid in range(1, 11):
            # Use packet handler ping
            # ping(port, id) returns model number or None/Exception wrapper?
            # scservo_sdk ping returns model number if success?
            # actually scs uses different methods.
            # bus.ping(mid) wraps it?
            
            # Use bus internal ping wrapper if possible, or low level
            # bus.ping(mid) isn't standard in MotorsBus but available in Feetech?
            # FeetechMotorsBus has _find_single_motor_p1 which calls self.ping(id)
            # Wait, self.ping doesn't exist in the class explicitly, likely inherited from MotorsBus? No.
            # Let's use low level scs logic found in _find_single_motor_p1
            
            model = bus.packet_handler.ping(bus.port_handler, mid)
            # ping returns (model_number, comm_result, error)
            # actually typical dynamixel sdk return is 3 values.
            # scservo sdk might be different. 
            # bus.packet_handler is scs.PacketHandler
            
            # Let's inspect library usage in lines 215 of feetech.py:
            # found_model = self.ping(id_) -> this implies self.ping exists?
            # I don't see self.ping definition in feetech.py.
            # Maybe it's dynamically added or from MotorsBus?
            # MotorsBus usually doesn't have ping.
            
            # Trace: _find_single_motor_p1 call `found_model = self.ping(id_)`.
            # Ah, maybe I missed it in view_file. Or likely it's `bus.packet_handler.ping`.
            # Wait! scs.PacketHandler.ping returns (model, result, error).
            
            model, res, err = bus.packet_handler.ping(bus.port_handler, mid)
            if res == scs.COMM_SUCCESS and err == 0:
                print(f"    FOUND ID {mid} (Model {model})")
                found.append(mid)
                
        if not found:
             print("  No motors found.")
        
        bus.port_handler.closePort()

    except Exception as e:
        print(f"  Failed: {e}")

