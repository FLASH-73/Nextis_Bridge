import sys
import scservo_sdk as scs

PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3']
BAUDRATE = 1000000

def check_port(port):
    print(f"Scanning {port}...")
    ph = scs.PortHandler(port)
    pkh = scs.PacketHandler(0) # Protocol 0
    
    if hasattr(ph, 'setPacketTimeout'):
        # Apply patch if needed, or just rely on default
        pass
        
    if ph.openPort():
        pass
    else:
        print(f"  [FAIL] Failed to open port")
        return
        
    if not ph.setBaudRate(BAUDRATE):
         print(f"  [FAIL] Failed to set baudrate")
         ph.closePort()
         return
    
    # Ping ID 1 (Base)
    model, res, err = pkh.ping(ph, 1)
    has_id1 = (res == scs.COMM_SUCCESS)
    
    # Ping ID 9 (Follower Gripper)
    model, res, err = pkh.ping(ph, 9)
    has_id9 = (res == scs.COMM_SUCCESS)
    
    print(f"  ID 1 Found: {has_id1}")
    print(f"  ID 9 Found: {has_id9}")
    
    if has_id1 and has_id9:
        print(f"  -> TYPE: FOLLOWER (Found ID 9)")
    elif has_id1 and not has_id9:
        print(f"  -> TYPE: LEADER (ID 9 missing)")
    elif not has_id1:
        print(f"  -> TYPE: NONE/UNKNOWN (ID 1 missing)")
        
    ph.closePort()

if __name__ == '__main__':
    for p in PORTS:
        check_port(p)
