#!/usr/bin/env python3
"""Test script to recover link2 (J4340P, CAN ID 0x03) from fault state.

Usage:
    sudo /home/roberto/miniconda3/bin/python3 test_link2_recovery.py

Steps:
1. Send disable command (0xFD)
2. Wait, then send enable command (0xFC)
3. Send a limp MIT command (kp=0, kd=0) to check if motor responds
"""

import can
import struct
import time

CAN_INTERFACE = "can0"
SLAVE_ID = 0x03      # link2
MASTER_ID = 0x13     # link2's master ID

def send_control_cmd(bus, slave_id, cmd_byte):
    """Send a control command (enable/disable/set_zero)."""
    data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, cmd_byte]
    msg = can.Message(arbitration_id=slave_id, data=data, is_extended_id=False)
    bus.send(msg)
    print(f"  Sent cmd 0x{cmd_byte:02X} to CAN ID 0x{slave_id:02X}")

def send_mit_limp(bus, slave_id, p_max=12.5, v_max=8.0, t_max=28.0):
    """Send a MIT command with kp=0, kd=0 (limp/probe)."""
    # Encode position=0, velocity=0, kp=0, kd=0, torque=0
    p_int = int((0.0 + p_max) / (2 * p_max) * 65535)  # 0 → midpoint
    v_int = int((0.0 + v_max) / (2 * v_max) * 4095)
    kp_int = 0  # kp=0
    kd_int = 0  # kd=0
    t_int = int((0.0 + t_max) / (2 * t_max) * 4095)

    data = [
        (p_int >> 8) & 0xFF,
        p_int & 0xFF,
        ((v_int >> 4) & 0xFF),
        (((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF)),
        kp_int & 0xFF,
        ((kd_int >> 4) & 0xFF),
        (((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF)),
        t_int & 0xFF,
    ]
    msg = can.Message(arbitration_id=slave_id, data=data, is_extended_id=False)
    bus.send(msg)
    print(f"  Sent MIT limp command (kp=0, kd=0) to CAN ID 0x{slave_id:02X}")

def read_responses(bus, timeout=0.5):
    """Read all available CAN responses."""
    responses = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = bus.recv(timeout=0.1)
        if msg:
            responses.append(msg)
            print(f"  Response: arb_id=0x{msg.arbitration_id:03X} data={msg.data.hex(' ')}")
    return responses

def main():
    print(f"=== Link2 Recovery Test ===")
    print(f"Motor: J4340P, CAN ID: 0x{SLAVE_ID:02X}, Master ID: 0x{MASTER_ID:02X}")
    print()

    bus = can.Bus(channel=CAN_INTERFACE, interface="socketcan", bitrate=1000000)

    # Flush any stale messages
    while bus.recv(timeout=0.05):
        pass

    # Step 1: Disable
    print("Step 1: Sending DISABLE (0xFD)...")
    send_control_cmd(bus, SLAVE_ID, 0xFD)
    time.sleep(0.2)
    responses = read_responses(bus, timeout=0.5)
    if not responses:
        print("  !! No response to disable command")
    print()

    # Step 2: Wait
    print("Step 2: Waiting 2 seconds...")
    time.sleep(2.0)
    # Flush
    while bus.recv(timeout=0.05):
        pass
    print()

    # Step 3: Enable
    print("Step 3: Sending ENABLE (0xFC)...")
    send_control_cmd(bus, SLAVE_ID, 0xFC)
    time.sleep(0.2)
    responses = read_responses(bus, timeout=0.5)
    if not responses:
        print("  !! No response to enable command")
    else:
        # Parse the response
        for r in responses:
            if r.arbitration_id == MASTER_ID:
                # MIT feedback: data[0]=id, data[1..2]=pos, data[3..4]=vel, data[5..6]=torque
                pos_raw = (r.data[1] << 8) | r.data[2]
                vel_raw = (r.data[3] << 4) | (r.data[4] >> 4)
                torque_raw = ((r.data[4] & 0xF) << 8) | r.data[5]
                # Decode
                pos = pos_raw / 65535.0 * 25.0 - 12.5
                vel = vel_raw / 4095.0 * 16.0 - 8.0
                torque = torque_raw / 4095.0 * 56.0 - 28.0
                print(f"  Decoded: pos={pos:.4f} rad, vel={vel:.4f} rad/s, torque={torque:.2f} Nm")
    print()

    # Step 4: Send limp MIT command
    print("Step 4: Sending MIT limp command (kp=0, kd=0)...")
    send_mit_limp(bus, SLAVE_ID)
    time.sleep(0.1)
    responses = read_responses(bus, timeout=0.5)
    if not responses:
        print("  !! No response to MIT command")
    else:
        for r in responses:
            if r.arbitration_id == MASTER_ID:
                pos_raw = (r.data[1] << 8) | r.data[2]
                pos = pos_raw / 65535.0 * 25.0 - 12.5
                print(f"  Motor responding! Position: {pos:.4f} rad")
    print()

    # Step 5: Disable again (leave motor safe)
    print("Step 5: Disabling motor (safe state)...")
    send_control_cmd(bus, SLAVE_ID, 0xFD)
    time.sleep(0.2)
    read_responses(bus, timeout=0.3)

    bus.shutdown()
    print()
    print("=== Done ===")
    print("If motor responded: LED should change from solid red to normal.")
    print("If no response: motor needs power cycle (disconnect & reconnect power).")

if __name__ == "__main__":
    main()
