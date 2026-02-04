#!/usr/bin/env python3
"""Diagnostic test: which CAN command modes does the Damiao motor respond to?

Test 1: POS_VEL mode (0x100+ID) — what we've been trying
Test 2: VEL mode (0x200+ID) — velocity-only command
Test 3: MIT mode (SlaveID) — low-level torque/position control

This isolates whether the motor is genuinely in POS_VEL mode or stuck in MIT.
"""

import sys, os, time, math, struct, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lerobot", "src"))
import can


class Motor:
    def __init__(self, can_id, master_id):
        self.SlaveID = can_id
        self.MasterID = master_id
        self.state_q = 0.0
        self.state_dq = 0.0
        self.state_tau = 0.0
        self.error_flags = 0
        self.param_dict = {}
    def getPosition(self): return self.state_q
    def getVelocity(self): return self.state_dq
    def getTorque(self): return self.state_tau


class CANControl:
    INT_RIDS = set(range(7, 11)) | set(range(13, 17)) | {35, 36}

    def __init__(self, channel='can0'):
        self.bus = can.interface.Bus(channel=channel, interface='socketcan')
        self.motors = {}
        self.running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def add_motor(self, motor):
        self.motors[motor.SlaveID] = motor

    def _recv_loop(self):
        while self.running and self.bus:
            try:
                msg = self.bus.recv(timeout=1.0)
                if msg:
                    self._parse(msg)
            except: pass

    def _parse(self, msg):
        data = msg.data
        if len(data) < 8: return
        if data[2] in (0x33, 0x55):
            slave_id = (data[1] << 8) | data[0]
            rid = data[3]
            motor = self.motors.get(slave_id)
            if motor:
                if rid in self.INT_RIDS:
                    val = struct.unpack('<I', bytes(data[4:8]))[0]
                else:
                    val = struct.unpack('<f', bytes(data[4:8]))[0]
                motor.param_dict[rid] = val
            return
        esc_id = data[0] & 0x0F
        motor = self.motors.get(esc_id)
        if motor:
            motor.error_flags = (data[0] >> 4) & 0x0F
            q_uint = (data[1] << 8) | data[2]
            dq_uint = (data[3] << 4) | (data[4] >> 4)
            tau_uint = ((data[4] & 0x0F) << 8) | data[5]
            motor.state_q = self._u2f(q_uint, -12.5, 12.5, 16)
            motor.state_dq = self._u2f(dq_uint, -30.0, 30.0, 12)
            motor.state_tau = self._u2f(tau_uint, -10.0, 10.0, 12)

    def _u2f(self, x, x_min, x_max, bits):
        return float(x) / ((1 << bits) - 1) * (x_max - x_min) + x_min

    def _send(self, arb_id, data):
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
        self.bus.send(msg)

    def refresh(self, motor):
        cid_l = motor.SlaveID & 0xFF
        cid_h = (motor.SlaveID >> 8) & 0xFF
        self._send(0x7FF, bytes([cid_l, cid_h, 0xCC, 0, 0, 0, 0, 0]))
        time.sleep(0.005)

    def read_param(self, motor, rid, timeout=0.5):
        motor.param_dict.pop(rid, None)
        cid_l = motor.SlaveID & 0xFF
        cid_h = (motor.SlaveID >> 8) & 0xFF
        self._send(0x7FF, bytearray([cid_l, cid_h, 0x33, rid, 0, 0, 0, 0]))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if rid in motor.param_dict:
                return motor.param_dict[rid]
            time.sleep(0.01)
        return None

    def write_param(self, motor, rid, value):
        cid_l = motor.SlaveID & 0xFF
        cid_h = (motor.SlaveID >> 8) & 0xFF
        if isinstance(value, int) or rid in self.INT_RIDS:
            val_bytes = struct.pack('<I', int(value))
        else:
            val_bytes = struct.pack('<f', float(value))
        data = bytearray(8)
        data[0], data[1] = cid_l, cid_h
        data[2] = 0x55
        data[3] = rid
        data[4:8] = val_bytes
        self._send(0x7FF, data)
        time.sleep(0.02)

    def switch_mode(self, motor, mode):
        self.write_param(motor, 10, mode)

    def save_params(self, motor):
        cid_l = motor.SlaveID & 0xFF
        cid_h = (motor.SlaveID >> 8) & 0xFF
        self._send(0x7FF, bytearray([cid_l, cid_h, 0xAA, 0, 0, 0, 0, 0]))
        time.sleep(0.1)

    def enable(self, motor):
        self._send(motor.SlaveID, bytes([0xFF]*7 + [0xFC]))
        time.sleep(0.1)

    def disable(self, motor):
        self._send(motor.SlaveID, bytes([0xFF]*7 + [0xFD]))
        time.sleep(0.05)

    def shutdown(self):
        self.running = False
        if self.bus: self.bus.shutdown()

    # --- Control modes ---

    def control_pos_vel(self, motor, p_des, v_des):
        """POS_VEL mode: arb_id = 0x100 + SlaveID"""
        data = struct.pack('<ff', p_des, v_des)
        self._send(0x100 + motor.SlaveID, data)

    def control_vel(self, motor, v_des):
        """VEL mode: arb_id = 0x200 + SlaveID"""
        data = bytearray(8)
        data[0:4] = struct.pack('<f', v_des)
        self._send(0x200 + motor.SlaveID, data)

    def control_mit(self, motor, p_des, v_des, kp, kd, t_ff,
                    p_max=12.5, v_max=30.0, t_max=10.0):
        """MIT mode: arb_id = SlaveID (low-level torque control)"""
        def f2u(x, x_min, x_max, bits):
            span = x_max - x_min
            x = max(x_min, min(x_max, x))
            return int((x - x_min) / span * ((1 << bits) - 1))

        p_int = f2u(p_des, -p_max, p_max, 16)
        v_int = f2u(v_des, -v_max, v_max, 12)
        kp_int = f2u(kp, 0, 500, 12)
        kd_int = f2u(kd, 0, 5, 12)
        t_int = f2u(t_ff, -t_max, t_max, 12)

        data = bytearray(8)
        data[0] = (p_int >> 8) & 0xFF
        data[1] = p_int & 0xFF
        data[2] = (v_int >> 4) & 0xFF
        data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
        data[4] = kp_int & 0xFF
        data[5] = (kd_int >> 4) & 0xFF
        data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)
        data[7] = t_int & 0xFF
        self._send(motor.SlaveID, data)


def get_pos(ctrl, motor):
    ctrl.refresh(motor)
    time.sleep(0.005)
    return motor.getPosition()


def test_pos_vel(ctrl, motor, start_pos):
    """Test 1: POS_VEL mode (0x100+ID)"""
    print("\n" + "="*60)
    print("TEST 1: POS_VEL mode (arb_id = 0x100 + ID)")
    print("="*60)

    ctrl.disable(motor)
    ctrl.switch_mode(motor, 2)  # POS_VEL
    time.sleep(0.02)
    ctrl.enable(motor)

    pos_before = get_pos(ctrl, motor)
    print(f"  Before: {pos_before:.4f} rad")

    # Send 200 commands: go to start_pos + 0.5
    target = start_pos + 0.5
    for i in range(200):
        ctrl.control_pos_vel(motor, target, 5.0)
        time.sleep(0.005)

    pos_after = get_pos(ctrl, motor)
    moved = abs(pos_after - pos_before) > 0.01
    print(f"  After:  {pos_after:.4f} rad  (target was {target:.4f})")
    print(f"  RESULT: {'MOVED' if moved else 'NO MOVEMENT'}")
    return moved


def test_vel(ctrl, motor):
    """Test 2: VEL mode (0x200+ID)"""
    print("\n" + "="*60)
    print("TEST 2: VEL mode (arb_id = 0x200 + ID)")
    print("="*60)

    ctrl.disable(motor)
    ctrl.switch_mode(motor, 3)  # VEL
    time.sleep(0.02)
    ctrl.enable(motor)

    pos_before = get_pos(ctrl, motor)
    print(f"  Before: {pos_before:.4f} rad")

    # Send velocity command: 2 rad/s for 1 second
    for i in range(200):
        ctrl.control_vel(motor, 2.0)
        time.sleep(0.005)

    pos_after = get_pos(ctrl, motor)
    moved = abs(pos_after - pos_before) > 0.01
    print(f"  After:  {pos_after:.4f} rad")
    print(f"  RESULT: {'MOVED' if moved else 'NO MOVEMENT'}")

    # Stop
    for _ in range(10):
        ctrl.control_vel(motor, 0.0)
        time.sleep(0.005)

    return moved


def test_mit(ctrl, motor, start_pos):
    """Test 3: MIT mode (arb_id = SlaveID)"""
    print("\n" + "="*60)
    print("TEST 3: MIT mode (arb_id = SlaveID)")
    print("="*60)

    ctrl.disable(motor)
    ctrl.switch_mode(motor, 1)  # MIT
    time.sleep(0.02)
    ctrl.enable(motor)

    pos_before = get_pos(ctrl, motor)
    print(f"  Before: {pos_before:.4f} rad")

    # Send MIT command: position with KP gain, no velocity/torque
    target = start_pos + 0.3
    for i in range(200):
        ctrl.control_mit(motor, target, 0.0, kp=50.0, kd=1.0, t_ff=0.0)
        time.sleep(0.005)

    pos_after = get_pos(ctrl, motor)
    moved = abs(pos_after - pos_before) > 0.01
    print(f"  After:  {pos_after:.4f} rad  (target was {target:.4f})")
    print(f"  RESULT: {'MOVED' if moved else 'NO MOVEMENT'}")
    return moved


def main():
    MOTOR_ID = 1
    MASTER_ID = 0x11
    CHANNEL = 'can0'

    print("="*60)
    print("DAMIAO MOTOR MODE DIAGNOSTIC")
    print("="*60)
    print(f"Testing motor {MOTOR_ID} on {CHANNEL}")
    print("This will test POS_VEL, VEL, and MIT modes to find which works.\n")

    ctrl = CANControl(CHANNEL)
    motor = Motor(MOTOR_ID, MASTER_ID)
    ctrl.add_motor(motor)
    time.sleep(0.3)

    # Check motor responds
    for _ in range(3):
        ctrl.refresh(motor)
    mode = ctrl.read_param(motor, 10)
    if mode is None:
        print("ERROR: Motor not responding!")
        ctrl.shutdown()
        return

    start_pos = get_pos(ctrl, motor)
    print(f"Motor responding. Current mode={mode}, pos={start_pos:.4f} rad\n")

    results = {}

    # Test 1: POS_VEL
    try:
        results['pos_vel'] = test_pos_vel(ctrl, motor, start_pos)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['pos_vel'] = False

    time.sleep(0.5)
    start_pos = get_pos(ctrl, motor)  # re-read in case it moved

    # Test 2: VEL
    try:
        results['vel'] = test_vel(ctrl, motor)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['vel'] = False

    time.sleep(0.5)
    start_pos = get_pos(ctrl, motor)

    # Test 3: MIT
    try:
        results['mit'] = test_mit(ctrl, motor, start_pos)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['mit'] = False

    # Cleanup
    ctrl.disable(motor)
    ctrl.shutdown()

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for mode_name, worked in results.items():
        status = "WORKS" if worked else "FAILED"
        print(f"  {mode_name:10s}: {status}")

    print()
    if results.get('mit') and not results.get('pos_vel'):
        print("DIAGNOSIS: Motor responds to MIT but NOT POS_VEL.")
        print("  The motor may be stuck in MIT mode despite reporting POS_VEL.")
        print("  FIX: Use MIT mode for control, OR power-cycle the motor after mode switch.")
    elif results.get('vel') and not results.get('pos_vel'):
        print("DIAGNOSIS: Motor responds to VEL but NOT POS_VEL.")
        print("  POS_VEL frame format may be incorrect for this motor firmware.")
    elif not any(results.values()):
        print("DIAGNOSIS: Motor doesn't respond to ANY control mode via SocketCAN.")
        print("  The SocketCAN adapter may not be transmitting correctly.")
        print("  Try a different CAN adapter or use the serial CAN bridge.")
    else:
        print("DIAGNOSIS: Check which modes work and adapt accordingly.")


if __name__ == "__main__":
    main()
