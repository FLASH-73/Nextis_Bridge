#!/usr/bin/env python3
"""
Interactive gripper calibration for Umbra Leader + Follower.

Guides you through capturing the physical open/closed positions of each gripper,
then writes corrected calibration to the runtime JSON files.

Usage:
    python calibrate_grippers.py

Requires: The app backend must NOT be running (serial ports must be free).
"""

import glob as glob_mod
import json
import sys
import time
import threading
from pathlib import Path

# Preferred ports (from settings.yaml) — used as hints, auto-detected if wrong
PREFERRED_LEADER_PORT = "/dev/ttyACM1"
PREFERRED_FOLLOWER_PORT = "/dev/ttyUSB1"

# Motor IDs
LEADER_GRIPPER_ID = 7
FOLLOWER_GRIPPER_ID = 9

# Runtime calibration file paths
LEADER_CAL_PATH = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/dynamixel_leader/umbra_leader.json"
FOLLOWER_CAL_PATH = Path.home() / ".cache/huggingface/lerobot/calibration/robots/umbra_follower/umbra_follower.json"


def find_motor_port(motor_id, bus_class, motor_model, norm_mode, preferred_port=None):
    """Auto-detect which serial port has the target motor.

    Scans all /dev/ttyACM* and /dev/ttyUSB* ports, trying preferred_port first.
    Returns (port, found_motors_dict) or (None, None) if not found.
    """
    from lerobot.motors.motors_bus import Motor

    candidates = sorted(glob_mod.glob("/dev/ttyACM*") + glob_mod.glob("/dev/ttyUSB*"))

    if not candidates:
        print("  ERROR: No serial ports found (/dev/ttyACM* or /dev/ttyUSB*).")
        print("  Check: Is the USB cable connected?")
        return None

    # Try preferred port first
    if preferred_port and preferred_port in candidates:
        candidates.remove(preferred_port)
        candidates.insert(0, preferred_port)

    print(f"  Scanning ports for motor ID {motor_id}: {', '.join(candidates)}")

    for port in candidates:
        try:
            bus = bus_class(
                port=port,
                motors={"probe": Motor(motor_id, motor_model, norm_mode)},
            )
            bus._connect(handshake=False)
            bus.set_timeout()
            result = bus.broadcast_ping()
            bus.port_handler.closePort()

            if result:
                ids = list(result.keys())
                if motor_id in result:
                    print(f"  Found motor {motor_id} on {port} (all motors on bus: {ids})")
                    return port
                else:
                    print(f"    {port}: found motors {ids} (not ID {motor_id})")
            else:
                print(f"    {port}: no motors responded")
        except Exception as e:
            print(f"    {port}: error — {e}")
            continue

    print(f"\n  ERROR: Motor ID {motor_id} not found on any port.")
    print("  Check: Is the arm powered? Is the USB cable connected?")
    return None


def wait_for_enter_with_live_ticks(bus, motor_name):
    """Print live raw ticks while waiting for user to press Enter."""
    stop = threading.Event()

    def print_loop():
        while not stop.is_set():
            try:
                pos = bus.sync_read("Present_Position", motor_name, normalize=False)
                raw = pos[motor_name]
                print(f"\r  Raw ticks: {int(raw):>5}    ", end="", flush=True)
            except Exception:
                pass
            time.sleep(0.15)

    t = threading.Thread(target=print_loop, daemon=True)
    t.start()
    input()  # Block until Enter
    stop.set()
    t.join(timeout=0.5)

    # Read final position
    pos = bus.sync_read("Present_Position", motor_name, normalize=False)
    raw = int(pos[motor_name])
    print(f"\r  Captured: {raw}          ")
    return raw


def wait_for_enter_with_wrap_tracking(bus, motor_name):
    """Print live raw ticks while waiting for Enter, tracking encoder wraps.

    Detects when the 12-bit encoder crosses the 4095→0 boundary (or vice versa)
    by monitoring consecutive readings for jumps > 2048 ticks.

    Returns (final_raw_pos, wrap_count) where wrap_count is:
      +1 per upward wrap (4095→0, ticks were increasing)
      -1 per downward wrap (0→4095, ticks were decreasing)
    """
    stop = threading.Event()
    wrap_count = [0]  # mutable container for thread access
    prev_raw = [None]

    def print_loop():
        while not stop.is_set():
            try:
                pos = bus.sync_read("Present_Position", motor_name, normalize=False)
                raw = int(pos[motor_name])

                if prev_raw[0] is not None:
                    delta = raw - prev_raw[0]
                    if delta < -2048:
                        # Jump from high to low → wrapped UP through 4095→0
                        wrap_count[0] += 1
                    elif delta > 2048:
                        # Jump from low to high → wrapped DOWN through 0→4095
                        wrap_count[0] -= 1

                prev_raw[0] = raw

                wraps_str = f" (wraps: {wrap_count[0]})" if wrap_count[0] != 0 else ""
                print(f"\r  Raw ticks: {raw:>5}{wraps_str}    ", end="", flush=True)
            except Exception:
                pass
            time.sleep(0.15)

    t = threading.Thread(target=print_loop, daemon=True)
    t.start()
    input()  # Block until Enter
    stop.set()
    t.join(timeout=0.5)

    # Read final position
    pos = bus.sync_read("Present_Position", motor_name, normalize=False)
    raw = int(pos[motor_name])
    wraps_str = f" (wraps: {wrap_count[0]})" if wrap_count[0] != 0 else ""
    print(f"\r  Captured: {raw}{wraps_str}          ")
    return raw, wrap_count[0]


def calibrate_leader():
    """Calibrate Dynamixel XL330 leader gripper (motor 7)."""
    from lerobot.motors.dynamixel import DynamixelMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    # Auto-detect port
    port = find_motor_port(
        motor_id=LEADER_GRIPPER_ID,
        bus_class=DynamixelMotorsBus,
        motor_model="xl330-m077",
        norm_mode=MotorNormMode.DEGREES,
        preferred_port=PREFERRED_LEADER_PORT,
    )
    if port is None:
        sys.exit(1)

    print(f"\n  Connecting to Dynamixel bus on {port}...")
    bus = DynamixelMotorsBus(
        port=port,
        motors={"gripper": Motor(LEADER_GRIPPER_ID, "xl330-m077", MotorNormMode.DEGREES)},
    )
    bus.connect(handshake=False)

    # Reboot motor to clear hardware errors and reset multi-turn counter.
    # This matches what the app's _handshake() does on connect — without it,
    # positions can be in multi-turn space (>4095) which becomes invalid after
    # the app reboots the motor.
    print("  Rebooting motor to clear errors and reset position counter...")
    bus.packet_handler.reboot(bus.port_handler, LEADER_GRIPPER_ID)
    time.sleep(2.0)

    # Disable torque so user can move freely
    bus.write("Torque_Enable", "gripper", 0, normalize=False)
    print("  Connected. Torque disabled — gripper moves freely.\n")

    # Capture closed position
    print("  Move the leader gripper to FULLY CLOSED, then press Enter.")
    closed_ticks = wait_for_enter_with_live_ticks(bus, "gripper")

    # Capture open position
    print("\n  Move the leader gripper to FULLY OPEN, then press Enter.")
    open_ticks = wait_for_enter_with_live_ticks(bus, "gripper")

    bus.disconnect()

    range_size = abs(closed_ticks - open_ticks)
    print(f"\n  Leader gripper: open={open_ticks}, closed={closed_ticks}, range={range_size} ticks")

    if range_size < 50:
        print("  WARNING: Range seems too small. Did you move the gripper?")

    return open_ticks, closed_ticks


def calibrate_follower():
    """Calibrate Feetech STS3215 follower gripper (motor 9)."""
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    # Auto-detect port
    port = find_motor_port(
        motor_id=FOLLOWER_GRIPPER_ID,
        bus_class=FeetechMotorsBus,
        motor_model="sts3215",
        norm_mode=MotorNormMode.RANGE_0_100,
        preferred_port=PREFERRED_FOLLOWER_PORT,
    )
    if port is None:
        sys.exit(1)

    print(f"\n  Connecting to Feetech bus on {port}...")
    bus = FeetechMotorsBus(
        port=port,
        motors={"gripper": Motor(FOLLOWER_GRIPPER_ID, "sts3215", MotorNormMode.RANGE_0_100)},
    )
    bus.connect(handshake=False)

    # Disable torque
    bus.write("Torque_Enable", "gripper", 0, normalize=False)

    # Reset Homing_Offset to 0 so we read raw actual encoder positions
    bus.write("Homing_Offset", "gripper", 0, normalize=False)
    time.sleep(0.2)
    print("  Connected. Torque disabled — gripper moves freely.\n")

    # Capture open position (simple — no wrap tracking needed)
    print("  Move the follower gripper to FULLY OPEN, then press Enter.")
    open_ticks = wait_for_enter_with_live_ticks(bus, "gripper")

    # Capture closed position WITH wrap tracking
    # The gripper may travel >360° crossing the 4095→0 encoder boundary
    print("\n  Now SLOWLY move the follower gripper to FULLY CLOSED, then press Enter.")
    print("  (Move continuously so the script can track encoder wraps)")
    closed_ticks, wraps = wait_for_enter_with_wrap_tracking(bus, "gripper")

    bus.disconnect()

    travel = (closed_ticks - open_ticks) + wraps * 4096
    print(f"\n  Follower gripper: open={open_ticks}, closed={closed_ticks}, wraps={wraps}")
    print(f"  Travel: {abs(travel)} ticks ({'wraps detected — will use homing_offset' if wraps else 'no wrapping'})")

    if abs(travel) < 50:
        print("  WARNING: Range seems too small. Did you move the gripper?")

    return open_ticks, closed_ticks, wraps


def update_calibration_file(cal_path, gripper_key, new_gripper_cal):
    """Update only the gripper entry in a calibration JSON file, preserving everything else."""
    if not cal_path.exists():
        print(f"  ERROR: Calibration file not found: {cal_path}")
        return False

    with open(cal_path) as f:
        cal = json.load(f)

    old = cal.get(gripper_key, {})
    print(f"  Old:  homing_offset={old.get('homing_offset')}, range_min={old.get('range_min')}, range_max={old.get('range_max')}")

    cal[gripper_key] = new_gripper_cal
    print(f"  New:  homing_offset={new_gripper_cal['homing_offset']}, range_min={new_gripper_cal['range_min']}, range_max={new_gripper_cal['range_max']}")

    # Write with nice formatting
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=4)

    print(f"  Saved: {cal_path}")
    return True


def main():
    print("=" * 50)
    print("  Gripper Calibration — Umbra Leader + Follower")
    print("=" * 50)
    print("\n  Make sure the app backend is NOT running.")
    print("  This script needs exclusive access to the serial ports.\n")

    # ── LEADER ──
    print("─" * 50)
    print("  STEP 1: Leader Gripper (Dynamixel XL330, ID 7)")
    print("─" * 50)

    try:
        leader_open, leader_closed = calibrate_leader()
    except Exception as e:
        print(f"\n  ERROR connecting to leader: {e}")
        sys.exit(1)

    # ── FOLLOWER ──
    print("\n" + "─" * 50)
    print("  STEP 2: Follower Gripper (Feetech STS3215, ID 9)")
    print("─" * 50)

    try:
        follower_open, follower_closed, wraps = calibrate_follower()
    except Exception as e:
        print(f"\n  ERROR connecting to follower: {e}")
        sys.exit(1)

    # ── SUMMARY & SAVE ──
    print("\n" + "=" * 50)
    print("  STEP 3: Save Calibration")
    print("=" * 50)

    # Leader: range_min = open, range_max = closed
    # DynamixelLeader.configure(): _gripper_open = range_min, _gripper_closed = range_max
    # get_action(): normalized = 1 - (raw - closed) / (open - closed) → 0=open, 1=closed
    leader_gripper_cal = {
        "id": LEADER_GRIPPER_ID,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": leader_open,
        "range_max": leader_closed,
    }

    # Follower: handle encoder wrapping for >360° travel
    # RANGE_0_100 normalization: 0% → range_min (open), 100% → range_max (closed)
    travel = (follower_closed - follower_open) + wraps * 4096

    if wraps == 0:
        # No wrapping — simple min/max
        f_homing_offset = 0
        f_range_min = min(follower_open, follower_closed)
        f_range_max = max(follower_open, follower_closed)
        print(f"\n  Follower: no encoder wrapping (travel={abs(travel)} ticks)")
    elif abs(travel) <= 4096:
        # Wrapping detected — use homing_offset to shift zero into dead zone
        gap_center = (follower_open + follower_closed) // 2
        f_homing_offset = gap_center
        f_range_min = (follower_open - f_homing_offset) % 4096
        f_range_max = (follower_closed - f_homing_offset) % 4096
        print(f"\n  Follower: encoder wrapping detected (wraps={wraps}, travel={abs(travel)} ticks)")
        print(f"  Applying homing_offset={f_homing_offset} to shift zero into dead zone")
        print(f"  Offset-adjusted: open={f_range_min}, closed={f_range_max}")
        if f_range_min >= f_range_max:
            print("  ERROR: Unexpected range order after offset. Check wrapping direction.")
            sys.exit(1)
    else:
        print(f"\n  ERROR: Travel ({abs(travel)} ticks) exceeds one full rotation (4096).")
        print("  Cannot fix with homing_offset alone. Check the gripper mechanism.")
        sys.exit(1)

    follower_gripper_cal = {
        "id": FOLLOWER_GRIPPER_ID,
        "drive_mode": 0,
        "homing_offset": f_homing_offset,
        "range_min": f_range_min,
        "range_max": f_range_max,
    }

    print(f"\n  Leader calibration file: {LEADER_CAL_PATH}")
    ok1 = update_calibration_file(LEADER_CAL_PATH, "gripper", leader_gripper_cal)

    print(f"\n  Follower calibration file: {FOLLOWER_CAL_PATH}")
    ok2 = update_calibration_file(FOLLOWER_CAL_PATH, "gripper", follower_gripper_cal)

    # ── DONE ──
    print("\n" + "=" * 50)
    if ok1 and ok2:
        print("  Done! Both calibration files updated.")
        print("\n  Summary:")
        print(f"    Leader:   open={leader_open}, closed={leader_closed} (range {abs(leader_closed - leader_open)})")
        print(f"    Follower: open={follower_open}, closed={follower_closed}, wraps={wraps}")
        if wraps != 0:
            print(f"              homing_offset={f_homing_offset}, range_min={f_range_min}, range_max={f_range_max}")
        else:
            print(f"              range_min={f_range_min}, range_max={f_range_max}")
        print(f"\n  Next steps:")
        print(f"    1. Start the app backend")
        print(f"    2. Connect the Umbra Leader — gripper should spring to open, NO red LED")
        print(f"    3. Start teleop — fully close the leader → follower should fully close")
    else:
        print("  Some files could not be updated. Check the errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
