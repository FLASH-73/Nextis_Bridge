#!/usr/bin/env python3
"""
Comprehensive Leader Gripper Diagnostic.

Finds BOTH calibration files (default + profile), connects to the leader
exactly like the app does, and walks through the full normalization pipeline.

Usage:
    /home/roberto/miniconda3/bin/python3 gripper_diagnostic.py
"""
import sys
import time
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

PORT = "/dev/ttyACM0"
ARM_ID = "aira_lead"
FOLLOWER_OPEN = 0.0      # radians
FOLLOWER_CLOSED = -5.27   # radians

# Paths
DEFAULT_CAL = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/dynamixel_leader" / f"{ARM_ID}.json"
PROFILE_CAL = Path("calibration_profiles/aira_lead/cal_test8.json")
RESULTS_FILE = Path("gripper_diagnostic_results.json")

results = {}
bus = None


def map_range(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def normalize_gripper(homed_val, gripper_open, gripper_closed):
    gripper_range = gripper_open - gripper_closed
    if gripper_range == 0:
        return 0.0
    return 1.0 - (homed_val - gripper_closed) / gripper_range


def read_cal_file(path, label):
    print(f"\n  [{label}] {path}")
    if not path.exists():
        print(f"    FILE NOT FOUND")
        return None
    try:
        data = json.loads(path.read_text())
        g = data.get("gripper", {})
        print(f"    gripper: offset={g.get('homing_offset')}, "
              f"range=[{g.get('range_min')}, {g.get('range_max')}]")
        return data
    except Exception as e:
        print(f"    ERROR reading: {e}")
        return None


try:
    print("=" * 65)
    print("  COMPREHENSIVE LEADER GRIPPER DIAGNOSTIC")
    print("=" * 65)

    # ── Step 1: Find both calibration files ──────────────────────────
    print("\n[1] CALIBRATION FILES")
    print("    The app loads from DEFAULT at startup.")
    print("    Profile is only loaded when you click 'Load' in the GUI.")

    default_data = read_cal_file(DEFAULT_CAL, "DEFAULT (loaded at startup)")
    profile_data = read_cal_file(PROFILE_CAL, "PROFILE (loaded on demand) ")

    default_gripper = default_data.get("gripper", {}) if default_data else {}
    profile_gripper = profile_data.get("gripper", {}) if profile_data else {}

    if default_gripper and profile_gripper:
        match = (default_gripper.get("homing_offset") == profile_gripper.get("homing_offset")
                 and default_gripper.get("range_min") == profile_gripper.get("range_min")
                 and default_gripper.get("range_max") == profile_gripper.get("range_max"))
        if match:
            print("\n    Files MATCH for gripper.")
        else:
            print("\n    *** FILES DIFFER! The app uses DEFAULT, not profile. ***")
            print(f"    DEFAULT: offset={default_gripper.get('homing_offset')}, "
                  f"range=[{default_gripper.get('range_min')}, {default_gripper.get('range_max')}]")
            print(f"    PROFILE: offset={profile_gripper.get('homing_offset')}, "
                  f"range=[{profile_gripper.get('range_min')}, {profile_gripper.get('range_max')}]")

    results["default_cal_path"] = str(DEFAULT_CAL)
    results["profile_cal_path"] = str(PROFILE_CAL)
    results["default_gripper"] = default_gripper
    results["profile_gripper"] = profile_gripper

    # Use the DEFAULT calibration (same as app)
    active_cal = default_gripper
    offset = active_cal.get("homing_offset", 0) if active_cal else 0
    cal_min = active_cal.get("range_min") if active_cal else None
    cal_max = active_cal.get("range_max") if active_cal else None

    # ── Step 2: Connect to leader bus ────────────────────────────────
    print("\n[2] CONNECTING TO LEADER BUS")
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.dynamixel import DynamixelMotorsBus
    from lerobot.motors.motors_bus import MotorCalibration

    bus = DynamixelMotorsBus(
        port=PORT,
        motors={
            "joint_1": Motor(1, "xl330-m077", MotorNormMode.DEGREES),
            "joint_2": Motor(2, "xl330-m077", MotorNormMode.DEGREES),
            "joint_3": Motor(3, "xl330-m077", MotorNormMode.DEGREES),
            "joint_4": Motor(4, "xl330-m077", MotorNormMode.DEGREES),
            "joint_5": Motor(5, "xl330-m077", MotorNormMode.DEGREES),
            "joint_6": Motor(6, "xl330-m077", MotorNormMode.DEGREES),
            "gripper": Motor(7, "xl330-m077", MotorNormMode.DEGREES),
        },
    )
    bus.connect()
    print(f"    Connected to {PORT}")

    # ── Step 3: Apply calibration (same as app connect()) ────────────
    print("\n[3] APPLYING CALIBRATION (same as app)")
    if default_data:
        bus.disable_torque()
        cal_dict = {}
        for motor_name, mc in default_data.items():
            cal_dict[motor_name] = MotorCalibration(
                id=mc["id"], drive_mode=mc["drive_mode"],
                homing_offset=mc["homing_offset"],
                range_min=mc["range_min"], range_max=mc["range_max"],
            )
        bus.write_calibration(cal_dict)

        gripper_id = bus.motors["gripper"].id
        sw_offset = bus._software_homing_offsets.get(gripper_id, 0)
        print(f"    write_calibration() done")
        print(f"    Software homing offset for gripper (id={gripper_id}): {sw_offset}")

        # Read back position limits from motor
        try:
            min_lim = bus.read("Min_Position_Limit", "gripper", normalize=False)
            max_lim = bus.read("Max_Position_Limit", "gripper", normalize=False)
            print(f"    Motor Min_Position_Limit = {min_lim}")
            print(f"    Motor Max_Position_Limit = {max_lim}")
            results["motor_min_position_limit"] = min_lim
            results["motor_max_position_limit"] = max_lim
        except Exception as e:
            print(f"    Could not read position limits: {e}")
    else:
        sw_offset = 0
        print("    No calibration to apply")

    results["software_homing_offset"] = sw_offset

    # ── Step 4: Show what configure() would compute ──────────────────
    print("\n[4] SIMULATING configure() GRIPPER LOGIC")
    if cal_min is not None:
        _gripper_open = cal_min + sw_offset
        _gripper_closed = cal_max + sw_offset
        spring_target = cal_min
        print(f"    CALIBRATED path (range_min=open, range_max=closed):")
        print(f"      _gripper_open  = range_min({cal_min}) + offset({sw_offset}) = {_gripper_open}")
        print(f"      _gripper_closed = range_max({cal_max}) + offset({sw_offset}) = {_gripper_closed}")
        print(f"      spring_target  = {spring_target} (raw)")
        in_limits = True
        if 'min_lim' in dir() and 'max_lim' in dir():
            in_limits = min_lim <= spring_target <= max_lim
            print(f"      spring_target in motor limits [{min_lim},{max_lim}]? {in_limits}")
    else:
        _gripper_open = 2820 + sw_offset
        _gripper_closed = 2280 + sw_offset
        spring_target = 2820
        print(f"    FALLBACK (config defaults):")
        print(f"      _gripper_open  = {_gripper_open}")
        print(f"      _gripper_closed = {_gripper_closed}")

    gripper_range = _gripper_open - _gripper_closed
    print(f"      gripper_range  = {gripper_range}")
    results["_gripper_open"] = _gripper_open
    results["_gripper_closed"] = _gripper_closed
    results["gripper_range"] = gripper_range
    results["spring_target"] = spring_target

    # ── Step 5: Verify read() vs sync_read() ─────────────────────────
    print("\n[5] COMPARING read() vs sync_read()")
    raw_read = bus.read("Present_Position", "gripper", normalize=False)
    sync_all = bus.sync_read(normalize=False, data_name="Present_Position")
    sync_val = sync_all.get("gripper", "N/A")
    print(f"    bus.read(normalize=False)      = {raw_read}")
    print(f"    bus.sync_read(normalize=False)  = {sync_val}")
    print(f"    Difference                      = {sync_val - raw_read if isinstance(sync_val, (int, float)) else 'N/A'}")
    print(f"    (Both should include software offset of {sw_offset})")
    results["read_value"] = raw_read
    results["sync_read_value"] = sync_val

    # ── Step 6: Interactive OPEN position ─────────────────────────────
    print("\n" + "=" * 65)
    print("[6] Move gripper to FULLY OPEN (relaxed). Press Enter...")
    input()

    readings = []
    for i in range(5):
        readings.append(bus.sync_read(normalize=False, data_name="Present_Position")["gripper"])
        time.sleep(0.05)
    open_homed = int(sum(readings) / len(readings))
    open_raw = open_homed - sw_offset
    norm_open = normalize_gripper(open_homed, _gripper_open, _gripper_closed)
    follower_open_rad = map_range(max(0.0, min(1.0, norm_open)), 0.0, 1.0, FOLLOWER_OPEN, FOLLOWER_CLOSED)

    print(f"    Raw (no offset)   = {open_raw}")
    print(f"    Homed (sync_read) = {open_homed}")
    print(f"    Normalized        = {norm_open:.4f}")
    print(f"    Follower would get = {follower_open_rad:.3f} rad")
    print(f"    EXPECTED: norm ~ 0.0, follower ~ 0.0 rad")
    results["open"] = {"raw": open_raw, "homed": open_homed,
                       "normalized": round(norm_open, 4),
                       "follower_rad": round(follower_open_rad, 3)}

    # ── Step 7: Interactive CLOSED position ───────────────────────────
    print(f"\n[7] Move gripper to FULLY CLOSED (squeeze). Press Enter...")
    input()

    readings = []
    for i in range(5):
        readings.append(bus.sync_read(normalize=False, data_name="Present_Position")["gripper"])
        time.sleep(0.05)
    closed_homed = int(sum(readings) / len(readings))
    closed_raw = closed_homed - sw_offset
    norm_closed = normalize_gripper(closed_homed, _gripper_open, _gripper_closed)
    follower_closed_rad = map_range(max(0.0, min(1.0, norm_closed)), 0.0, 1.0, FOLLOWER_OPEN, FOLLOWER_CLOSED)

    print(f"    Raw (no offset)   = {closed_raw}")
    print(f"    Homed (sync_read) = {closed_homed}")
    print(f"    Normalized        = {norm_closed:.4f}")
    print(f"    Follower would get = {follower_closed_rad:.3f} rad")
    print(f"    EXPECTED: norm ~ 1.0, follower ~ -5.27 rad")
    results["closed"] = {"raw": closed_raw, "homed": closed_homed,
                         "normalized": round(norm_closed, 4),
                         "follower_rad": round(follower_closed_rad, 3)}

    # ── Step 8: Analysis ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("[8] ANALYSIS")
    print(f"    Open:   raw={open_raw}, homed={open_homed} → norm={norm_open:.3f}")
    print(f"    Closed: raw={closed_raw}, homed={closed_homed} → norm={norm_closed:.3f}")

    ok = abs(norm_open) < 0.15 and abs(norm_closed - 1.0) < 0.15
    inverted = abs(norm_open - 1.0) < 0.15 and abs(norm_closed) < 0.15

    if ok:
        print("\n    CORRECT: open~0, closed~1. Pipeline works!")
    elif inverted:
        print("\n    INVERTED: open~1, closed~0.")
        print("    Fix: swap range_min and range_max in calibration.")
    else:
        print(f"\n    BAD: values not near 0 or 1.")
        print(f"    The calibration range doesn't match the motor's actual range.")

    # Recommend correct calibration values (always using raw, offset=0)
    actual_open_raw = min(open_raw, closed_raw) if (open_raw < closed_raw) else open_raw
    actual_closed_raw = max(open_raw, closed_raw) if (open_raw < closed_raw) else closed_raw
    # Determine direction: which raw value = open?
    recommended = {
        "id": 7, "drive_mode": 0, "homing_offset": 0,
        "range_min": open_raw,   # raw value at physical open
        "range_max": closed_raw,  # raw value at physical closed
        "note": "range_min=open (lower ticks), range_max=closed (higher ticks)"
    }
    # Correct for swapped direction
    if open_raw > closed_raw:
        recommended["range_min"] = closed_raw
        recommended["range_max"] = open_raw
        recommended["note"] = "range_min=closed (lower ticks), range_max=open (higher ticks)"

    print(f"\n    RECOMMENDED calibration (gripper):")
    print(f"    {json.dumps(recommended, indent=4)}")
    print(f"\n    Files to update:")
    print(f"      {DEFAULT_CAL}")
    print(f"      {PROFILE_CAL}")
    results["recommended"] = recommended

    # ── Step 9: Live monitor ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("[9] LIVE MONITOR (Ctrl+C to stop)")
    print(f"    {'Homed':>8}  {'Norm':>8}  {'Follower':>10}  {'Bar'}")
    print("    " + "-" * 48)

    try:
        while True:
            val = bus.sync_read(normalize=False, data_name="Present_Position")["gripper"]
            n = normalize_gripper(val, _gripper_open, _gripper_closed)
            n_clamped = max(0.0, min(1.0, n))
            f_rad = map_range(n_clamped, 0.0, 1.0, FOLLOWER_OPEN, FOLLOWER_CLOSED)
            bar_len = int(n_clamped * 30)
            bar = "O" + "=" * bar_len + ">" + " " * (30 - bar_len) + "X"
            print(f"\r    {val:8d}  {n:8.3f}  {f_rad:10.3f}  {bar}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()

except Exception as e:
    print(f"\n*** ERROR: {e} ***")
    traceback.print_exc()
    results["error"] = str(e)

finally:
    # Save results
    try:
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"Could not save results: {e}")

    # Disconnect
    try:
        if bus is not None and bus.is_connected:
            bus.disconnect()
            print("Bus disconnected.")
    except Exception:
        pass
