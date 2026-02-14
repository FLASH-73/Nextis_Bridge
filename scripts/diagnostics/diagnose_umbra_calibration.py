#!/usr/bin/env python3
"""
Diagnostic tool for Umbra leader↔follower calibration.

Captures position data from both arms at matching physical poses
to identify mapping errors (wrong range, wrong inversion, offset, etc.).

Usage:
    Move BOTH arms to the same physical position, press Enter.
    Repeat for multiple poses. Results saved to tools/calibration_diagnosis.json.

Run with:
    /home/roberto/miniconda3/bin/python3 tools/diagnose_umbra_calibration.py
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root and lerobot/src to path (same as app/main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

from app.core.arm_registry import ArmRegistryService

# Joint mapping: leader name → follower name
JOINT_MAP = {
    "joint_1": "base",
    "joint_2": "link1",
    "joint_3": "link2",
    "joint_4": "link3",
    "joint_5": "link4",
    "joint_6": "link5",
    "gripper": "gripper",
}

LEADER_ID = "umbra_leader"
FOLLOWER_ID = "umbra_follower"
OUTPUT_FILE = PROJECT_ROOT / "tools" / "calibration_diagnosis.json"


def load_leader_calibration(leader):
    """Load leader's calibration ranges from its calibration data."""
    cal_ranges = {}
    if hasattr(leader, 'calibration') and leader.calibration:
        for joint_name, cal in leader.calibration.items():
            if joint_name != "gripper":
                cal_ranges[joint_name] = {
                    "range_min": cal.range_min,
                    "range_max": cal.range_max,
                    "homing_offset": cal.homing_offset,
                }
    return cal_ranges


def load_follower_calibration(follower):
    """Load follower's calibration ranges."""
    cal_ranges = {}
    if hasattr(follower, 'calibration') and follower.calibration:
        for motor_name, cal in follower.calibration.items():
            cal_ranges[motor_name] = {
                "range_min": cal.range_min,
                "range_max": cal.range_max,
                "homing_offset": cal.homing_offset,
            }
    return cal_ranges


def load_inversions():
    """Load follower inversions from disk."""
    inv_path = PROJECT_ROOT / "calibration_profiles" / "umbra_follower" / "inversions.json"
    if inv_path.exists():
        with open(inv_path) as f:
            return json.load(f)
    return {}


def leader_rad_to_pct(rad_value, cal_range):
    """Convert leader radians to ±100% using leader's calibration range.
    This is the same formula used in teleop_service.py absolute mapping (with unwrap)."""
    rmin = cal_range["range_min"]
    rmax = cal_range["range_max"]
    ticks = (rad_value + np.pi) * 4096.0 / (2 * np.pi)
    # Unwrap: if homed ticks crossed the 0/4096 encoder boundary (same as teleop_service.py)
    center = (rmin + rmax) * 0.5
    while ticks < center - 2048:
        ticks += 4096
    while ticks > center + 2048:
        ticks -= 4096
    pct = ((ticks - rmin) / (rmax - rmin)) * 200 - 100
    return ticks, pct


def capture_pose(leader, follower, leader_cal, follower_cal, inversions):
    """Capture current position from both arms."""
    # --- Leader ---
    leader_raw = leader.bus.sync_read(normalize=False, data_name="Present_Position")
    leader_action = leader.get_action()

    # --- Follower ---
    follower_raw = follower.bus.sync_read("Present_Position", normalize=False)
    follower_obs = follower.get_observation(include_images=False)

    pose = {}
    for l_joint, f_joint in JOINT_MAP.items():
        l_key = f"{l_joint}.pos"
        f_key = f"{f_joint}.pos"

        l_raw_ticks = leader_raw.get(l_joint, None)
        l_rad = leader_action.get(l_key, None)
        f_raw_ticks = follower_raw.get(f_joint, None)
        f_pct = follower_obs.get(f_key, None)
        f_inverted = inversions.get(f_joint, False)

        entry = {
            "leader_joint": l_joint,
            "follower_joint": f_joint,
            "leader_raw_ticks": int(l_raw_ticks) if l_raw_ticks is not None else None,
            "leader_radians": float(l_rad) if l_rad is not None else None,
            "follower_raw_ticks": int(f_raw_ticks) if f_raw_ticks is not None else None,
            "follower_pct": float(f_pct) if f_pct is not None else None,
            "follower_inverted": f_inverted,
        }

        if l_joint == "gripper":
            entry["leader_gripper_01"] = float(l_rad) if l_rad is not None else None
            entry["leader_computed_pct"] = float(l_rad * 100.0) if l_rad is not None else None
        elif l_joint in leader_cal and l_rad is not None:
            ticks, pct = leader_rad_to_pct(l_rad, leader_cal[l_joint])
            entry["leader_homed_ticks"] = float(ticks)
            entry["leader_computed_pct"] = float(pct)
            entry["leader_cal_range_min"] = leader_cal[l_joint]["range_min"]
            entry["leader_cal_range_max"] = leader_cal[l_joint]["range_max"]
        else:
            entry["leader_computed_pct"] = None

        if f_joint in follower_cal:
            entry["follower_cal_range_min"] = follower_cal[f_joint]["range_min"]
            entry["follower_cal_range_max"] = follower_cal[f_joint]["range_max"]

        # Delta: what teleop sends vs what follower reads
        if entry["leader_computed_pct"] is not None and entry["follower_pct"] is not None:
            entry["delta_pct"] = round(entry["leader_computed_pct"] - entry["follower_pct"], 2)
        else:
            entry["delta_pct"] = None

        pose[l_joint] = entry

    return pose


def print_pose_table(pose, pose_num):
    """Print a formatted comparison table for one pose."""
    print(f"\n{'='*90}")
    print(f"  POSE {pose_num}")
    print(f"{'='*90}")
    header = f"{'Joint':<10} {'L_ticks':>8} {'L_rad':>8} {'L→pct':>8} {'F_ticks':>8} {'F_pct':>8} {'Delta':>8} {'Inv':>4}"
    print(header)
    print("-" * 90)

    for l_joint in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]:
        e = pose[l_joint]
        l_ticks = f"{e['leader_raw_ticks']:>8}" if e['leader_raw_ticks'] is not None else "    N/A"
        l_rad = f"{e['leader_radians']:>8.3f}" if e['leader_radians'] is not None else "     N/A"
        l_pct = f"{e['leader_computed_pct']:>8.1f}" if e['leader_computed_pct'] is not None else "     N/A"
        f_ticks = f"{e['follower_raw_ticks']:>8}" if e['follower_raw_ticks'] is not None else "     N/A"
        f_pct = f"{e['follower_pct']:>8.1f}" if e['follower_pct'] is not None else "     N/A"
        delta = f"{e['delta_pct']:>8.1f}" if e['delta_pct'] is not None else "     N/A"
        inv = " yes" if e['follower_inverted'] else "  no"

        label = f"{l_joint}→{e['follower_joint']}"
        print(f"{label:<16} {l_ticks} {l_rad} {l_pct} {f_ticks} {f_pct} {delta} {inv}")

    print()


def print_summary(captures):
    """Print a summary across all captures."""
    if len(captures) < 2:
        return

    print(f"\n{'='*90}")
    print(f"  SUMMARY: Range of movement between poses")
    print(f"{'='*90}")
    header = f"{'Joint':<16} {'L_pct range':>14} {'F_pct range':>14} {'Avg delta':>12} {'Sign match':>12}"
    print(header)
    print("-" * 90)

    for l_joint in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]:
        l_pcts = [c[l_joint]["leader_computed_pct"] for c in captures if c[l_joint]["leader_computed_pct"] is not None]
        f_pcts = [c[l_joint]["follower_pct"] for c in captures if c[l_joint]["follower_pct"] is not None]
        deltas = [c[l_joint]["delta_pct"] for c in captures if c[l_joint]["delta_pct"] is not None]

        f_joint = JOINT_MAP[l_joint]
        label = f"{l_joint}→{f_joint}"

        if len(l_pcts) >= 2 and len(f_pcts) >= 2:
            l_range = max(l_pcts) - min(l_pcts)
            f_range = max(f_pcts) - min(f_pcts)
            avg_delta = sum(deltas) / len(deltas)

            # Check if leader and follower move in the same direction
            l_diff = l_pcts[-1] - l_pcts[0]
            f_diff = f_pcts[-1] - f_pcts[0]
            same_sign = "OK" if (l_diff * f_diff > 0) else "INVERTED!"

            print(f"{label:<16} {l_range:>14.1f} {f_range:>14.1f} {avg_delta:>12.1f} {same_sign:>12}")
        else:
            print(f"{label:<16} {'N/A':>14} {'N/A':>14} {'N/A':>12} {'N/A':>12}")

    print()


def main():
    print("=" * 60)
    print("  Umbra Calibration Diagnostic Tool")
    print("=" * 60)
    print()
    print("This tool captures position data from both the leader")
    print("and follower arms at matching physical poses.")
    print()
    print("Instructions:")
    print("  1. Move BOTH arms to the same physical position")
    print("  2. Press Enter to capture")
    print("  3. Repeat for multiple poses (type 'done' to finish)")
    print()

    # Connect arms via arm_registry
    config_path = PROJECT_ROOT / "app" / "config" / "settings.yaml"
    registry = ArmRegistryService(str(config_path))

    print(f"Connecting {LEADER_ID}...")
    result = registry.connect_arm(LEADER_ID)
    if not result["success"]:
        print(f"FAILED to connect leader: {result['error']}")
        sys.exit(1)
    leader = registry.arm_instances[LEADER_ID]
    print(f"  Connected: {type(leader).__name__}")

    print(f"Connecting {FOLLOWER_ID}...")
    result = registry.connect_arm(FOLLOWER_ID)
    if not result["success"]:
        print(f"FAILED to connect follower: {result['error']}")
        registry.disconnect_arm(LEADER_ID)
        sys.exit(1)
    follower = registry.arm_instances[FOLLOWER_ID]
    print(f"  Connected: {type(follower).__name__}")

    # Disable follower torque so user can move it freely
    if hasattr(follower, 'bus'):
        try:
            follower.bus.disable_torque()
            print("  Follower torque disabled (free to move)")
        except Exception as e:
            print(f"  Warning: could not disable torque: {e}")

    # Load calibration data
    leader_cal = load_leader_calibration(leader)
    follower_cal = load_follower_calibration(follower)
    inversions = load_inversions()

    print(f"\nLeader calibration: {len(leader_cal)} joints")
    for j, c in leader_cal.items():
        print(f"  {j}: range=[{c['range_min']}, {c['range_max']}] (span={c['range_max']-c['range_min']})")

    print(f"\nFollower calibration: {len(follower_cal)} joints")
    for j, c in follower_cal.items():
        print(f"  {j}: range=[{c['range_min']}, {c['range_max']}] (span={c['range_max']-c['range_min']})")

    print(f"\nFollower inversions: {inversions}")
    print()

    # Capture loop
    captures = []
    pose_num = 0

    try:
        while True:
            pose_num += 1
            user_input = input(f"[Pose {pose_num}] Move both arms to matching position, press Enter (or 'done'): ")
            if user_input.strip().lower() in ('done', 'quit', 'exit', 'q'):
                break

            pose = capture_pose(leader, follower, leader_cal, follower_cal, inversions)
            captures.append(pose)
            print_pose_table(pose, pose_num)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    # Summary
    if captures:
        print_summary(captures)

        # Save to JSON
        output = {
            "timestamp": datetime.now().isoformat(),
            "leader_id": LEADER_ID,
            "follower_id": FOLLOWER_ID,
            "leader_calibration": leader_cal,
            "follower_calibration": follower_cal,
            "inversions": inversions,
            "poses": captures,
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {OUTPUT_FILE}")
    else:
        print("No poses captured.")

    # Disconnect
    print("\nDisconnecting...")
    registry.disconnect_arm(FOLLOWER_ID)
    registry.disconnect_arm(LEADER_ID)
    print("Done.")


if __name__ == "__main__":
    main()
