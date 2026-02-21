#!/usr/bin/env python3
"""CLI tool for viewing and editing calibration profiles.

Usage:
    python scripts/edit_calibration.py list
    python scripts/edit_calibration.py show <arm_id> [--profile <name>]
    python scripts/edit_calibration.py edit <arm_id> <motor> <field> <value> [--profile <name>]
    python scripts/edit_calibration.py diff <arm_id> <profile_a> <profile_b>

Examples:
    python scripts/edit_calibration.py list
    python scripts/edit_calibration.py show aira_zero
    python scripts/edit_calibration.py show aira_zero --profile cal_test_7_02_1821
    python scripts/edit_calibration.py edit aira_zero base range_min -1.6
    python scripts/edit_calibration.py diff aira_zero cal_test_7_02_1821 cal_v2
"""

import argparse
import json
import sys
from pathlib import Path

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

CALIBRATION_DIR = PROJECT_ROOT / "calibration_profiles"

VALID_FIELDS = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}


def load_calibration(fpath: Path) -> dict:
    """Load a calibration JSON file using draccus for consistency."""
    try:
        import draccus
        from lerobot.motors.motors_bus import MotorCalibration
        with open(fpath) as f, draccus.config_type("json"):
            return draccus.load(dict[str, MotorCalibration], f)
    except ImportError:
        # Fallback to plain JSON if draccus not available
        with open(fpath) as f:
            return json.load(f)


def save_calibration(fpath: Path, calibration: dict):
    """Save a calibration dict using draccus for consistency."""
    try:
        import draccus
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(calibration, f, indent=4)
    except ImportError:
        # Fallback to plain JSON
        with open(fpath, "w") as f:
            json.dump(calibration, f, indent=4)


def cal_to_dict(cal) -> dict:
    """Convert a MotorCalibration (or dict) to a plain dict."""
    if hasattr(cal, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(cal)
    return dict(cal)


def find_default_profile(arm_dir: Path) -> Path | None:
    """Find the most recently modified .json file in the arm directory."""
    json_files = list(arm_dir.glob("*.json"))
    # Exclude inversions.json
    json_files = [f for f in json_files if f.stem != "inversions"]
    if not json_files:
        return None
    return max(json_files, key=lambda f: f.stat().st_mtime)


def resolve_profile(arm_id: str, profile_name: str | None) -> Path:
    """Resolve a profile name to a file path."""
    arm_dir = CALIBRATION_DIR / arm_id
    if not arm_dir.exists():
        print(f"Error: No calibration directory for '{arm_id}'")
        print(f"  Expected: {arm_dir}")
        sys.exit(1)

    if profile_name:
        fpath = arm_dir / f"{profile_name}.json"
        if not fpath.exists():
            print(f"Error: Profile '{profile_name}' not found for '{arm_id}'")
            print(f"  Expected: {fpath}")
            available = [f.stem for f in arm_dir.glob("*.json") if f.stem != "inversions"]
            if available:
                print(f"  Available: {', '.join(available)}")
            sys.exit(1)
        return fpath
    else:
        fpath = find_default_profile(arm_dir)
        if not fpath:
            print(f"Error: No calibration profiles found for '{arm_id}'")
            sys.exit(1)
        return fpath


def cmd_list(args):
    """List all calibration directories and their profiles."""
    if not CALIBRATION_DIR.exists():
        print(f"No calibration_profiles/ directory found at {CALIBRATION_DIR}")
        return

    arm_dirs = sorted(d for d in CALIBRATION_DIR.iterdir() if d.is_dir())
    if not arm_dirs:
        print("No arm calibration directories found.")
        return

    for arm_dir in arm_dirs:
        profiles = sorted(
            f for f in arm_dir.glob("*.json") if f.stem != "inversions"
        )
        inv_file = arm_dir / "inversions.json"
        inv_label = " [+inversions]" if inv_file.exists() else ""

        print(f"{arm_dir.name}/{inv_label}")
        if profiles:
            for p in profiles:
                import datetime
                mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
                print(f"  {p.stem:30s}  {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"  (no profiles)")
        print()


def cmd_show(args):
    """Show calibration data as a table."""
    fpath = resolve_profile(args.arm_id, args.profile)
    calibration = load_calibration(fpath)

    print(f"Profile: {fpath.relative_to(PROJECT_ROOT)}")
    print()

    # Header
    print(f"{'Motor':<15s} {'ID':>4s} {'DrvMode':>7s} {'HomingOff':>12s} {'RangeMin':>12s} {'RangeMax':>12s}")
    print("-" * 68)

    for motor_name, cal in calibration.items():
        d = cal_to_dict(cal)
        motor_id = d.get("id", "?")
        drive_mode = d.get("drive_mode", "?")
        homing_offset = d.get("homing_offset", "?")
        range_min = d.get("range_min", "?")
        range_max = d.get("range_max", "?")

        # Format floats nicely
        ho_str = f"{homing_offset}" if isinstance(homing_offset, int) else f"{homing_offset:.6f}"
        rmin_str = f"{range_min}" if isinstance(range_min, int) else f"{range_min:.6f}"
        rmax_str = f"{range_max}" if isinstance(range_max, int) else f"{range_max:.6f}"

        print(f"{motor_name:<15s} {motor_id:>4} {drive_mode:>7} {ho_str:>12s} {rmin_str:>12s} {rmax_str:>12s}")


def cmd_edit(args):
    """Edit a single field for a single motor."""
    field = args.field
    if field not in VALID_FIELDS:
        print(f"Error: Invalid field '{field}'. Valid fields: {', '.join(sorted(VALID_FIELDS))}")
        sys.exit(1)

    fpath = resolve_profile(args.arm_id, args.profile)
    calibration = load_calibration(fpath)

    motor = args.motor
    if motor not in calibration:
        print(f"Error: Motor '{motor}' not found. Available: {', '.join(calibration.keys())}")
        sys.exit(1)

    cal = calibration[motor]
    d = cal_to_dict(cal)
    old_value = d.get(field)

    # Parse the new value to the appropriate type
    raw_value = args.value
    if field in ("id", "drive_mode"):
        new_value = int(raw_value)
    elif field == "homing_offset":
        # Could be int (Dynamixel/Feetech) or float (Damiao)
        new_value = int(raw_value) if "." not in raw_value else float(raw_value)
    else:
        # range_min, range_max
        new_value = int(raw_value) if "." not in raw_value else float(raw_value)

    # Apply the change
    if hasattr(cal, field):
        setattr(cal, field, new_value)
    else:
        cal[field] = new_value

    print(f"Profile: {fpath.relative_to(PROJECT_ROOT)}")
    print(f"Motor:   {motor}")
    print(f"Field:   {field}")
    print(f"Before:  {old_value}")
    print(f"After:   {new_value}")

    save_calibration(fpath, calibration)
    print(f"\nSaved to {fpath.relative_to(PROJECT_ROOT)}")


def cmd_diff(args):
    """Compare two profiles for the same arm."""
    arm_dir = CALIBRATION_DIR / args.arm_id
    if not arm_dir.exists():
        print(f"Error: No calibration directory for '{args.arm_id}'")
        sys.exit(1)

    fpath_a = arm_dir / f"{args.profile_a}.json"
    fpath_b = arm_dir / f"{args.profile_b}.json"

    for label, fp in [("A", fpath_a), ("B", fpath_b)]:
        if not fp.exists():
            print(f"Error: Profile {label} not found: {fp}")
            sys.exit(1)

    cal_a = load_calibration(fpath_a)
    cal_b = load_calibration(fpath_b)

    print(f"A: {fpath_a.relative_to(PROJECT_ROOT)}")
    print(f"B: {fpath_b.relative_to(PROJECT_ROOT)}")
    print()

    all_motors = sorted(set(list(cal_a.keys()) + list(cal_b.keys())))
    has_diff = False

    for motor in all_motors:
        if motor not in cal_a:
            print(f"  {motor}: only in B")
            has_diff = True
            continue
        if motor not in cal_b:
            print(f"  {motor}: only in A")
            has_diff = True
            continue

        da = cal_to_dict(cal_a[motor])
        db = cal_to_dict(cal_b[motor])

        diffs = []
        for field in sorted(set(list(da.keys()) + list(db.keys()))):
            va = da.get(field)
            vb = db.get(field)
            if va != vb:
                diffs.append((field, va, vb))

        if diffs:
            has_diff = True
            print(f"  {motor}:")
            for field, va, vb in diffs:
                print(f"    {field}: A={va}  B={vb}")

    if not has_diff:
        print("  Profiles are identical.")


def main():
    parser = argparse.ArgumentParser(
        description="View and edit calibration profiles in calibration_profiles/"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List all calibration directories and profiles")

    # show
    show_parser = subparsers.add_parser("show", help="Show calibration data as a table")
    show_parser.add_argument("arm_id", help="Arm identifier (e.g., aira_zero, umbra_follower)")
    show_parser.add_argument("--profile", default=None, help="Profile name (default: most recent)")

    # edit
    edit_parser = subparsers.add_parser("edit", help="Edit a single field for a motor")
    edit_parser.add_argument("arm_id", help="Arm identifier")
    edit_parser.add_argument("motor", help="Motor name (e.g., base, link1, gripper)")
    edit_parser.add_argument("field", help=f"Field to edit: {', '.join(sorted(VALID_FIELDS))}")
    edit_parser.add_argument("value", help="New value")
    edit_parser.add_argument("--profile", default=None, help="Profile name (default: most recent)")

    # diff
    diff_parser = subparsers.add_parser("diff", help="Compare two profiles")
    diff_parser.add_argument("arm_id", help="Arm identifier")
    diff_parser.add_argument("profile_a", help="First profile name")
    diff_parser.add_argument("profile_b", help="Second profile name")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "show": cmd_show,
        "edit": cmd_edit,
        "diff": cmd_diff,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
