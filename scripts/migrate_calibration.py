#!/usr/bin/env python3
"""Migrate calibration files from HuggingFace cache to project-local calibration_profiles/.

Scans ~/.cache/huggingface/lerobot/calibration/ for .json calibration files and
copies them to calibration_profiles/ (the project-local single source of truth).

Usage:
    python scripts/migrate_calibration.py              # Dry-run (show what would happen)
    python scripts/migrate_calibration.py --apply      # Actually copy files
    python scripts/migrate_calibration.py --delete-hf  # Copy + delete HF cache originals
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration_profiles"

# HF cache calibration path (same logic as lerobot/utils/constants.py)
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", HF_HOME / "lerobot")).expanduser()
HF_CALIBRATION = Path(os.environ.get("HF_LEROBOT_CALIBRATION", HF_LEROBOT_HOME / "calibration")).expanduser()


def find_hf_calibration_files():
    """Find all .json calibration files in the HF cache."""
    if not HF_CALIBRATION.exists():
        return []

    files = []
    for json_file in HF_CALIBRATION.rglob("*.json"):
        # relative path from HF_CALIBRATION, e.g., robots/damiao_follower/aira_zero.json
        rel = json_file.relative_to(HF_CALIBRATION)
        parts = rel.parts
        # Expected structure: {category}/{type_name}/{id}.json
        # e.g., robots/damiao_follower/aira_zero.json
        #        teleoperators/dynamixel_leader/aira_zero_leader.json
        if len(parts) >= 2:
            # Map to calibration_profiles/{type_name}/{id}.json
            # (strip the robots/ or teleoperators/ prefix)
            type_name = parts[-2]
            filename = parts[-1]
            dest = CALIBRATION_DIR / type_name / filename
        else:
            # Flat file directly in calibration/ — use filename as subdir
            filename = parts[-1]
            dest = CALIBRATION_DIR / filename

        files.append({
            "source": json_file,
            "dest": dest,
            "rel": str(rel),
        })

    return files


def files_are_identical(path_a, path_b):
    """Compare two JSON files for semantic equality."""
    try:
        with open(path_a) as f:
            data_a = json.load(f)
        with open(path_b) as f:
            data_b = json.load(f)
        return data_a == data_b
    except (json.JSONDecodeError, OSError):
        return False


def show_diff(path_a, path_b):
    """Print differences between two calibration JSON files."""
    try:
        with open(path_a) as f:
            data_a = json.load(f)
        with open(path_b) as f:
            data_b = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"    Could not compare: {e}")
        return

    all_motors = set(list(data_a.keys()) + list(data_b.keys()))
    for motor in sorted(all_motors):
        cal_a = data_a.get(motor, {})
        cal_b = data_b.get(motor, {})
        if cal_a != cal_b:
            print(f"    Motor '{motor}':")
            all_fields = set(list(cal_a.keys()) + list(cal_b.keys()))
            for field in sorted(all_fields):
                va = cal_a.get(field, "<missing>")
                vb = cal_b.get(field, "<missing>")
                if va != vb:
                    print(f"      {field}: HF={va}  project={vb}")


def main():
    parser = argparse.ArgumentParser(description="Migrate calibration from HF cache to calibration_profiles/")
    parser.add_argument("--apply", action="store_true", help="Actually copy files (default is dry-run)")
    parser.add_argument("--delete-hf", action="store_true", help="Delete HF cache originals after copying")
    args = parser.parse_args()

    print(f"HF calibration cache: {HF_CALIBRATION}")
    print(f"Project calibration:  {CALIBRATION_DIR}")
    print()

    if not HF_CALIBRATION.exists():
        print("No HF calibration cache found. Nothing to migrate.")
        return

    files = find_hf_calibration_files()
    if not files:
        print("No .json calibration files found in HF cache.")
        return

    print(f"Found {len(files)} calibration file(s) in HF cache:\n")

    to_copy = []
    already_exists_identical = []
    already_exists_different = []

    for entry in files:
        src = entry["source"]
        dst = entry["dest"]
        rel = entry["rel"]

        if dst.exists():
            if files_are_identical(src, dst):
                already_exists_identical.append(entry)
                print(f"  [IDENTICAL] {rel}")
                print(f"              -> {dst.relative_to(PROJECT_ROOT)}")
            else:
                already_exists_different.append(entry)
                print(f"  [DIFFERS]   {rel}")
                print(f"              -> {dst.relative_to(PROJECT_ROOT)}")
                show_diff(src, dst)
        else:
            to_copy.append(entry)
            print(f"  [NEW]       {rel}")
            print(f"              -> {dst.relative_to(PROJECT_ROOT)}")

    print(f"\nSummary:")
    print(f"  New files to copy:      {len(to_copy)}")
    print(f"  Already identical:      {len(already_exists_identical)}")
    print(f"  Already exist (differ): {len(already_exists_different)}")

    if already_exists_different:
        print(f"\n  WARNING: {len(already_exists_different)} file(s) differ between HF cache and project.")
        print(f"  Review the differences above. The project version will be kept.")

    if not args.apply and not args.delete_hf:
        print(f"\nDry run. Use --apply to copy files, --delete-hf to also remove HF originals.")
        return

    # Copy new files
    if to_copy:
        print(f"\nCopying {len(to_copy)} file(s)...")
        for entry in to_copy:
            dst = entry["dest"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry["source"], dst)
            print(f"  Copied: {entry['rel']} -> {dst.relative_to(PROJECT_ROOT)}")
    else:
        print("\nNo new files to copy.")

    # Delete HF cache originals
    if args.delete_hf:
        all_entries = to_copy + already_exists_identical
        if already_exists_different:
            print(f"\nSkipping deletion of {len(already_exists_different)} differing file(s) — review manually.")
        if all_entries:
            print(f"\nDeleting {len(all_entries)} HF cache original(s)...")
            for entry in all_entries:
                entry["source"].unlink()
                print(f"  Deleted: {entry['source']}")

            # Clean up empty directories
            for dirpath in sorted(HF_CALIBRATION.rglob("*"), reverse=True):
                if dirpath.is_dir() and not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    print(f"  Removed empty dir: {dirpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
