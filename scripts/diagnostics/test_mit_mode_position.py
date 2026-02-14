#!/usr/bin/env python3
"""Test MIT mode position control on selected motors.

Uses the DamiaoMotorsBus class with use_mit_mode=True to verify
the full MIT mode code path works: configure -> safe_enable -> sync_write.

Each motor: hold position for 2s, move +0.1 rad, return.

Usage:
    python test_mit_mode_position.py          # motors 1+2 only (J8009P)
    python test_mit_mode_position.py j4340    # motors 3+4 only (J4340P)
    python test_mit_mode_position.py j4310    # motors 5+6+7 only (J4310)
    python test_mit_mode_position.py all      # all 7 motors
"""
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lerobot" / "src"))

from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig, DamiaoMotorsBus

# Motor subsets for safe incremental testing
MOTORS_J8009P = {
    "base":  {"motor_type": "J8009P", "can_id": 0x01, "master_id": 0x11},
    "link1": {"motor_type": "J8009P", "can_id": 0x02, "master_id": 0x12},
}

MOTORS_J4340P = {
    "link2": {"motor_type": "J4340P", "can_id": 0x03, "master_id": 0x13},
    "link3": {"motor_type": "J4340P", "can_id": 0x04, "master_id": 0x14},
}

MOTORS_J4310 = {
    "link4":   {"motor_type": "J4310", "can_id": 0x05, "master_id": 0x15},
    "link5":   {"motor_type": "J4310", "can_id": 0x06, "master_id": 0x16},
    "gripper": {"motor_type": "J4310", "can_id": 0x07, "master_id": 0x17},
}

ALL_MOTORS = {**MOTORS_J8009P, **MOTORS_J4340P, **MOTORS_J4310}

RATE_HZ = 60
HOLD_SECONDS = 2
MOVE_SECONDS = 2
MOVE_OFFSET = 0.1  # radians


def main():
    # Select motor set
    subset = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    MOTOR_SETS = {
        "all": ALL_MOTORS,
        "j4340": MOTORS_J4340P,
        "j4340p": MOTORS_J4340P,
        "j4310": MOTORS_J4310,
        "j8009": MOTORS_J8009P,
        "j8009p": MOTORS_J8009P,
    }
    motors = MOTOR_SETS.get(subset, MOTORS_J8009P)
    motor_names = list(motors.keys())

    config = DamiaoMotorsBusConfig(
        port="can0",
        motors=motors,
        velocity_limit=0.1,
        use_mit_mode=True,
        mit_kp=15.0,
        mit_kd=1.5,
    )

    bus = DamiaoMotorsBus(config)

    try:
        print("=" * 70)
        print(f"MIT Mode Position Test — {len(motors)} Motors: {', '.join(motor_names)}")
        print(f"Gains: kp=15.0, kd=1.5 (same for all motors)")
        print("=" * 70)

        print("\nConnecting to CAN bus...")
        bus.connect()

        print("Configuring motors in MIT mode...")
        bus.configure()

        # Read starting positions
        positions = bus.sync_read("Present_Position")
        start_positions = dict(positions)
        print(f"\nStarting positions:")
        for name, pos in start_positions.items():
            print(f"  {name:8s}: {pos:+.4f} rad")

        # Filter to only motors that responded (quarantined motors are excluded)
        active_names = [n for n in motor_names if n in start_positions]
        skipped_names = [n for n in motor_names if n not in start_positions]
        if skipped_names:
            print(f"\n  SKIPPED (quarantined/unresponsive): {', '.join(skipped_names)}")

        # ─── Phase 1: Hold all motors at current position ───
        print(f"\n--- Phase 1: Hold for {HOLD_SECONDS}s ---")
        frames = int(RATE_HZ * HOLD_SECONDS)
        for i in range(frames):
            bus.sync_write("Goal_Position", start_positions)
            time.sleep(1 / RATE_HZ)
            if i == 0:
                print("  Motors should be holding firm (no vibration).")

        positions = bus.sync_read("Present_Position")
        hold_drift = {}
        for name in active_names:
            hold_drift[name] = abs(positions.get(name, 0) - start_positions[name])

        # ─── Phase 2: Move all motors +0.1 rad ───
        targets = {name: start_positions[name] + MOVE_OFFSET for name in active_names}
        print(f"\n--- Phase 2: Move +{MOVE_OFFSET} rad ---")
        frames = int(RATE_HZ * MOVE_SECONDS)
        for i in range(frames):
            bus.sync_write("Goal_Position", targets)
            time.sleep(1 / RATE_HZ)

        positions = bus.sync_read("Present_Position")
        move_error = {}
        for name in active_names:
            move_error[name] = abs(positions.get(name, 0) - targets[name])

        # ─── Phase 3: Return to start ───
        print(f"\n--- Phase 3: Return to start ---")
        frames = int(RATE_HZ * MOVE_SECONDS)
        for i in range(frames):
            bus.sync_write("Goal_Position", start_positions)
            time.sleep(1 / RATE_HZ)

        positions = bus.sync_read("Present_Position")
        return_error = {}
        for name in active_names:
            return_error[name] = abs(positions.get(name, 0) - start_positions[name])

        # ─── Summary ───
        print(f"\n{'=' * 70}")
        print(f"TEST SUMMARY — {len(active_names)}/{len(motors)} Motors (kp=15.0, kd=1.5)")
        print(f"{'=' * 70}")
        print(f"  {'Motor':8s}  {'Type':8s}  {'Hold Drift':>11s}  {'Move Err':>11s}  {'Return Err':>11s}  {'Result':>6s}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*6}")

        all_pass = True
        for name in active_names:
            hd = hold_drift[name]
            me = move_error[name]
            re = return_error[name]
            mtype = motors[name]["motor_type"]
            ok = hd < 0.02 and me < 0.1 and re < 0.1
            if not ok:
                all_pass = False
            print(f"  {name:8s}  {mtype:8s}  {hd:>9.4f} r  {me:>9.4f} r  {re:>9.4f} r  {'PASS' if ok else 'WARN':>6s}")

        for name in skipped_names:
            mtype = motors[name]["motor_type"]
            print(f"  {name:8s}  {mtype:8s}  {'---':>11s}  {'---':>11s}  {'---':>11s}  {'SKIP':>6s}")

        if skipped_names:
            all_pass = False
        print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME WARNINGS/SKIPPED'}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        print("\nDisabling all motors and disconnecting...")
        bus.disconnect(disable_torque=True)
        print("Done.")


if __name__ == "__main__":
    main()
