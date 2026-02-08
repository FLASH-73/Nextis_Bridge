#!/usr/bin/env python3
"""Re-zero a Damiao motor's encoder.

When a motor spins out of control, the encoder position shifts permanently.
This script resets the encoder to 0 at the current physical position.

IMPORTANT: Before running, manually position the arm joint to the desired
zero reference pose (e.g., straight out, or calibration position).

Usage:
    python rezero_motor.py link1          # Re-zero link1 only
    python rezero_motor.py base link1     # Re-zero base and link1
    python rezero_motor.py all            # Re-zero ALL motors (careful!)
"""
import sys
import time

sys.path.insert(0, "lerobot/src")

from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig, DamiaoMotorsBus
from lerobot.motors.damiao.tables import DEFAULT_DAMIAO_MOTORS

ALL_MOTORS = DEFAULT_DAMIAO_MOTORS.copy()


def main():
    if len(sys.argv) < 2:
        print("Usage: python rezero_motor.py <motor_name> [motor_name2 ...]")
        print("       python rezero_motor.py all")
        print("       python rezero_motor.py --force link1  (skip confirmation)")
        print()
        print("Available motors:", ", ".join(ALL_MOTORS.keys()))
        sys.exit(1)

    # Parse --force flag
    args = sys.argv[1:]
    force = "--force" in args
    if force:
        args.remove("--force")

    # Determine which motors to re-zero
    if not args:
        print("ERROR: No motor names specified.")
        sys.exit(1)
    if args[0] == "all":
        target_names = list(ALL_MOTORS.keys())
    else:
        target_names = args
        for name in target_names:
            if name not in ALL_MOTORS:
                print(f"ERROR: Unknown motor '{name}'. Available: {', '.join(ALL_MOTORS.keys())}")
                sys.exit(1)

    config = DamiaoMotorsBusConfig(
        port="can0",
        motors=ALL_MOTORS,
        velocity_limit=0.1,
        use_mit_mode=True,
    )

    bus = DamiaoMotorsBus(config)

    try:
        print("=" * 70)
        print(f"Encoder Re-Zero — {len(target_names)} Motor(s): {', '.join(target_names)}")
        print("=" * 70)

        print("\nConnecting to CAN bus...")
        bus.connect()

        # Phase 1: Read current positions of ALL motors
        print("\n--- Phase 1: Current Encoder Positions ---\n")
        for name in ALL_MOTORS:
            motor = bus._motors[name]
            bus._control.refresh_motor_status(motor)
            time.sleep(0.01)
            pos = motor.getPosition()
            marker = " *** WILL RE-ZERO ***" if name in target_names else ""
            print(f"  {name:8s}: {pos:+10.4f} rad{marker}")

        # Phase 2: Confirm with user
        print(f"\n{'!'*70}")
        print(f"WARNING: This will set the CURRENT physical position as 0.0 rad")
        print(f"for: {', '.join(target_names)}")
        print(f"{'!'*70}")
        print()

        if force:
            print("  --force flag: skipping confirmation")
        else:
            response = input("Type 'YES' to confirm re-zeroing: ")
            if response != "YES":
                print("Aborted.")
                return

        # Phase 3: Re-zero each motor
        print(f"\n--- Phase 3: Re-Zeroing ---\n")
        for name in target_names:
            motor = bus._motors[name]

            # Read position before
            bus._control.refresh_motor_status(motor)
            time.sleep(0.01)
            pos_before = motor.getPosition()

            # Disable -> set zero -> disable (openarm-style, no save_motor_param)
            # The 0xFE command writes to flash on its own.
            # Note: save_motor_param (0xAA) after 0xFE can corrupt the zero.
            bus._control.disable(motor)
            time.sleep(0.05)
            bus._control.set_zero_position(motor)
            time.sleep(0.5)
            bus._control.disable(motor)
            time.sleep(0.1)

            # Read back to verify
            bus._control.refresh_motor_status(motor)
            time.sleep(0.01)
            pos_after = motor.getPosition()

            print(f"  {name:8s}: {pos_before:+10.4f} rad -> {pos_after:+10.4f} rad "
                  f"({'OK' if abs(pos_after) < 0.1 else 'VERIFY'})")

        # Phase 4: Final verification
        print(f"\n--- Phase 4: Final Verification ---\n")
        for name in ALL_MOTORS:
            motor = bus._motors[name]
            bus._control.refresh_motor_status(motor)
            time.sleep(0.01)
            pos = motor.getPosition()
            status = ""
            if name in target_names:
                status = " [RE-ZEROED]" if abs(pos) < 0.1 else " [CHECK!]"
            print(f"  {name:8s}: {pos:+10.4f} rad{status}")

        print(f"\nDone. Re-zeroed {len(target_names)} motor(s).")
        print("You can now run the position test: python test_mit_mode_position.py all")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        print("\nDisabling all motors and disconnecting...")
        bus.disconnect(disable_torque=True)
        print("Done.")


if __name__ == "__main__":
    main()
